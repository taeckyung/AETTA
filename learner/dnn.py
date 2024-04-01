from collections import deque
from typing import Union, Tuple, Optional

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms.functional import to_pil_image, rotate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from copy import deepcopy

import conf
from data_loader.OutDistDataset import OUTDIST_CLASS_IDX
from data_loader.data_loader import load_cache, save_cache
from models.ResNet import ResNetDropout50, ResNetDropout18
from utils.augmentation import _augmix_aug
from utils.cotta_utils import get_tta_transforms
from utils import memory
from utils.logging import *
from utils.loss_functions import softmax_entropy, calc_energy, JSD
from utils.memory import FIFO, ConfFIFO, HUS, Uniform, PBRS
from utils.memory_rotta import CSTU
from utils.normalize_layer import *
from utils.sam_optimizer import SAM
from utils import memory_rotta
import utils.reset_utils as reset_utils


device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

TRAINED = 0
SKIPPED = 1
FINISHED = 2


class DNN():
    def __init__(self, model_, corruption_list_=None):
        self.device = device
        
        if conf.args.dataset in ['cifar10', 'cifar100'] and conf.args.tgt_train_dist == 0:
            self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        else:
            self.tgt_train_dist = conf.args.tgt_train_dist
            
        ################## Init & prepare model###################
        self.conf_list = []
   
        # Load model
        
        if "pretrained" in conf.args.model:
            pretrained = model_(pretrained=True)
            if conf.args.model == "resnet18_pretrained":
                model = ResNetDropout18()
                model.load_state_dict(pretrained.state_dict())
            elif conf.args.model == "resnet50_pretrained":
                model = ResNetDropout50()
                model.load_state_dict(pretrained.state_dict())
            else:
                raise NotImplementedError
            del pretrained
        else:
            model = model_()

        if 'resnet' in conf.args.model:
            if conf.args.dataset not in ['imagenet', 'imagenetoutdist', 'imagenetA', 'imagenetR']: # imagenetA and imagenetR keeps predicted classes to 1000
                num_feats = model.fc.in_features
                if conf.args.outdist == "divide":
                    num_class = conf.args.opt['num_class'] - len(conf.args.outdist_class)
                else:
                    num_class = conf.args.opt['num_class']
                model.fc = nn.Linear(num_feats, num_class)  # match class number
            self.net = model
        else:
            self.net = model.Net()

        if conf.args.load_checkpoint_path:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        norm_layer = get_normalize_layer(conf.args.dataset)

        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)
            
        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.net.to(device)

        ##########################################################

        # Important: some TTA methods would overwrite this optimizer
        if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10outdist', 'cifar100outdist', 'tinyimagenet', 'imagenet', 'imagenetoutdist', 'imagenetA', 'imagenetR']:
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                conf.args.opt['learning_rate'],
                momentum=conf.args.opt['momentum'],
                weight_decay=conf.args.opt['weight_decay'],
                nesterov=True)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                        weight_decay=conf.args.opt['weight_decay'])

        self.class_criterion = nn.CrossEntropyLoss()
        
        # online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'HUS':
            self.mem = memory.HUS(capacity=conf.args.memory_size, threshold=conf.args.high_threshold, num_class=conf.args.opt['num_class'])
        elif conf.args.memory_type == 'CSTU':
            self.mem = memory_rotta.CSTU(capacity=conf.args.memory_size, num_class=conf.args.opt['num_class'],
                                         lambda_t=1, lambda_u=1)  # replace memory with original RoTTA
        elif conf.args.memory_type == 'ConfFIFO':
            self.mem = memory.ConfFIFO(capacity=conf.args.memory_size, threshold=conf.args.high_threshold)

        self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

        self.fifo = memory.FIFO(conf.args.update_every_x)
        self.previous_train_loss = 0
        self.mem_state = self.mem.save_state_dict()

        self.est_ema_dropout = None

        self.sa_src_net = deepcopy(self.net)
        self.sa_ref_net_1 = None

    def init_json(self, log_path):
        self.write_path = log_path
        self.eval_json = {
            k: [] for k in ['gt', 'accuracy', 'current_accuracy', 'f1_macro', 'distance_l2',
                            'pred', 'pred_outdist', 'pred_total',
                            'confidence', 'confidence_outdist', 'confidence_total',
                            'entropy', 'entropy_outdist', 'entropy_total',
                            'ood_pred', 'ood_gt',
                            'gt_outdist', 'energy', 'energy_outdist',
                            'grad', 'grad_outdist', 'grad_total',
                            'embedding_dist', 'cp_embedding_dist',
                            'src_gt', 'src_pred', 'src_accuracy', 'src_f1_macro', 'src_distance_l2',
                            'src_confidence', 'src_entropy', 'src_energy', 'src_grad', 'src_embedding_dist',
                            'num_reset']
        }
        self.acc_est_json = {
            k: [] for k in ['adv_perturb', 'src_validation', 'gde', 'softmax_score',
                            'est_dropout', 'est_dropout_softmax_mean', 'est_dropout_softmax_std',
                            'aetta', 'est_dropout_avg_entropy']
        }
 
    def set_target_data(self, source_data_loader, source_val_data_loader, target_data_loader, corruption):
        self.source_dataloader = source_data_loader
        self.source_val_dataloader = source_val_data_loader
        self.target_dataloader = target_data_loader

        dataset = conf.args.dataset
        cond = corruption

        filename = f"{dataset}_{conf.args.outdist}_{conf.args.outdist_size}_{conf.args.outdist_class}_{conf.args.seed}_dist{conf.args.tgt_train_dist}"

        if conf.args.tgt_train_dist == 4:
            filename += f"_gamma{conf.args.dirichlet_beta}"
            
        file_path = conf.args.opt['file_path'] + "_target_train_set"

        self.target_train_set = load_cache(filename, cond, file_path, transform=None)

        if not self.target_train_set:
            self.target_data_processing()
            save_cache(self.target_train_set, filename, cond, file_path, transform=None)

        if conf.args.method == 'Src':
            if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10outdist', 'cifar100outdist']:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=conf.args.epoch *
                                                                                  len(self.source_dataloader['train']))
        
    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):
            #must be loaded from dataloader, due to transform in the __getitem__()
            
            features.append(feat.squeeze(0))
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        tgt_train_dist_ = self.tgt_train_dist
        # real distribution
        if tgt_train_dist_ == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        elif tgt_train_dist_ == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        else:
            raise NotImplementedError

        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            try:
                self.target_train_set = (torch.stack(result_feats),
                                        result_cl_labels,
                                        torch.stack(result_do_labels))
            except: # for dataset which each image has different shape
                self.target_train_set = (result_feats,
                                        result_cl_labels,
                                        torch.stack(result_do_labels))

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0],NormalizeLayer):
                cp = self.net[1]
        else:
            cp = self.net

        if isinstance(self.net, nn.DataParallel):
            cp = self.net.module

        torch.save(cp.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        self.net.load_state_dict(self.checkpoint, strict=True)
        self.net.to(device)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):

        if condition == 'train_online':
            # print loss
            print('{:s}: [current_sample: {:d}]'.format(
                condition, epoch
            ))
        else:
            # print loss
            print('{:s}: [epoch: {:d}]\tLoss: {:.6f} \t'.format(
                condition, epoch, loss_avg
            ))

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()

        class_loss_sum = 0.0

        total_iter = 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)

                # compute the feature
                preds = self.net(feats)
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()
                if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10outdist', 'cifar100outdist']:
                    self.scheduler.step()

        # Logging
        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')

    def evaluation(self, epoch, condition):
        # Evaluate with a batch of samples, which is a typical way of evaluation. Used for pre-training or offline eval.

        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs) if conf.args.opt['indices_in_1k'] == None else self.net(tgt_inputs)[:, conf.args.opt['indices_in_1k']]

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)

        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, current_samples):
        # Evaluate with online samples that come one by one while keeping the order.
        self.net.eval()

        with torch.no_grad():  # we don't log grad here
            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples
            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            self.evaluation_online_body(epoch, current_samples, feats, cls, dls)

    def model_inference(self, feats, net=None, temp=1.0):
        if net is None:
            net = self.net

        net.eval()

        # Normalization layer: self.net[0] / ResNet: self.net[1]
        y_logit = net(feats) if conf.args.opt['indices_in_1k'] == None else net(feats)[:, conf.args.opt['indices_in_1k']]
            
        y_entropy: torch.Tensor = softmax_entropy(y_logit)
        y_pred_softmax: torch.Tensor = F.softmax(y_logit/temp, dim=1)
        y_conf: torch.Tensor = y_pred_softmax.max(1, keepdim=False)[0]
        y_energy: torch.Tensor = calc_energy(y_logit).cpu()
        y_pred: torch.Tensor = y_logit.max(1, keepdim=False)[1]
        
        return y_pred, y_conf, y_entropy, y_energy, None, y_pred_softmax, y_logit
        
    def evaluation_online_body(self, epoch, current_samples, feats, cls, dls):
        # get lists from json
        true_cls_list = self.eval_json['gt']
        pred_cls_list = self.eval_json['pred']
        accuracy_list = self.eval_json['accuracy']
        conf_list = self.eval_json['confidence']
        current_accuracy_list = self.eval_json['current_accuracy']

        num_reset = self.eval_json['num_reset']

        cls = cls.to(torch.int32)  # bugfix when comparing outdist_index
        feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

        # Inference
        y_pred, y_conf, y_entropy, y_energy, y_embeddings, _, y_logit = self.model_inference(feats)

        if "softmax_score" in conf.args.acc_est_method:
            self.avg_confidence(y_logit)
        
        if "gde" in conf.args.acc_est_method:
            self.agreement(feats, y_pred)
        
        if "src_validation" in conf.args.acc_est_method:
            self.src_validation()

        if "adv_perturb" in conf.args.acc_est_method:
            self.adv(feats)

        if "aetta" in conf.args.acc_est_method:
            self.aetta(feats, y_pred)

        # append values to lists
        current_true_cls_list = [int(c) for c in cls.tolist()]
        true_cls_list += current_true_cls_list
        current_pred_cls_list = [int(c) for c in y_pred.tolist()]
        pred_cls_list += current_pred_cls_list
        conf_list += [float(c) for c in y_conf[cls != OUTDIST_CLASS_IDX].tolist()]

        if len(true_cls_list) > 0:
            current_accuracy = sum(1 for gt, pred in zip(current_true_cls_list, current_pred_cls_list) if gt == pred)\
                               / float(len(current_true_cls_list)) * 100
            current_accuracy_list.append(current_accuracy)
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred)\
                             / float(len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

            # epoch: 1~len(self.target_train_set[0])
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')
        
        # update self.json file
        self.eval_json = {
            'gt': true_cls_list,
            'pred': pred_cls_list,
            'accuracy': accuracy_list,
            'confidence': conf_list,
            'current_accuracy': current_accuracy_list,
            'num_reset': num_reset,
            
        }
                
    def dump_eval_online_result(self, is_train_offline=False):
        if is_train_offline:
            feats, cls, dls = self.target_train_set
            batchsize = conf.args.opt['batch_size']
            for num_sample in range(0, len(feats), batchsize):
                current_sample = feats[num_sample:num_sample+batchsize], cls[num_sample:num_sample+batchsize], dls[num_sample:num_sample+batchsize]
                self.evaluation_online(num_sample + batchsize,
                                       [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json = self.eval_json | self.acc_est_json
        json_subsample = {key: json[key] for key in json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')
        return class_accuracy_of_test_data, loss

    def add_instance_to_memory(self, current_sample, mem):
        with torch.no_grad():
            self.net.eval()

            if isinstance(mem, FIFO):
                mem.add_instance(current_sample)

            else:
                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                y_pred, y_conf, y_entropy, y_energy, y_embeddings, y_pred_softmax, _ = self.model_inference(f.unsqueeze(0))

                if isinstance(mem, ConfFIFO) or isinstance(mem, HUS) or isinstance(mem, Uniform) or isinstance(mem, PBRS):
                    mem.add_instance([f, y_pred.item(), d, y_conf.item()])

                elif isinstance(mem, CSTU):
                    mem.add_instance([f, y_pred.item(), y_entropy.item()])

                else:
                    raise NotImplementedError

    def train_online(self, current_num_sample, add_memory = True, evaluation = True):
        """
        Train the model
        """
        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        if add_memory:
            self.add_instance_to_memory(current_sample, self.fifo)  # for batch-based inference
            self.add_instance_to_memory(current_sample, self.mem)  # used for test-time adaptation

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data
                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        if evaluation:
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        if conf.args.no_adapt:  # for ablation
            return TRAINED

        data_loader = self.get_dataloader_for_adaptation()
        if not data_loader:
            return

        for e in range(conf.args.epoch):
            for batch_idx, feats in enumerate(data_loader):
            
                net, loss_item = self.test_time_adaptation(*feats)  # Each TTA algorithms must implement this
                if conf.args.reset_function:
                    self.reset_function(net, loss_item)

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def get_dataloader_for_adaptation(self) -> Optional[torch.utils.data.DataLoader]:
        if conf.args.memory_type in ['CSTU']:
            feats, _ = self.mem.get_memory()
        else:
            feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats).to(device)

        if len(feats) == 0:
            return None

        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)
        return data_loader

    def test_time_adaptation(self, feats) -> Tuple[Optional[torch.nn.Module], Optional[float]]:
        """
        Adapt with input `feats` and return (1) network with gradient and (2) loss value.
        """
        if len(feats) == 1:
            self.net.eval()  # avoid BN error
        else:
            self.net.train()

        if conf.args.method in ['Src']:
            return None, None
        else:
            raise NotImplementedError

    def fgsm_attack(self, x, epsilon, net=None):
        if net is None:
            net = self.net

        # from TIPI paper
        x = x.clone().to(device)
        epsilon = epsilon.to(device)
        delta = torch.zeros_like(x)
        delta.requires_grad = True

        with torch.enable_grad():
            pred = net(x + delta) if conf.args.opt['indices_in_1k'] == None else net(x + delta)[:, conf.args.opt['indices_in_1k']]
            loss = F.kl_div(pred, pred.detach(), reduction='batchmean')
            grad = torch.autograd.grad(loss, [delta])[0]

        delta = delta.detach() + torch.sign(grad.detach())
        delta *= epsilon.view(-1, 1, 1, 1) / torch.linalg.vector_norm(delta, ord=2, dim=(1, 2, 3), keepdim=True)  # l2 norm
        x_adv = torch.clamp(x + delta, 0, 1)

        net.zero_grad()

        return x_adv

    def pgd_attack(self, x, epsilon=8/255, net=None,
                   alpha=2/255, steps=10, random_start=True):
        """ Code from torchattacks library. © Copyright 2020, harrykim. Revision 936e86d6."""
        if net is None:
            net = self.net

        x = x.clone()

        adv_images = x.clone().detach()

        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -epsilon, epsilon
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        with torch.enable_grad():
            for _ in range(steps):
                adv_images.requires_grad = True
                outputs = net(adv_images) if conf.args.opt['indices_in_1k'] == None else net(adv_images)[:, conf.args.opt['indices_in_1k']]

                cost = F.kl_div(outputs, outputs.detach(), reduction='batchmean')

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + alpha * grad.sign()
                delta = torch.clamp(adv_images - x, min=-epsilon, max=epsilon)
                adv_images = torch.clamp(x + delta, min=0, max=1).detach()

        net.zero_grad()

        return adv_images

    def data_volume_density(self, src, tgt):
        std_prev = torch.std(src, (0, 2, 3))  # reduce except r, g, b
        std_curr = torch.std(tgt, (0, 2, 3))  # reduce except r, g, b
        c_den = (std_curr / std_prev).sum().item() / 3
        return c_den

    def data_distribution_diff(self, src, tgt):
        std_prev = torch.std(src, (0, 2, 3))  # reduce except r, g, b
        std_curr = torch.std(tgt, (0, 2, 3))  # reduce except r, g, b
        c_den = torch.linalg.vector_norm(std_prev - std_curr) / 3
        return c_den + 1  # To fit range [1, )

    def dropout_inference(self, x, n_iter, dropout, net=None):
        if net is None:
            net = self.net

        predictions = []
        with torch.no_grad():
            for _ in range(n_iter):
                pred = net[1]((net[0](x)), dropout=dropout)  # batch_size, n_classes
                pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # batch_size, n_iter, n_classes
        pred_class = torch.argmax(predictions, dim=2)
        mean_pred = torch.mean(predictions, dim=1)
        mean_pred_class = torch.argmax(mean_pred, dim=1)
        std_pred = torch.std(predictions, dim=1)
        return std_pred[:, mean_pred_class].diagonal(), pred_class, mean_pred, std_pred

    def evaluate_consistency_body(self, curr_feats):
        if len(curr_feats) == 0:
            return None, None

        curr_pred, curr_conf, curr_entropy, curr_energy, curr_embeddings, curr_softmax, _ = self.model_inference(curr_feats)

        eps = torch.tensor(1.0 / 255.0).to(device)

        self.sa_src_net.eval()
        curr_feats = self.fgsm_attack(curr_feats, epsilon=eps, net=self.sa_src_net)

        prev_preds, prev_conf, _, _, _, prev_softmax, _ = self.model_inference(curr_feats, net=self.sa_src_net)

        equal_rate = ((prev_preds == curr_pred).sum() / len(curr_pred)).item()
        conf_dist = (torch.linalg.vector_norm(prev_softmax.max() - curr_conf, ord=1) / len(curr_pred)).item()

        equal_rate = equal_rate if not np.isnan(equal_rate) else None
        conf_dist = conf_dist if not np.isnan(conf_dist) else None

        return equal_rate, conf_dist

    def evaluate_dropout(self, feats, net, n_iter=10, dropout=0.5):
        curr_pred, curr_conf, _, _, _, curr_softmax, _ = self.model_inference(feats, net=net)

        if dropout < 0:
            if conf.args.dataset == "cifar10outdist":
                dropout = 0.4
            elif conf.args.dataset == "cifar100outdist":
                dropout = 0.3
            elif conf.args.dataset == "imagenetoutdist":
                dropout = 0.2
            elif conf.args.dataset == "imagenetA":
                dropout = 0.7
            elif conf.args.dataset == "imagenetR":
                dropout = 0.3
            else:
                raise NotImplementedError

        _, pred, mean, std = self.dropout_inference(feats, n_iter=n_iter, dropout=dropout, net=net)

        conf_mean = mean[:, curr_pred].diagonal()
        conf_std = std[:, curr_pred].diagonal()
        mean_for_curr_pred = conf_mean.mean()
        std_for_curr_pred = conf_std.mean()

        total_avg_softmax = torch.mean(mean, dim=0)
        e_avg = (-total_avg_softmax * torch.log(total_avg_softmax + 1e-6)).sum()

        match_ratio = (curr_pred.unsqueeze(dim=1).repeat(1, n_iter) == pred).sum(dim=1, dtype=float) / n_iter
        acc = match_ratio.mean()
        return acc.item(), mean_for_curr_pred.item(), std_for_curr_pred.item(), e_avg.item()

    def init_reset(self, net, optimizer):
        self.prev_net = deepcopy(net)
        self.prev_net_state_queue = deque()
        self.prev_feat_queue = deque()
        self.prev_optim_state_queue = deque()
        self.curr_iter = 0
        self.num_reset = 0
        
        self.er_moving_average = None

        self.net_state, self.optimizer_state = reset_utils.copy_model_and_optimizer(net, optimizer)

        if not conf.args.reset_function:
            return
        
        if conf.args.reset_function == "SAR":
            self.ema = None
            
        elif conf.args.reset_function == "CoTTA":
            self.rst = conf.args.restoration_factor
        
        elif conf.args.reset_function == "PETAL":
            self.perc = 0.03  # for CIFAR100-C
            
        elif conf.args.reset_function == "periodic":
            self.count_for_reset = 0
            self.reset_every_x = conf.args.reset_every_x

        elif conf.args.reset_function == "aetta":
            self.reset_memory = []

        else:
            raise NotImplementedError
        
    def hard_reset(self):
        print("reset the model")
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net, self.optimizer, self.net_state, self.optimizer_state)
        self.num_reset += 1

    def soft_reset(self, net_state, optim_state):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net, self.optimizer, net_state, optim_state)
        self.num_reset += 1
        
    def reset_function(self, net=None, loss=None):
        if conf.args.reset_function == "SAR":
            if loss and not np.isnan(loss):
                self.ema = reset_utils.update_ema(self.ema, loss)

            if self.ema is not None and self.ema < 0.2:
                print("ema < 0.2, now reset the model")
                ema = self.ema
                self.hard_reset()
                self.ema = ema

        elif conf.args.reset_function == "CoTTA":
            for nm, m in net.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
                            
        elif conf.args.reset_function == "PETAL":
            if net is None:
                return
            self.fisher_dict = {}

            for nm, m in net.named_modules():  # previously used model, but now using self.model
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        self.fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
            fisher_list = []
            for name in self.fisher_dict:
                fisher_list.append(self.fisher_dict[name].reshape(-1))
            fisher_flat = torch.cat(fisher_list)
            reset_threshold_PETAL = reset_utils.find_quantile(fisher_flat, self.perc)

            for nm, m in net.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        # masking makes it restore candidate
                        mask_fish = (self.fisher_dict[f"{nm}.{npp}"] < reset_threshold_PETAL).float().cuda()
                        mask = mask_fish
                        with torch.no_grad():
                            p.data = self.net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
                            
        elif conf.args.reset_function == "periodic":
            self.count_for_reset += 1
            if self.count_for_reset == self.reset_every_x:
                print(f"reset every {self.reset_every_x}. Now reset")
                self.hard_reset()  # need to implement period of reset
                self.count_for_reset = 0
        
        elif conf.args.reset_function == "aetta":
            acc_preds = self.acc_est_json['aetta']
            self.reset_memory += [acc_preds[-1]]
            if acc_preds[-1] < 20:
                self.hard_reset()
                self.reset_memory = []
            if len(self.reset_memory) >= 10:
                self.reset_memory = self.reset_memory[-10:]
                if sum(self.reset_memory[-5:])/5 + 2 < sum(self.reset_memory[-10:-5])/5:
                    self.hard_reset()
                    self.reset_memory = []

        else:
            raise NotImplementedError

    def avg_confidence(self, y_logit):
        T = 2
        ac_conf = F.softmax(y_logit/T, dim=1).max(1, keepdim=False)[0]
        average_conf = self.acc_est_json[f'softmax_score']
        average_conf += [ac_conf.mean().item()]
        self.acc_est_json[f'softmax_score'] = average_conf
        
    def agreement(self, feats, y_pred):
        if self.sa_ref_net_1 is None:
            self.sa_ref_net_1 = deepcopy(self.net)
            
        self.prev_net_state_queue.appendleft(deepcopy(self.net.state_dict()))
        if len(self.prev_net_state_queue) > 10:
            self.prev_net_state_queue.pop()

        if len(self.prev_net_state_queue) > 1:
            self.sa_ref_net_1.load_state_dict(self.prev_net_state_queue[1], strict=True)
        
        self.sa_ref_net_1.eval() 
        with torch.no_grad():
            agreement = self.acc_est_json['gde']
            ref_y_pred, _, _, _, _, _, _ = self.model_inference(feats, net=self.sa_ref_net_1)
            agreement += [(ref_y_pred == y_pred).float().mean().item()]
            self.acc_est_json['gde'] = agreement

    def src_validation(self):
        src_feats, src_cls, src_dls = self.source_val_dataloader
        src_cls = src_cls.to(torch.int32)  # bugfix when comparing outdist_index
        src_feats, src_cls, src_dls = src_feats.to(device), src_cls.to(device), src_dls.to(device)

        src_accuracy = self.acc_est_json['src_validation']
        src_y_pred, _, _, _, _, _, _ = self.model_inference(src_feats)
        src_accuracy += [(src_y_pred == src_cls).float().mean().item()]
        self.acc_est_json['src_validation'] = src_accuracy

    def adv(self, feats):
        er, _ = self.evaluate_consistency_body(feats)
        self.acc_est_json['adv_perturb'] += [er]

    def aetta(self, feats, y_pred):
        est_acc, mean, std, e_avg = self.evaluate_dropout(feats, self.net, dropout=conf.args.dropout_rate)
        self.acc_est_json['est_dropout'] += [est_acc]
        self.acc_est_json['est_dropout_avg_entropy'] += [e_avg]
        self.acc_est_json['est_dropout_softmax_mean'] += [mean]
        self.acc_est_json['est_dropout_softmax_std'] += [std]

        est_err = 1 - est_acc
        if self.est_ema_dropout is None:
            self.est_ema_dropout = est_err

        if conf.args.dataset == "cifar10outdist":
            MAX_ENTROPY = 2.3026  # cifar10
            N_CLASS = 10
        elif conf.args.dataset == "cifar100outdist":
            MAX_ENTROPY = 4.6052  # cifar100
            N_CLASS = 100
        elif conf.args.dataset in ["imagenetA", "imagenetR"] :
            MAX_ENTROPY = 5.2983  # imagenetA, imagenetR
            N_CLASS = 200
        else:
            MAX_ENTROPY = 6.9078  # imagenet
            N_CLASS = 1000

        updated = est_err / (e_avg / MAX_ENTROPY) ** 3
        updated = max(0., min(1. - 1. / N_CLASS, updated))

        updated = self.est_ema_dropout * 0.6 + updated * 0.4
        self.est_ema_dropout = updated

        self.acc_est_json['aetta'] += [100 * (1. - updated)]
