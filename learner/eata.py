
import conf
from data_loader.CIFAR100Dataset import CIFAR100Dataset
from data_loader.CIFAR10Dataset import CIFAR10Dataset
from data_loader.IMAGENETDataset import ImageNetDataset
from data_loader.IMAGENETADataset import ImageNetADataset
from data_loader.IMAGENETRDataset import ImageNetRDataset

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

from torch import optim

import conf
from data_loader.OutDistDataset import OutDistDataset
from data_loader.data_loader import datasets_to_dataloader
from utils.reset_utils import copy_model_and_optimizer
from .dnn import DNN
from torch.utils.data import  random_split

from utils.loss_functions import *
from utils import memory


class ETA(DNN):
    # EATA without anti-forgetting
    def __init__(self, model_, corruption_list_):
        super(ETA, self).__init__(model_, corruption_list_)

        configure_model(self.net)

        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=conf.args.opt['learning_rate'],
                                   momentum=conf.args.opt['momentum'],
                                   weight_decay=conf.args.opt['weight_decay'])

        # self.steps = 1 # SoTTA: replaced to epoch
        # assert self.steps > 0, "EATA requires >= 1 step(s) to forward and update"
        # self.episodic = False  # SoTTA: we don't use episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = conf.args.e_margin  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = conf.args.d_margin  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)

        self.fishers = None  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = conf.args.fisher_alpha  # trade-off \beta for two losses (Eqn. 8)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.net, self.optimizer)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()
        self.init_reset(self.net, self.optimizer)

    def test_time_adaptation(self, feats):
        if len(feats) == 1:
            self.net.eval()  # avoid BN error
        else:
            self.net.train()

        result = forward_and_adapt_eata(feats, self.net,
                                        self.optimizer,
                                        self.fishers,
                                        self.e_margin,
                                        self.current_model_probs,
                                        fisher_alpha=self.fisher_alpha,
                                        num_samples_update=self.num_samples_update_2,
                                        d_margin=self.d_margin)
        outputs, num_counts_2, num_counts_1, updated_probs, loss = result

        self.num_samples_update_2 += num_counts_2
        self.num_samples_update_1 += num_counts_1
        self.reset_model_probs(updated_probs)
        return self.net, loss.item()

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


class EATA(ETA):
    def __init__(self, model_, corruption_list_):
        super(EATA, self).__init__(model_, corruption_list_)

        # only use the first domain for fisher importance calculation
        if conf.args.dataset in ["cifar10outdist", "cifar100outdist", "imagenetoutdist"]:
            fisher_dataset = OutDistDataset(base=conf.args.dataset, domains=[corruption_list_[0]], max_source=9999,
                                            transform='val',
                                            outdist=conf.args.outdist, outdist_size=conf.args.outdist_size,
                                            outdist_class=conf.args.outdist_class)
        elif conf.args.dataset == "cifar10":
            fisher_dataset = CIFAR10Dataset(file="", domains=[corruption_list_[0]], max_source=9999, transform='val')
        elif conf.args.dataset == "cifar100":
            fisher_dataset = CIFAR100Dataset(file="", domains=[corruption_list_[0]], max_source=9999, transform='val')
        elif conf.args.dataset == "imagenet":
            fisher_dataset = ImageNetDataset(file="", domain=corruption_list_[0], max_source=9999, transform='val')
        elif conf.args.dataset == "imagenetA":
            fisher_dataset = ImageNetADataset(file="", domain=corruption_list_[0], max_source=9999, transform='val')
        elif conf.args.dataset == "imagenetR":
            fisher_dataset = ImageNetRDataset(file="", domain=corruption_list_[0], max_source=9999, transform='val')
        else:
            raise NotImplementedError

        fisher_dataset = random_split(fisher_dataset, [conf.args.fisher_size, len(fisher_dataset)-conf.args.fisher_size])[0]
        fisher_loader = datasets_to_dataloader([fisher_dataset], batch_size=64, concat=True, shuffle=True)

        subnet = configure_model(self.net)
        params, param_names = collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets, domains) in enumerate(fisher_loader, start=1):
            if conf.args.gpu_idx is not None:
                images = images.cuda(conf.args.gpu_idx, non_blocking=True)
            # if torch.cuda.is_available():
            #     targets = targets.cuda(conf.args.gpu_idx, non_blocking=True)
            outputs = subnet(images) if conf.args.opt['indices_in_1k'] == None else subnet(images)[:, conf.args.opt['indices_in_1k']]
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        # logger.info("compute fisher matrices finished")
        del ewc_optimizer
        self.fishers = fishers  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0,
                           d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return:
    1. model outputs;
    2. the number of reliable and non-redundant samples;
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward 
    outputs = model(x) if conf.args.opt['indices_in_1k'] == None else model(x)[:, conf.args.opt['indices_in_1k']]
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0),
                                                  outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1]) ** 2).sum()
        loss += ewc_loss
    optimizer.zero_grad()  # move it here to make PETAL reset works!
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    # optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def check_model(model):
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"
