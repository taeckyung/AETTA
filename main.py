# -*- coding: utf-8 -*-

import sys
import argparse
import random

import math
import numpy as np
import torch
import time
import os
import conf
from copy import deepcopy


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# For results reproducibility; would increase GPU memory ~24MiB
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_path():
    # path = 'log_231026/aapnocof/'
    path = conf.args.log_name
    # path = 'log_231026/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type

    if conf.args.outdist:
        path += conf.args.method + "_outdist/"
    else:
        path += conf.args.method + '/'

    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    if conf.args.tgt:
        if conf.args.outdist:
            path += 'tgt_' + conf.args.tgt + '_{}'.format(conf.args.outdist) + '_{}'.format(
                conf.args.outdist_size) + '_{}/'.format(conf.args.outdist_class)
        else:
            path += 'tgt_' + conf.args.tgt + '/'

    if conf.args.reset_function:
        path += "reset_{}/".format(conf.args.reset_function)

    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyperparameters #################
    if 'cifar100' in conf.args.dataset:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in conf.args.dataset:
        opt = conf.CIFAR10Opt
    elif conf.args.dataset in ['imagenet', 'imagenetoutdist']:
        opt = conf.IMAGENET_C
    elif 'imagenetA' in conf.args.dataset:
        opt = conf.IMAGENET_A
    elif 'imagenetR' in conf.args.dataset:
        opt = conf.IMAGENET_R
    else:
        raise NotImplementedError

    conf.args.opt = opt
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay

    model = None
    tokenizer = None  # for language models

    if conf.args.model == "resnet18":
        from models import ResNet
        model = ResNet.ResNet18
    elif conf.args.model == "resnet18_pretrained":
        import torchvision
        model = torchvision.models.resnet18
    elif conf.args.model == "resnet50":
        from models import ResNet
        model = ResNet.ResNet50
    elif conf.args.model == "resnet50_pretrained":
        import torchvision
        model = torchvision.models.resnet50
    elif conf.args.model == "wide_resnet":
        from models import wide_resnet
        model = wide_resnet.WideResNet

    # import modules after setting the seed
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    from learner.tent import TENT
    from learner.sotta import SoTTA
    from learner.cotta import CoTTA
    from learner.sar import SAR
    from learner.rotta import RoTTA
    from learner.eata import ETA, EATA

    result_path, checkpoint_path, log_path = get_path()

    ########## Dataset loading ############################

    if conf.args.method == 'Src':
        learner_method = DNN
    elif conf.args.method == 'TENT':
        learner_method = TENT
    elif conf.args.method == 'CoTTA':
        learner_method = CoTTA
    elif conf.args.method == 'SAR':
        learner_method = SAR
    elif conf.args.method == "RoTTA":
        learner_method = RoTTA
    elif conf.args.method == 'SoTTA':
        learner_method = SoTTA
    elif conf.args.method == "EATA":
        learner_method = EATA
    elif conf.args.method == "ETA":
        learner_method = ETA
    else:
        raise NotImplementedError

    corruption_list = []

    # modify for continual adaptation
    if conf.args.tgt == "cont":
        if conf.args.cont_seq not in conf.CONT_SEQUENCE.keys():
            corruption_list_ = conf.CONT_SEQUENCE[0]
            random.shuffle(corruption_list_)
        else:
            corruption_list_ = conf.CONT_SEQUENCE[conf.args.cont_seq]

        corruption_list += corruption_list_
    else:
        corruption_list += [conf.args.tgt]

    learner = learner_method(model, corruption_list)
    original_result_path, original_checkpoint_path, original_log_path = result_path, checkpoint_path, log_path

    for corruption in corruption_list:

        if conf.args.tgt == "cont":
            result_path = original_result_path + corruption + "/"
            checkpoint_path = original_checkpoint_path + corruption + "/"
            log_path = original_log_path + corruption + "/"

        else:
            result_path = original_result_path
            checkpoint_path = original_checkpoint_path
            log_path = original_log_path

        learner.init_json(log_path)
        learner.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

        since = time.time()

        print('##############Source Data Loading...##############')
        set_seed()  # reproducibility
        source_train_data_loader, source_val_data_loader = data_loader.domain_data_loader(conf.args.dataset,
                                                                                          conf.args.src,
                                                                                          conf.args.opt['file_path'],
                                                                                          batch_size=conf.args.opt[
                                                                                              'batch_size'],
                                                                                          valid_split=0,
                                                                                          # to be used for the validation
                                                                                          test_split=0, is_src=True,
                                                                                          num_source=conf.args.num_source)

        print('##############Target Data Loading...##############')
        set_seed()  # reproducibility
        target_data_loader, _ = data_loader.domain_data_loader(conf.args.dataset, corruption,
                                                               conf.args.opt['file_path'],
                                                               batch_size=conf.args.opt['batch_size'],
                                                               valid_split=0,
                                                               test_split=0, is_src=False,
                                                               num_source=conf.args.num_source)

        set_seed()  # reproducibility
        learner.set_target_data(source_train_data_loader, source_val_data_loader, target_data_loader, corruption)

        time_elapsed = time.time() - since
        print('Data Loading Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        #################### Training #########################     

        since = time.time()

        # make dir if it doesn't exist
        if not os.path.exists(result_path):
            oldumask = os.umask(0)
            os.makedirs(result_path, 0o777)
            os.umask(oldumask)
        if not os.path.exists(checkpoint_path):
            oldumask = os.umask(0)
            os.makedirs(checkpoint_path, 0o777)
            os.umask(oldumask)
        script = ' '.join(sys.argv[1:])

        set_seed()  # reproducibility

        if conf.args.online == False:
            start_epoch = 1
            best_acc = -9999
            best_epoch = -1

            for epoch in range(start_epoch, conf.args.epoch + 1):
                learner.train(epoch)

            learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                    checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
            learner.dump_eval_online_result(is_train_offline=True)  # eval with final model

            time_elapsed = time.time() - since
            print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

        elif conf.args.online == True:

            current_num_sample = 1
            num_sample_end = conf.args.nsample
            best_acc = -9999
            best_epoch = -1

            TRAINED = 0
            SKIPPED = 1
            FINISHED = 2

            finished = False

            while not finished and current_num_sample < num_sample_end:

                ret_val = learner.train_online(current_num_sample)

                if ret_val == FINISHED:
                    break
                elif ret_val == SKIPPED:
                    pass
                elif ret_val == TRAINED:
                    pass
                current_num_sample += 1

            if not conf.args.remove_cp:
                learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                        checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
            learner.dump_eval_online_result()

            time_elapsed = time.time() - since
            print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

        if conf.args.remove_cp:
            last_path = checkpoint_path + 'cp_last.pth.tar'
            best_path = checkpoint_path + 'cp_best.pth.tar'
            try:
                os.remove(last_path)
                os.remove(best_path)
            except Exception as e:
                pass

def parse_arguments(argv):
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ###MANDATORY###

    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [ichar, icsr, dsa, hhar, opportunity, gait, pamap2]')

    parser.add_argument('--model', type=str, default='HHAR_model',
                        help='Which model to use')

    parser.add_argument('--method', type=str, default='',
                        help='specify the method name')

    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgt', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    ###Optional###
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight_decay to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nsample', type=int, default=20000000,
                        help='How many samples do you want use for train')
    parser.add_argument('--log_prefix', type=str, default='',
                        help='Prefix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')
    parser.add_argument('--data_gen', action='store_true',
                        help='generate training data with source for training estimator')

    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')
    parser.add_argument('--parallel', type=bool, default=False)
    #### Distribution ####
    parser.add_argument('--tgt_train_dist', type=int, default=1,
                        help='0: real selection'
                             '1: random selection'
                        )
    parser.add_argument('--online', action='store_true', help='training via online learning?')
    parser.add_argument('--update_every_x', type=int, default=1, help='number of target samples used for every update')
    parser.add_argument('--memory_size', type=int, default=1,
                        help='number of previously trained data to be used for training')
    parser.add_argument('--memory_type', type=str, default='FIFO', help='FIFO')

    # CoTTA
    parser.add_argument('--ema_factor', type=float, default=0.999,
                        help='hyperparam for CoTTA')
    parser.add_argument('--restoration_factor', type=float, default=0.01,
                        help='hyperparam for CoTTA')
    parser.add_argument('--aug_threshold', type=float, default=0.92,
                        help='hyperparam for CoTTA')

    # OOD
    parser.add_argument('--outdist', default="original", type=str,
                        help='outdist test data : divide, repeat, oneclassrepeat, cifar100, cifar100c, gaussian, uniform')
    parser.add_argument('--outdist_size', default=None, type=int, help='outdist test data size')
    parser.add_argument('--outdist_class', default=None, nargs="*", type=int, help='outdist target class')

    # SoTTA
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='momentum')
    parser.add_argument('--use_learned_stats', action='store_true', help='Use learned stats')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for HLoss')
    parser.add_argument('--loss_scaler', type=float, default=0,
                        help='loss_scaler for entropy_loss')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation data instead of test data for hyperparameter tuning')
    parser.add_argument('--adapt_then_eval', action='store_true',
                        help='Evaluation after adaptation - unrealistic and causing additoinal latency, but common in TTA.')
    parser.add_argument('--no_adapt', action='store_true', help='no adaptation')
    parser.add_argument('--skip_thres', type=int, default=1,
                        help='skip threshold to discard adjustment')

    parser.add_argument('--dummy', action='store_true', default=False, help='do nothing')

    # SAM (currently only supports SoTTA)
    parser.add_argument('--sam', action='store_true', default=False, help='changes Adam to SAM-Adam')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    parser.add_argument('--high_threshold', default=0.99, type=float, help='High confidence threshold')

    parser.add_argument('--reset_function', type=str, default=None,
                        help='reset function for adding Recovery ability to any TTA algorithm')
    
    parser.add_argument('--reset_every_x', default=1, type=int, help='cycle (in batch) of periodic reset')
    
    parser.add_argument('--num_eval_source', default=1000, type=int, help='number of source data used for SrcValidation')

    parser.add_argument('--turn_off_reset', action='store_true', default=False,
                        help="turn off default resetting algorithm, ie SAR and CoTTA")

    parser.add_argument('--acc_est_method', default='', type=str, nargs="*",
                        help='accuracy estimation methods to evaluate')

    parser.add_argument("--dropout_rate", default=-1, type=float, help='dropout rate for dropout inference')

    parser.add_argument('--log_name', type=str, default='log/')
    parser.add_argument('--cont_seq', default=0, type=int, help='switch to various order of cont datastream')
    
    parser.add_argument("--debug", action='store_true', default=False)
    
    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments(sys.argv[1:])
    print(conf.args)
    set_seed()
    if conf.args.debug:
        import debugpy
        debugpy.listen(5678)
        print("wait for debugger")
        debugpy.wait_for_client()
        print("attach")

    main()
