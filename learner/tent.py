from torch import optim

import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
from utils import memory

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TENT(DNN):
    def __init__(self, *args, **kwargs):
        super(TENT, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        if conf.args.dataset in ['imagenet', 'imagenetoutdist']:  # TENT use SGD for imagenet
            self.optimizer = optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                       weight_decay=conf.args.opt['weight_decay'])

        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                        weight_decay=0.0)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()

        self.init_reset(self.net, self.optimizer)

    def test_time_adaptation(self, feats):
        if len(feats) == 1:
            self.net.eval()  # avoid BN error
        else:
            self.net.train()

        entropy_loss = HLoss()

        preds_of_data = self.net(feats) if conf.args.opt['indices_in_1k'] == None else self.net(feats)[:, conf.args.opt['indices_in_1k']]

        loss = entropy_loss(preds_of_data)

        self.optimizer.zero_grad()


        loss.backward()

        self.optimizer.step()

        return self.net, loss.item()
