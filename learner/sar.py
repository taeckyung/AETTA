from copy import deepcopy

import math

import conf
from utils import memory
from utils.reset_utils import copy_model_and_optimizer, load_model_and_optimizer
from utils.sam_optimizer import sam_collect_params, SAM
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(DNN):
    def __init__(self, *args, **kwargs):
        super(SAR, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        self.net.train()

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            # TODO: support use_learned_stats
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.requires_grad_(True)

        params, _ = sam_collect_params(self.net)
        self.optimizer = SAM(params, torch.optim.SGD, lr=conf.args.opt['learning_rate'],
                             momentum=conf.args.opt['momentum'])

        # SAR-specific hyperparameters
        if conf.args.outdist == "divide":
            num_class = conf.args.opt['num_class'] - len(conf.args.outdist_class)
        else:
            num_class = conf.args.opt['num_class']

        self.margin_e0 = 0.4 * math.log(num_class) # math.log(1000)
        self.reset_constant_em = 0.2
        self.ema = None

        self.net_state, self.optimizer_state = copy_model_and_optimizer(self.net, self.optimizer)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)  # required for evaluation
        self.mem_state = self.mem.save_state_dict()

        self.init_reset(self.net, self.optimizer)

    def hard_reset(self):
        super(SAR, self).hard_reset()
        if not conf.args.turn_off_reset:
            self.ema = None

    def test_time_adaptation(self, feats):
        if len(feats) == 1:
            self.net.eval()  # avoid BN error
        else:
            self.net.train()

        feats = feats.to(device)

        self.optimizer.zero_grad()

        preds_of_data = self.net(feats) if conf.args.opt['indices_in_1k'] == None else self.net(feats)[:, conf.args.opt['indices_in_1k']]

        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(preds_of_data)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)


        loss.backward()


        # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        self.optimizer.first_step(zero_grad=True)

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        entropys2 = softmax_entropy(self.net(feats)) if conf.args.opt['indices_in_1k'] == None else softmax_entropy(self.net(feats)[:, conf.args.opt['indices_in_1k']])

        entropys2 = entropys2[filter_ids_1]
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)
        loss_second = entropys2[filter_ids_2].mean(0)
        


        loss_second.backward()

        self.optimizer.second_step(zero_grad=False)

        if conf.args.turn_off_reset is False and conf.args.reset_function != "SAR":
            if not np.isnan(loss_second.item()):
                self.ema = update_ema(self.ema, loss_second.item())
            if self.ema is not None and self.ema < 0.2:
                print("ema < 0.2, now reset the model")
                ema = self.ema
                self.hard_reset()
                self.ema = ema

        return self.net, loss_second.item()
