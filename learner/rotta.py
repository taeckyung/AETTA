from typing import Optional, Tuple

import conf
from .dnn import DNN, TRAINED
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.loss_functions import *
from utils import memory, bn_layers_rotta, memory_rotta, reset_utils
from copy import deepcopy
from utils.custom_transforms import get_tta_transforms
from torch import nn
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class RoTTA(DNN):
    def __init__(self, *args, **kwargs):
        super(RoTTA, self).__init__(*args, **kwargs)

        self.net.requires_grad_(False)
        self.alpha = 0.05
        normlayer_names = []

        for name, sub_module in self.net.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.net, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = bn_layers_rotta.RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = bn_layers_rotta.RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.alpha)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.net, name, momentum_bn)

        params, param_names = self.collect_params(self.net)
        self.optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

        net_ema = deepcopy(self.net)
        for param in net_ema.parameters():
            param.detach_()

        self.net_not_ema = self.net
        self.net = net_ema  # set self.net to self.net_ema
        self.net.to(device)

        self.nu = 0.001

        self.init_reset(self.net_not_ema, self.optimizer)

    def hard_reset(self):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net_not_ema, self.optimizer, self.net_state, self.optimizer_state)

        net_ema = deepcopy(self.net_not_ema)
        for param in net_ema.parameters():
            param.detach_()

        self.net = net_ema  # set self.net to self.net_ema
        self.net.to(device)
        self.num_reset += 1
    
    def soft_reset(self, net_state, optim_state):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net_not_ema, self.optimizer, net_state, optim_state)
        
        net_ema = deepcopy(self.net_not_ema)
        for param in net_ema.parameters():
            param.detach_()

        self.net = net_ema  # set self.net to self.net_ema
        self.net.to(device)
        self.num_reset += 1
        
    def set_target_data(self, *args, **kwargs):
        super(RoTTA, self).set_target_data(*args, **kwargs)
        self.transform = get_tta_transforms(tuple(self.target_train_set[0][0].shape[1:]))

    def get_dataloader_for_adaptation(self) -> Optional[torch.utils.data.DataLoader]:
        assert(conf.args.memory_type in ['CSTU'])
        feats, ages = self.mem.get_memory()
        feats = torch.stack(feats).to(device)
        ages = torch.Tensor(ages).to(device)

        if len(feats) == 0:
            return None

        dataset = torch.utils.data.TensorDataset(feats, ages)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)
        return data_loader

    def test_time_adaptation(self, feats, ages=None):
        assert(ages is not None)

        # setup models
        self.net_not_ema.train()
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net_not_ema.eval()
            self.net.eval()

        strong_sup_aug = self.transform(feats)
        ema_sup_out = self.net(feats) if conf.args.opt['indices_in_1k'] == None else self.net(feats)[:, conf.args.opt['indices_in_1k']]
        stu_sup_out = self.net_not_ema(strong_sup_aug) if conf.args.opt['indices_in_1k'] == None else self.net_not_ema(strong_sup_aug)[:, conf.args.opt['indices_in_1k']]
        instance_weight = self.timeliness_reweighting(ages)
        loss = (softmax_entropy_rotta(stu_sup_out, ema_sup_out) * instance_weight).mean()

        if loss is not None:
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        self.net = self.update_ema_variables(self.net, self.net_not_ema, self.nu)

        return self.net_not_ema, loss.item()  # RoTTA backprops on student model. No gradient in teacher.

    def timeliness_reweighting(self, ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))

    def update_ema_variables(self, ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)