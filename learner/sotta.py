import conf
from utils import reset_utils
from .dnn import DNN
from torch.utils.data import DataLoader
from utils.loss_functions import *
from utils.sam_optimizer import SAM, sam_collect_params


device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SoTTA(DNN):
    def __init__(self, *args, **kwargs):
        super(SoTTA, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

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

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):  # ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):  # language models
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        if conf.args.sam:
            params, _ = sam_collect_params(self.net)
            self.optimizer = SAM(params, torch.optim.Adam, rho=0.05, lr=conf.args.opt['learning_rate'],
                                 weight_decay=conf.args.opt['weight_decay'])

        self.init_reset(self.net, self.optimizer)

    def test_time_adaptation(self, feats):
        if len(feats) == 1:
            self.net.eval()  # avoid BN error
        else:
            self.net.train()

        loss_fn = HLoss(conf.args.temperature)

        self.net.train()
        feats = feats.to(device)
        preds_of_data = self.net(feats) if conf.args.opt['indices_in_1k'] == None else self.net(feats)[:, conf.args.opt['indices_in_1k']]

        loss_first = loss_fn(preds_of_data)

        self.optimizer.zero_grad()

        loss_first.backward()

        if not isinstance(self.optimizer, SAM):
            self.optimizer.step()
            return self.net, loss_first.item()
        else:
            # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            self.optimizer.first_step(zero_grad=True)

            preds_of_data = self.net(feats) if conf.args.opt['indices_in_1k'] == None else self.net(feats)[:, conf.args.opt['indices_in_1k']]

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second = loss_fn(preds_of_data)

            loss_second.backward()

            self.optimizer.second_step(zero_grad=False)
            return self.net, loss_second.item()
