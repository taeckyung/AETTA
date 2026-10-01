import random

import numpy as np
import torch

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def reset(self):
        self.data = [[], [], []]


class HUS:
    def __init__(self, capacity, threshold=None, num_class=None):
        self.data = [[[], [], [], []] for _ in
                     range(num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.threshold = threshold
        self.num_class = num_class

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']
        self.num_class = state_dict['num_class']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold
        dic['num_class'] = self.num_class

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][0])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][0])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)
            self.data[tgt_idx][3].append(aux[i])

    def reset(self):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf


class ConfFIFO:
    def __init__(self, capacity, threshold):
        self.data = [[], [], [], []]
        self.capacity = capacity
        self.threshold = threshold
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        self.threshold = state_dict['threshold']
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold
        return dic

    def get_memory(self):
        return self.data[:3]

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 4)

        if instance[3] < self.threshold:
            return

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def reset_value(self, feats, cls, aux):
        self.data = [[], [], [], []]

    def reset(self):
        self.data = [[], [], [], []]


class ReplayMemory:
    def __init__(self, batch_size, interval):
        self.batch_size = batch_size
        self.interval = interval
        self.features = None
        self.pseudo_labels = None
        self.confidences = None

    def get_memory(self):
        return self.features, self.pseudo_labels, self.confidences

    def pop_memory(self):
        target_size = [self.batch_size, len(self.features) - self.batch_size]
        feats, self.features = torch.split(self.features, target_size)
        pls, self.pseudo_labels = torch.split(self.pseudo_labels, target_size)
        confs, self.confidences = torch.split(self.confidences, target_size)
        return feats, pls, confs

    def add_instance(self, instance):
        """
        Assumes incoming features and pseudo_labels are in shape of (B, ...) and (B, N)
        """
        assert (len(instance) == 3)  # features, pseudo_labels
        self.features = torch.cat((self.features, instance[0])) if self.features is not None else instance[0]
        self.pseudo_labels = torch.cat((self.pseudo_labels, instance[1])) if self.pseudo_labels is not None else instance[1]
        self.confidences = torch.cat((self.confidences, instance[2])) if self.confidences is not None else instance[2]

    def reset(self):
        self.features = None
        self.pseudo_labels = None
        self.confidences = None


class Uniform:
    def __init__(self, capacity, num_class = None):
        self.data = [[[], [], [], []] for _ in
                     range(num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.threshold = 0.0
        self.num_class = num_class

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']
        self.num_class = state_dict['num_class']
    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold
        dic['num_class'] = self.num_class

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][0])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][0])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)
            self.data[tgt_idx][3].append(aux[i])

    def reset(self):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

class PBRS():

    def __init__(self, capacity, num_class=None):
        self.data = [[[], [], []] for _ in range(num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.num_class = num_class

    def reset(self):
        self.data = [[[], [], []] for _ in range(self.num_class)] #feat, pseudo_cls, domain, cls, loss

    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
