import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder

from PIL import Image

import pandas as pd
import time
import numpy as np
import sys
import conf
import json
import tqdm as tqdm

opt = conf.IMAGENET_A


class ImageNetADataset(torch.utils.data.Dataset):

    def __init__(self, file='',
                 domain=None, activities=None,
                 max_source=100, transform='none'):
        
        st = time.time()
        self.domain = domain
        self.activity = activities
        self.max_source = max_source

        self.domain = domain
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']
        self.transform_type = transform

        assert (len(domain) > 0)
        if domain.startswith('original'):
            self.path = 'dataset/Imagenet-C/origin/Data/CLS-LOC/train/'
        elif domain.startswith('test'):
            self.path = 'dataset/Imagenet-C/origin/Data/CLS-LOC/val/'
        elif domain == "corrupt":
            self.path = None
        else:
            raise NotImplementedError
            # self.path = 'corrupted/'
            # corruption, severity = domain.split('-')
            # self.path += corruption + '/' + severity + '/'
            
        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        elif transform == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        path = self.file_path if self.path == None else self.path
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        print('preprocessing images..')
        self.dataset = ImageFolder(path)

    def load_features(self):
        path = self.file_path if self.path == None else self.path
        dataset = ImageFolder(path, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=False, drop_last=False)
        # transformed_dataset = []
        for b_i, data in enumerate(dataloader):  # must be loaded from dataloader, due to transform in the __getitem__()
            feat, cl = data
            # convert a batch of tensors to list, and then append to our list one by one
            feats = torch.unbind(feat, dim=0)
            cls = torch.unbind(cl, dim=0)
            for i in range(len(feats)):
                # transformed_dataset.append((feats[i], cls[i]))
                self.features.append(feats[i])
                self.class_labels.append(cls[i])
                self.domain_labels.append(0)
        self.features = np.stack(self.features)
        self.class_labels = np.stack(self.class_labels)
        self.domain_labels = np.stack(self.domain_labels)

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return 1

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, torch.tensor(0)


if __name__ == '__main__':
    ### code for making imagenet validation data compatiable with ImageFolder!
    '''
    import os
    root = '/mnt/sting/tsgong/WWW/dataset/ImageNet-C/origin/Data/CLS-LOC/val/'
    f = open(root+'LOC_val_solution.csv', 'r')
    i=0
    for l in f:
        if i ==0: # ignore header
            i += 1
            continue
        filename=l.split(',')[0]
        label=l.split(',')[1].split(' ')[0]
        dir = root+label
        ### 1. make dir
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # print(os.path.join(root,filename,'.JPEG'))
        ### 2. move files to dir
        print(label)
        if os.path.isfile(os.path.join(root, filename + '.JPEG')):
            os.rename(os.path.join(root, filename + '.JPEG'), os.path.join(dir, filename + '.JPEG'))
        i += 1
    print(i)
    '''
