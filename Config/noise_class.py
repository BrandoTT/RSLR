import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch
'''
MNIST -> Noise 
Class -> torchvision.datasets.MNIST() 
'''
class MNISTNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_classes = 10
    def uniform_mix(self, noise_rate, mixing_ratio, num_classes): #mixing_ration is corruption_level, noise_rate is gold_fraction
        # ntm = mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        #       (1 - mixing_ratio) * np.eye(num_classes)
        ntm = (1 - mixing_ratio) * np.identity(num_classes) + \
                (np.ones(num_classes) - np.identity(num_classes)) * (mixing_ratio / (num_classes-1))
        #噪声样本个数和干净样本个数
        num_noise = int(len(self.data) * noise_rate)
        num_gold = int(len(self.data) * (1 - noise_rate))
        for i in range(num_noise):
            self.targets[i] = np.random.choice(num_classes, p=ntm[self.targets[i]])
        return ntm, num_noise, num_gold

    def flip(self, noise_rate, mixing_ratio, num_classes):
        ntm = np.eye(num_classes) * (1 - mixing_ratio)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            ntm[i][np.random.choice(row_indices[row_indices != i])] = mixing_ratio

        #噪声样本个数和干净样本个数
        num_noise = int(len(self.data) * noise_rate)
        num_gold = int(len(self.data) * (1 - noise_rate))
        for i in range(num_noise):
            self.targets[i] = np.random.choice(num_classes, p=ntm[self.targets[i]])
        #[0 : num_silver-1] is noise; [num_silver: ] is gold data
        return ntm, num_noise, num_gold

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)
