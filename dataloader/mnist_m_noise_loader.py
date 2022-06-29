"""Noise training Data >> MNIST_m"""
"""Mnist_m is target Data"""
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
from torchvision import datasets, transforms, utils
import os
import torch.utils.data as data
import matplotlib.pyplot as plt

class GetLoader(Dataset):
    def __init__(self, dataset,data_root, data_list, transform=None):
        self.name = dataset
        self.root = data_root
        self.transform = transform
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()
        self.n_data = len(data_list)
        self.img_paths = []
        self.img_labels = []
        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
            
    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data



class mnist_m_dataloader():
    def __init__(self,root,batch_size,img_size,num_workers=8):
        self.name = "mnist_m"
        self.root = os.path.join(root, self.name)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
            transforms.Resize(self.img_size),
            # transforms.RandomCrop(self.img_size, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def run(self,mode):
        if mode == "train":
            self.train=True
            train_list = os.path.join(self.root, 'mnist_m_train_labels.txt')
            dataset_target = GetLoader(
                dataset=self.name,
                data_root=os.path.join(self.root, 'mnist_m_train'),
                data_list=train_list,
                transform=self.transform_train
            )
            dataloader = torch.utils.data.DataLoader(#最终对目标域进行操作的数据是dataloader_target
                dataset=dataset_target,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8
            )
            return dataloader
        elif mode == "test":
            self.train = False
            test_list = os.path.join(self.root,'mnist_m_test_labels.txt')
            dataset_target_test = GetLoader(
                dataset=self.name,
                data_root=os.path.join(self.root, 'mnist_m_test'),
                data_list=test_list,
                transform=self.transform_test
            )
            dataloader = torch.utils.data.DataLoader(#最终对目标域进行操作的数据是dataloader_target
                dataset=dataset_target_test,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8
            )
            return dataloader
        elif mode == "warmup":
            self.train=True
            train_list = os.path.join(self.root, 'mnist_m_train_labels.txt')
            dataset_target = GetLoader(
                dataset=self.name,
                data_root=os.path.join(self.root, 'mnist_m_train'),
                data_list=train_list,
                transform=self.transform_train
            )
            dataloader = torch.utils.data.DataLoader(#最终对目标域进行操作的数据是dataloader_target
                dataset=dataset_target,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8
            )
            return dataloader

        


if __name__=="__main__":

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    loader_target = mnist_m_dataloader(root=root,batch_size=128,img_size=28)
    dataloader_mnist = loader_target.run('warmup')
    for batch_idx, (inputs, labels) in enumerate(dataloader_mnist):
        fig = plt.figure()
        inputs = inputs.detach().cpu()
        grid = utils.make_grid(inputs)
        print("Labels", labels)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig(dir_path+'/dataset/mnist_m.png')
        break