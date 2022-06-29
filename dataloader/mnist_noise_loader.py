
import sys, os 
sys.path.append(os.path.dirname(__file__) + os.sep+'../')
import torch.utils.data 
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
from dataloader.randaugment import RandAugmentMC
from Config.utils import sym_noise_new,asym_noise_new
from Record.Drawing import data_labeldistribution
#from torchnet.meter import AUCMeter
import operator
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

# Source is Source 
class MNISTNoisy(datasets.MNIST):
    """作为MNISTNoisy来做"""
    def __init__(self,root,mode,ratio,noise_level,noise_type,train=True,transform=None,target_transform=None,download=False,pred=[], probability=[], noise_file=''):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_classes = 10
        self.mode = mode # all, labeled, unlabeled, test Data
        self.ratio = ratio
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.name = 'mnist'
        if self.mode=='ALL_Data':
            self.weak_transform = transform['weak']
            self.strong_transform = transform['strong']
            self.normalize_transform = transform['normalize']
        else:
            self.test_transform = transform['test']
        # class_ratio = np.zeros(31)
        # total_num = len(self.targets)
        # for label in range(self.num_classes):
        #     count_label = self.targets.count(label)
        #     label_ratio = count_label / total_num
        #     class_ratio[label] = label_ratio
        # class_ratio = class_ratio.tolist()
        # json.dump(class_ratio, open(self.root+'class_ratio_{}'.format(self.name), "w")) # 用于采样服从目标域的测试
        # noise_label = json.load(open(noise_file, "r"))
        # data_labeldistribution(root,self.targets,noise_label,self.name)
        # exit()

        # asym noise structure
        if self.mode == "test":
            self.TestData = self.data
            self.test_label = self.targets        
        else:
            self.data = self.data.numpy().tolist()
            self.targets_list = self.targets.numpy().tolist()
            print("Noise numbers:",self.ratio*len(self.data))
            if os.path.exists(noise_file):
                print("{} noise labels has been made\n".format(self.ratio))
                noise_label = json.load(open(noise_file, "r"))
            else:
                if self.noise_type == 'sym':
                    noise_label = sym_noise_new(self.ratio,self.num_classes,self.targets_list)
                elif self.noise_type == 'asym':
                    noise_label = asym_noise_new(self.ratio,self.num_classes,self.targets_list)
                print("Saving noisy labels to %s ... "%noise_file)    
                json.dump(noise_label, open(noise_file, "w"))
                exit()

            Data = torch.tensor(self.data)
            Label = torch.tensor(noise_label)
            self.TrainData = Data
            self.noise_label = Label
            
    def __len__(self):
        if self.mode=='test':
            return len(self.TestData)
        else:
            return len(self.TrainData)

    def __getitem__(self, index):
        
        if self.mode == 'ALL_Data':
            img, noise_target, ground = self.TrainData[index], int(self.noise_label[index]),int(self.targets[index])
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='L').convert("RGB")
            weak_img = self.normalize_transform(self.weak_transform(img))
            strong_img = self.normalize_transform(self.strong_transform(img))
            return weak_img, strong_img, noise_target, ground, index
        
        elif self.mode == 'test':
            img, target = self.TestData[index], int(self.test_label[index])
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='L').convert("RGB")
            img = self.test_transform(img)
            return img, target


class mnist_dataloader():
    def __init__(self,root,ratio,level,type,batch_size,img_size,num_workers=8,noise_file=''):
        self.name = "MNIST"
        self.root = root
        self.ratio = ratio
        self.level = level
        self.type = type
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.noise_file = noise_file
        self.all_transform = { 
            'weak':transforms.Compose([transforms.Resize(self.img_size),
                                       transforms.RandomHorizontalFlip(),
                                       #,transforms.RandomRotation(degrees=5) 
                                       ]),
            'strong':transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=5),
                RandAugmentMC(n=2,m=10)
            ]),
            'normalize':transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5, 0.5, 0.5))]),
            'test':transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5, 0.5, 0.5))])
        }

    def run(self,MODE):
        if MODE == 'test':
            self.train = False
            test_dataset = MNISTNoisy(
                root=self.root,
                mode="test",
                ratio=self.ratio,
                noise_level=self.level,
                noise_type=self.type,
                train=self.train,
                transform=self.all_transform,
                download=False,
                noise_file=self.noise_file,
            )
            testdata_loader = torch.utils.data.DataLoader(
                dataset = test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return len(test_dataset),testdata_loader
            
        elif MODE == 'ALL_Data':
           self.train = True
           all_dataset = MNISTNoisy(
                root=self.root,
                mode="ALL_Data",
                ratio=self.ratio,
                noise_level=self.level,
                noise_type=self.type,
                train=self.train,
                transform=self.all_transform,
                download=False,
                noise_file=self.noise_file
           )
           return all_dataset


# # Mnist is Target Data

# class mnist_dataloader_target():
#     def __init__(self,root,batch_size,img_size,num_workers=8):
#         self.name = "MNIST"
#         self.root = root
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.num_workers = num_workers
#         self.transform_train = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.RandomHorizontalFlip(), 
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.repeat(3,1,1)),
#             transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5, 0.5, 0.5))
#         ])
#         self.transform_test = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.repeat(3,1,1)),
#             transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5, 0.5, 0.5))
#         ])
#         self.transform_unlabeled = {
#             'weak':transforms.Compose([transforms.Resize(size=self.img_size),transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=10),
#                                        transforms.RandomCrop(256)]),
#             'strong':transforms.Compose([
#                 transforms.Resize(size=self.img_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(256),
#                 RandAugmentMC(n=2,m=10)
#             ]),
#             'normalize':transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#         }

#     def run(self,mode,get_dataset=True):
#         if mode=='train':
#             self.train=True
#             labeled_dataset = datasets.MNIST(
#                 root=self.root,
#                 train=self.train,
#                 transform=self.transform_train,
#                 download=False
#             )
#             if get_dataset==True:
#                 return labeled_dataset
#             else:
#                 dataloader = torch.utils.data.DataLoader(
#                     dataset=labeled_dataset,
#                     batch_size=self.batch_size,
#                     shuffle=True, 
#                     num_workers=8
#                 )
#                 return dataloader
#         elif mode=='test':
#             self.train=False
#             labeled_dataset = datasets.MNIST(
#                 root=self.root,
#                 train=self.train,
#                 transform=self.transform_test,
#                 download=False
#             )
#             dataloader = torch.utils.data.DataLoader(#最终对目标域进行操作的数据是dataloader_target
#                 dataset=labeled_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=8
#             )
#             return dataloader
#         elif mode=='warmup':
#             self.train=True
#             labeled_dataset = datasets.MNIST(
#                 root=self.root,
#                 train=self.train,
#                 transform=self.transform_train,
#                 download=False
#             )
#             if get_dataset==True:
#                 return labeled_dataset
#             else:
#                 dataloader = torch.utils.data.DataLoader(
#                     dataset=labeled_dataset,
#                     batch_size=self.batch_size,
#                     shuffle=True, 
#                     num_workers=8
#                 )
#                 return dataloader







if __name__=="__main__":
    pass
    # dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # root = dir_path+'/dataset/'
    # Source_loader = mnist_dataloader_target(root=root,
    #                                 batch_size=128,
    #                                 img_size=32
    # )
    # dataloader_mnist = Source_loader.run('warmup')
    # for batch_idx, (inputs, labels) in enumerate(dataloader_mnist):
    #     fig = plt.figure()
    #     inputs = inputs.detach().cpu()
    #     grid = utils.make_grid(inputs)
    #     print("Labels", labels)
    #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #     plt.savefig(dir_path+'/dataset/mnist_target.png')
    #     break