from numpy.lib.npyio import load
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import random
from PIL import Image
import torch.utils.data
import os
import os.path
import json
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms, utils
from dataloader.randaugment import RandAugmentMC
#from randaugment import RandAugmentMC
from Config.utils import sym_noise_new
from Record.Drawing import data_labeldistribution
import numpy as np

r"""
office-home:
Art(Ar)
Clipart(Cl)
Product(Pr)
Real World(Rw)
"""

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def make_dataset(image_list):

    len_ = len(image_list)
    image_list = [image_list[i].strip() for i in range(len_)] # 去掉头尾换行符
    # return [images,ground,noise]
    if len(image_list[0].split(' ',2)) == 3: # 是源域训练数据
        image = []
        noise_labels = []
        ground_labels = []
        for val in image_list:
            image.append(val.split(' ',2)[0])
            noise_labels.append(int(val.split(' ',2)[1]))
            ground_labels.append(int(val.split(' ',2)[2]))
        return image, ground_labels, noise_labels
    elif len(image_list[0].split(' ',2)) == 2: # 测试数据或者是目标域数据
        image = []
        ground_labels = []
        for val in image_list:
            image.append(val.split(' ',2)[0])
            ground_labels.append(int(val.split(' ',2)[1]))
        return image, ground_labels
    

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)       


class office_home_Noisy(object):
    def __init__(self,root,name,mode,domain,transform=None,target_transform=None,noise_file='',loader=default_loader):
        
        self.num_classes = 65
        self.mode = mode
        self.name = name
        self.domain = domain
        if self.mode == 'ALL_Data':
            self.weak_transform = transform['weak']
            self.strong_transform = transform['strong']
            self.normalize_transform = transform['normalize']
        else:
            self.test_transform = transform['test']
        # loader images
        self.loader = loader
        # txt -> list
        image_la_list = open(noise_file).readlines()
        if self.domain == 'source' and self.mode == 'ALL_Data':
            self.path, self.ground, self.noise_labels = make_dataset(image_la_list)
        elif self.domain == 'target' or self.mode == 'test':
            self.path, self.ground = make_dataset(image_la_list) 
        #self.Data = [self.loader(i) for i in self.path]
    
    def __getitem__(self, index):
        path = self.path[index]
        self.Data = self.loader(path)
        if self.mode == 'ALL_Data':
            if self.domain == 'target':
                img, noise_target, clean_targets = self.Data, int(self.ground[index]), int(self.ground[index])
            elif self.domain == 'source':    
                img, noise_target, clean_targets = self.Data, int(self.noise_labels[index]), int(self.ground[index])
            weak_img = self.normalize_transform(self.weak_transform(img))
            strong_img = self.normalize_transform(self.strong_transform(img))
            return weak_img, strong_img, noise_target, clean_targets, index
        
        elif self.mode == 'test':
            img, target = self.Data, int(self.ground[index])
            img = self.test_transform(img)
            return img, target
    
    def __len__(self):
        return len(self.path)    

class office_home():
    def __init__(self,root,type,domain,dataset_name,batch_size,img_size,num_workers=8):
        assert dataset_name in ['Art', 'Clipart', 'Product','Real_World']
        self.name = dataset_name
        self.domain = domain # Source or target
        self.root = os.path.join(root,self.name)
        self.batch_size = batch_size
        self.img_size = img_size
        self.type = type
        self.num_workers = num_workers
        self.transform_all = {
            'weak':transforms.Compose([
                   ResizeImage(256),
                   #transforms.RandomCrop(size=224),
                   transforms.RandomHorizontalFlip(p=0.5),
                   ]),
            'strong':transforms.Compose([
                ResizeImage(256),
                #transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(p=0.5),
                RandAugmentMC(n=2,m=10)
            ]),
            'normalize':transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
            'test':transforms.Compose([ResizeImage(256),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        }
    
    def run(self,mode,data_path):     
        if mode == 'test': # when data is target data, data is always test data
            self.train = 'test'
            office_test = office_home_Noisy(
                root=self.root,
                name=self.name,
                mode='test',
                domain=self.domain,
                transform=self.transform_all,# if train=='test' else self.transform_train,
                noise_file=data_path
            )
            office_loader = torch.utils.data.DataLoader(
                dataset=office_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                drop_last=False
            )
            return len(office_test),office_loader
            
        elif mode == 'ALL_Data':
            self.train = 'train'
            office_all_dataset = office_home_Noisy(
                root=self.root,
                name=self.name,
                mode='ALL_Data',
                domain=self.domain,
                transform=self.transform_all,
                noise_file=data_path
            )
            return office_all_dataset
        


if __name__=='__main__':
    
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    name = '/amazon/images/'
    # data_transforms = {
    #         'train': transforms.Compose([
    #         transforms.Resize(size=256),
    #         #transforms.RandomRotation(degrees=15),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(size=256,
    #                               padding=int(256*0.125),
    #                               padding_mode='reflect'),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ]),
    #         'test': transforms.Compose([
    #             transforms.Resize((256, 256)),
    #             #transforms.CenterCrop((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    # }
    # #office = office_Noisy(amazon_root,name='amazon',mode='test',noise_type='sym',ratio=0.5,transform=data_transforms['train'])
    # # print(office.classes)
    # # print(office.class_to_idx)
    # # print(office[2816])
    # # office_loader = torch.utils.data.DataLoader(
    # #     dataset=office,
    # #     batch_size=32,
    # #     shuffle=False,
    # #     num_workers=8
    # # )
    # # iter_office = iter(office_loader)
    # # first = iter_office.next()
    # # img,labels = first
    # # print(img)
    # # print(labels)   
    # Target_loader = office_31(root=root+'/office31/',
    #                                         ratio=0.0,
    #                                         domain='target',
    #                                         level=0,
    #                                         type='sym',
    #                                         dataset_name='webcam',
    #                                         batch_size=32, # 32
    #                                         img_size=256,
    # )
    # loadersource = Target_loader.run('warmup',train='train')
    # for batch_idx, (inputs, labels,index) in enumerate(loadersource):
    #     fig = plt.figure()
    #     inputs = inputs.detach().cpu()
    #     grid = utils.make_grid(inputs)
    #     print("Labels", labels)
    #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #     plt.savefig(dir_path+'/dataset/office31_webcam{}_target.png'.format(0.0,batch_idx))
    #     break
