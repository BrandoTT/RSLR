"""
Syndigits is Target Data
"""
# dataset custom
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.utils import download_url
import torch
from scipy.io import loadmat
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import torch
from torchvision import datasets, transforms, utils
import os
import matplotlib.pyplot as plt
import random
from Config.utils import sym_noise_new,asym_noise_new
#from Config.utils import make_data_loader
from Config.utils import make_data_loader
from dataloader.randaugment import RandAugmentMC
from Record.Drawing import data_labeldistribution
#from randaugment import RandAugmentMC
# # SynDigits is Target Data that is no need to add noise
# class _BaseDataset(Dataset):
#     urls = None
#     # training_file = None
#     # test_file     = None
#     training_file = 'synth_train_32x32.mat'
#     test_file = 'synth_test_32x32.mat'
    
#     def __init__(self, root, split = 'train', transform = None,
#                  label_transform = None, download=False):

#         super().__init__()
        
#         self.root = root
#         self.which = split 
        
#         self.transform = transform
#         self.label_transform = label_transform

#         if download:
#             self.download()

#         self.get_data(self.which)
        
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, index):
        
#         #x = Image.fromarray(self.images[index])
#         img, target = self.images[index], int(self.labels[index])
#         img = Image.fromarray(np.transpose(img, (1, 2, 0)))
#         #target = int(self.labels[index])
        
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.label_transform is not None:
#             target= self.label_tranform(target)
            
#         return img, target

#     def get_data(self, name):
#         """Utility for convenient data loading."""
#         if name in ['train', 'unlabeled']:
#             self.extract_images_labels(os.path.join(self.root, self.training_file))
#         elif name == 'test':
#             self.extract_images_labels(os.path.join(self.root, self.test_file))

#     def extract_images_labels(self, filename):
#         raise NotImplementedError

#     def _check_exists(self):
#         return os.path.exists(os.path.join(self.root, self.training_file)) and \
#             os.path.exists(os.path.join(self.root, self.test_file))

#     def download(self):
#         if self._check_exists():
#             return

#         os.makedirs(self.root, exist_ok = True)

#         for url in self.urls:
#             filename = url.rpartition('/')[2]
#             file_path = os.path.join(self.root, filename)
#             download_url(url, root=self.root,
#                          filename=filename, md5=None)
#         print('Done!')
# class Synth(_BaseDataset):
#     """ Synthetic images dataset
#     """

#     num_labels  = 10
#     #image_shape = [32, 32, 3]
    
#     urls = {
#         "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32.mat?raw=true", 
#         "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32.mat?raw=true"
#     }
#     training_file = 'synth_train_32x32.mat'
#     test_file = 'synth_test_32x32.mat'
    
#     def extract_images_labels(self, filename):
#         print('Extracting', filename)
#         mat = loadmat(filename)
#         self.images = mat['X'].transpose((3,2,0,1))
#         self.labels = mat['y'].squeeze()


  

# class SynthSmall(_BaseDataset):

#     """ Synthetic images dataset
#     """

#     num_labels  = 10
#     image_shape = [16, 16, 1]
    
#     urls = {
#         "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32_small.mat?raw=true", 
#         "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32_small.mat?raw=true"
#     }
#     training_file = 'synth_train_32x32_small.mat?raw=true'
#     test_file = 'synth_test_32x32.mat_small?raw=true'
    
#     def extract_images_labels(self, filename):
#         print('Extracting', filename)

#         mat = loadmat(filename)

#         self.images = mat['X'].transpose((3,0,1,2))
#         self.labels = mat['y'].squeeze()

# class syndigits_dataloader():
#     def __init__(self,root,batch_size,img_size,num_workers=8):
#         self.name = "SynthDigits"
#         self.root = os.path.join(root,self.name)
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.num_workers = num_workers
#         self.transform_train = transforms.Compose([
#             transforms.Resize(self.img_size),
#             # transforms.RandomCrop(self.img_size, padding=4),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         self.transform_test = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#     def run(self, mode):
#         if mode == "train":
#             self.train = 'train'
#             Syn_dataset = Synth(
#                 root=self.root,
#                 split=self.train,
#                 transform=self.transform_train,
#                 download=False
#             )
#             Syn_dataloader = torch.utils.data.DataLoader(
#                 dataset=Syn_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=8
#             )
#             return Syn_dataloader
#         elif mode == "test":
#             self.train = "test"
#             Syn_dataset = Synth(
#                 root=self.root,
#                 split=self.train,
#                 transform=self.transform_test,
#                 download=False
#             )
#             Syn_dataloader = torch.utils.data.DataLoader(
#                 dataset=Syn_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=8
#             )
#             return Syn_dataloader

#         elif mode == "warmup":
#             self.train = "train"
#             Syn_dataset = Synth(
#                 root=self.root,
#                 split=self.train,
#                 transform=self.transform_train,
#                 download=False
#             )
#             Syn_dataloader = torch.utils.data.DataLoader(
#                 dataset=Syn_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=8
#             )
#             return Syn_dataloader




# SynDigits is Source Data which is need to be added noise
class _BaseDataset(Dataset):
    urls = None
    # training_file = None
    # test_file     = None
    training_file = 'synth_train_32x32.mat'
    test_file = 'synth_test_32x32.mat'
    
    def __init__(self, root, split = 'train', transform = None,
                 label_transform = None, download=False,pred=None,prob=None,mode=None):

        super().__init__()
        
        self.root = root
        self.which = split 
        
        self.transform = transform
        self.label_transform = label_transform
        self.mode = mode
        print(self.mode)
        if self.mode == 'labeled':
            self.pred_idx = pred.nonzero()[0]
        elif self.mode == 'unlabeled':
            self.pred_idx = (1-pred).nonzero()[0]
        if download:
            self.download()

        self.get_data(self.which)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        #x = Image.fromarray(self.images[index])
        img, target = self.images[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        #target = int(self.labels[index])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            target= self.label_tranform(target)
            
        return img, target

    def get_data(self, name):
        """Utility for convenient data loading."""
        if name in ['train', 'unlabeled']:
            self.extract_images_labels(os.path.join(self.root, self.training_file),split='train')
        elif name == 'test':
            self.extract_images_labels(os.path.join(self.root, self.test_file),split='test')

    def extract_images_labels(self, filename,split):
        raise NotImplementedError

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok = True)

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root,
                         filename=filename, md5=None)
        print('Done!')

class Synth(_BaseDataset):
    """ Synthetic images dataset
    """

    num_labels  = 10
    #image_shape = [32, 32, 3]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32.mat?raw=true"
    }
    training_file = 'synth_train_32x32.mat'
    test_file = 'synth_test_32x32.mat'
    
    def extract_images_labels(self, filename,split):
        #print('Extracting', filename)
        print("Extracting SYDN...")
        mat = loadmat(filename)
        if split=='train':
            self.images = mat['X'].transpose((3,2,0,1))
            self.labels = mat['y'].squeeze()

        elif split=='test':
            self.images = mat['X'].transpose((3,2,0,1))
            self.labels = mat['y'].squeeze()

class SynthSmall(_BaseDataset):

    """ Synthetic images dataset
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32_small.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32_small.mat?raw=true"
    }
    training_file = 'synth_train_32x32_small.mat?raw=true'
    test_file = 'synth_test_32x32.mat_small?raw=true'
    
    def extract_images_labels(self, filename):
        print('Extracting', filename)

        mat = loadmat(filename)

        self.images = mat['X'].transpose((3,0,1,2))
        self.labels = mat['y'].squeeze()

class SynthDigits(Synth):
    def __init__(self,root,mode,ratio,noise_level,noise_type,split='train',transform=None,target_transform=None,download=False,pred=[], probability=[], noise_file=''):
        super().__init__(root, split=split, transform=transform, label_transform=target_transform, download=download,pred=pred,prob=probability,mode=mode)
        self.num_classes = 10
        self.mode = mode # all, test
        self.ratio = ratio
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.name = 'syndigits'
        if self.mode == 'ALL_Data':
            self.weak_transform = transform['weak']
            self.strong_transform = transform['strong']
            self.normalize_transform = transform['normalize']
        else:
            self.test_transform = transform['test']
        
        # noise_label = json.load(open(noise_file, "r"))
        # data_labeldistribution(root,self.labels,noise_label,self.name)
        # exit()
        if noise_type=="asym":
            ntm_asym = np.eye(self.num_classes) * (1 - self.ratio)
            row_indices = np.arange(self.num_classes)
            for i in range(self.num_classes):
                ntm_asym[i][np.random.choice(row_indices[row_indices != i])] = self.ratio
        if self.mode=='test':
            self.TestData = torch.tensor(self.images) 
            self.test_label = torch.tensor(self.labels)
        else:
            self.images = self.images.tolist()
            self.labels_list = self.labels.tolist()
            print("SynDigits Noise numbers:",self.ratio*len(self.images))
            if os.path.exists(noise_file):
                print("{} noise labels has been made\n".format(self.ratio))
                noise_label = json.load(open(noise_file, "r"))
            else:
                if self.noise_type == 'sym':
                    noise_label = sym_noise_new(self.ratio,self.num_classes,self.labels_list)
                elif self.noise_type == 'asym':
                    noise_label = asym_noise_new(self.ratio,self.num_classes,self.labels_list)
                #noise_label = sym_noise_new(self.ratio,self.num_classes,self.labels_list)
                print("Saving noisy labels to %s ... "%noise_file)    
                json.dump(noise_label, open(noise_file, "w"))
                exit()

            Data = torch.tensor(self.images)
            Label = torch.tensor(noise_label)

            self.TrainData = Data
            self.noise_label = Label

    def __len__(self):
        if self.mode == 'test':
            return len(self.TestData)
        else:
            return len(self.TrainData)
    def __getitem__(self, index):

        if self.mode == 'test':
            img, target = self.TestData[index], int(self.test_label[index])
            img = Image.fromarray(np.transpose(img.numpy().astype(np.uint8), (1, 2, 0)))
            img = self.test_transform(img)
            return img, target

        elif self.mode == 'ALL_Data':
            img, noise_labels, ground = self.TrainData[index], int(self.noise_label[index]),int(self.labels[index])
            img = Image.fromarray(np.transpose(img.numpy().astype(np.uint8), (1, 2, 0)))
            weak_img = self.normalize_transform(self.weak_transform(img))
            strong_img = self.normalize_transform(self.strong_transform(img))
            return weak_img, strong_img, noise_labels, ground, index






class syndigits_dataloader():
    def __init__(self,root,ratio,level,type,batch_size,img_size,num_workers=8,noise_file=''):
        """
        ratio: Propotation of noise samples
        level: Noise level
        type: Noise type {uniform, Asym}
        """
        self.name = "SynthDigits"
        self.root = os.path.join(root,self.name)
        self.ratio = ratio
        #print('----------------------',self.ratio)
        self.level = level
        self.type = type
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.noise_file = noise_file
        #print('-----------------------',self.noise_file)
        # Data Augmentation for Unlabeled Data include Weak and Strong like FixMatch
        self.transform_all = {
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

    def run(self, mode):
        if mode == "test":
            self.train = "test"
            test_Syn_dataset = SynthDigits(
                root=self.root,
                mode='test',
                split=self.train,
                ratio=self.ratio,
                noise_level=self.level,
                noise_type=self.type,
                transform=self.transform_all,
                download=False,
                noise_file=self.noise_file
            )
            test_Syn_dataloader = torch.utils.data.DataLoader(
                dataset=test_Syn_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                drop_last=True
            )
            return len(test_Syn_dataset),test_Syn_dataloader

        elif mode == "ALL_Data":
            self.train = "train"
            Syn_all_dataset = SynthDigits(
                root=self.root,
                mode=mode, # ALL_Data
                ratio=self.ratio,
                noise_level=self.level,
                noise_type=self.type,
                split=self.train,
                transform=self.transform_all,
                download=False,
                noise_file=self.noise_file
            )
            
            return Syn_all_dataset





if __name__=="__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    ratio=0.0
    loader_source = syndigits_dataloader(root=root,
                                                        ratio=ratio,
                                                        level=0.0,
                                                        type='sym',
                                                        batch_size=32,
                                                        img_size=32,
                                                        noise_file='%s/%.1f_%s.json'%(root+'/SynDigits_NoiseLabels',ratio,"small_sym")
    )
    data_syn = loader_source.run('ALL_Data')
    dataloader = make_data_loader(data_syn,batch=32,shuffle=True)
    for batch_idx, (weak,strong,labels,index) in enumerate(dataloader):
        fig = plt.figure()
        weak = weak.detach().cpu()
        strong = strong.detach().cpu()
        grid = utils.make_grid(weak)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig(dir_path+'/dataset/syndigit_w_{}.png'.format(ratio))

        grid = utils.make_grid(strong)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig(dir_path+'/dataset/syndigit_s_{}.png'.format(ratio))
        print("Labels", labels)
        break