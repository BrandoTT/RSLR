import gzip
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets.utils import download_url

import torch

from scipy.io import loadmat

class _BaseDataset(Dataset):

    urls = None
    # training_file = None
    # test_file     = None
    training_file = 'synth_train_32x32.mat'
    test_file = 'synth_test_32x32.mat'
    
    def __init__(self, root, split = 'train', transform = None,
                 label_transform = None, download=False):

        super().__init__()
        
        self.root = root
        self.which = split 
        
        self.transform = transform
        self.label_transform = label_transform

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
            self.extract_images_labels(os.path.join(self.root, self.training_file))
        elif name == 'test':
            self.extract_images_labels(os.path.join(self.root, self.test_file))

    def extract_images_labels(self, filename):
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
    
    def extract_images_labels(self, filename):
        print('Extracting', filename)
        mat = loadmat(filename)
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
