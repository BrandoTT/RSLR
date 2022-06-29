import os
import random
from matplotlib.pyplot import axis
from numpy.lib.function_base import percentile
import pandas
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import datetime
import torch
from numpy.testing import assert_array_almost_equal
import dataloader
# Dataloader
from dataloader.mnist import get_loader_mnist
from dataloader.mnist_m import get_loader_mnist_m
import torch.nn.functional as F
from model import Feature, Classifier, Discriminator #mnist->mnist_m
import model as Model
import seaborn as sns
import matplotlib.pylab as plt

class record_config(object):
    """用来记录Acc和Loss曲线Log"""
    def __init__(self, log_base,log_name=None,dset=None):
        # 208 Log record
        self.log_base  = log_base
        #self.log_base = self.log_base+'/Log/SynDigits2MNIST/'
        if dset == 'M2Syn': 
            self.log_base = self.log_base+'/Log/MNIST2SynDigits/'
        elif dset == 'office_a_w' or dset == 'office_a_d' or dset == 'office_d_w' or dset == 'office_d_a' or dset == 'office_w_a'\
            or dset == 'office_w_d':
            self.log_base = self.log_base+'/Log/Office/'
        elif dset == 'Syn2M':
            self.log_base = self.log_base+'/Log/SynDigits2MNIST/'
        elif dset == 'cifar2stl' or dset == 'stl2cifar':
            self.log_base = self.log_base+'/Log/cifar_stl/'
        elif dset[0:4] == 'home':
            self.log_base = self.log_base+'/Log/office_home/'

        self.log_name = log_name
        self.log_root = os.path.expanduser(os.path.join('~', self.log_name))
        self.log_root = os.path.join(self.log_base, self.log_name + '_' + datetime.datetime.now().strftime('%Y_%m_%d|%H_%M'))
        print(self.log_root)
    def make_dir(self): 
        '''创建Log目录'''
        try:
            os.makedirs(self.log_root)
        except Exception:
            pass


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

class random_seed(object):
    def __init__(self, seed=None):
        if seed == None:
            self.seed = 1234 #random.randint(1, 10000) # 3293 2708
        else:
            self.seed = seed

    def init_random_seed(self):
        print("random seed is {}".format(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

def enable_cudnn_benchmark():
    """Turn on the cudnn autotuner that selects efficient algorithms."""
    if torch.cuda.is_available():
        cuda = True
        cudnn.benchmark = True
        cudnn.deterministic = True

model_root = '../models'
def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(net.state_dict(),
               os.path.join(model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(model_root,
                                                             filename)))

def get_data_loader(name, train=True, get_dataset=False, batch_size=128, image_size=28):
    """Get data loader by name."""
    print("Load:{}, BatchSize={}, ImageSize={}, If train:{}.".format(name, batch_size, image_size, train))
    if name == "MNIST":
        return get_loader_mnist(train, get_dataset, batch_size=batch_size, image_size=image_size)
    elif name == "MNIST_M":
        return get_loader_mnist_m(train, get_dataset, batch_size=batch_size, image_size=image_size)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def creat_model(model,args):
    gpus = [0, 1]
    print('create model...')
    if model=="mnist_mnist_m":
        feature=Feature()
        feature = feature.cuda()
        classifier_1=Classifier()
        classifier_1 = classifier_1.cuda()
        classifier_2=Classifier()
        classifier_2 = classifier_2.cuda()
        discriminator=Discriminator()
        discriminator = discriminator.cuda()
        return feature,classifier_1,classifier_2,discriminator
    elif model=="mnist_syndigits":
        feature = Model.Feature_Synth().cuda()
        classifier_1 = Model.Classifier_Synth().cuda()
        classifier_2 = Model.Classifier_Synth().cuda()
        cnn_tar = Model.Classifier_Synth().cuda()

        feature = torch.nn.DataParallel(feature,device_ids=gpus)
        classifier_1 = torch.nn.DataParallel(classifier_1,device_ids=gpus)
        classifier_2 = torch.nn.DataParallel(classifier_2,device_ids=gpus)
        cnn_tar = torch.nn.DataParallel(cnn_tar,device_ids=gpus)
       
        return feature, classifier_1, classifier_2, cnn_tar

    elif model=='office31': # office amazon -> webcam
        Feature_L = Model.office_Feature(args).cuda()
        Classifier_1 = Model.office_classifier(args).cuda()
        Classifier_2 = Model.office_classifier(args).cuda()
        CNN_Target = Model.office_classifier(args).cuda()

        Feature_L = torch.nn.DataParallel(Feature_L,device_ids=gpus)
        Classifier_1 = torch.nn.DataParallel(Classifier_1,device_ids=gpus)
        Classifier_2 = torch.nn.DataParallel(Classifier_2,device_ids=gpus)
        CNN_Target = torch.nn.DataParallel(CNN_Target,device_ids=gpus)

        return Feature_L,Classifier_1,Classifier_2,CNN_Target
    
    elif model=="cifar_stl":

        feature = Model.cifar_feature(args).cuda()
        classifier_1 = Model.cifar_classifier(args).cuda()
        classifier_2 = Model.cifar_classifier(args).cuda()
        cnn_tar = Model.cifar_classifier(args).cuda()

        feature = torch.nn.DataParallel(feature,device_ids=gpus)
        classifier_1 = torch.nn.DataParallel(classifier_1,device_ids=gpus)
        classifier_2 = torch.nn.DataParallel(classifier_2,device_ids=gpus)
        cnn_tar = torch.nn.DataParallel(cnn_tar,device_ids=gpus)
       
        return feature, classifier_1, classifier_2, cnn_tar


def linear_rampup(current, warm_up, rampup_length=16, lambda_u=25):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch, warm_up)

class Labeled_Loss(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx

class Unlabeled_Loss(object):
    def __call__(self, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lu, linear_rampup(epoch, warm_up)

class Labeled_Loss_mask(object):
    def __call__(self, outputs_x, targets_x, mask):
        Lx = torch.mean(-torch.sum((F.log_softmax(outputs_x, dim=1) * targets_x) * mask.reshape(len(mask),1), dim=1))
        return Lx

class Unlabeled_Loss_mask(object):
    def __call__(self, outputs_u, targets_u,mask,epoch,warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean(((probs_u - targets_u)*mask.reshape(len(mask),1))**2)
        return Lu, linear_rampup(epoch, warm_up)

############################################# Adding pseudo-labels for Target Dataset after warmming up ######################################

class SubsetSampler(data.Sampler):
    
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in torch.arange(0,len(self.indices)))
    def __len__(self):
        return len(self.indices)

def make_data_loader(dataset,batch=128,shuffle=True, num_worker=8,sampler=None,drop_last=False):
    """Make dataloader from dataset."""
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_worker,
        drop_last=drop_last
    )
    return data_loader

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        # for images, labels in data_loader:
        #     yield (images, labels)
        for w_images, s_images in data_loader:
            yield (w_images,s_images)

def get_sampled_data_loader(dataset, candidates_num, shuffle=False):
    """Get data loader for sampled dataset."""
    print("Sampling subset from target data...")
    total_indices = torch.arange(0, len(dataset))
    if shuffle == True:
        total_indices = torch.randperm(len(dataset)) 
    candidates_num = min(len(dataset), candidates_num) 
    sample_idx = total_indices.narrow(0, 0, int(candidates_num)).long()
    sampler = SubsetSampler(sample_idx)
    un_sample_idx = torch.tensor([i for i in total_indices if i not in sample_idx]).long()
    make_loader = make_data_loader(dataset,batch=128,shuffle=False,sampler=sampler,drop_last=False)
    return total_indices,sample_idx,un_sample_idx,make_loader



class DummyDataset_tar(data.Dataset):
    def __init__(self,original_dataset,excerpt,pseudo_labels=[],l_tar_weight=[],label=False):
        """Init DummyDataset"""
        super(DummyDataset_tar,self).__init__()
        self.label = label
        if self.label == True:
            assert len(excerpt) == len(pseudo_labels),\
                "Size of excerpt images({}) and pseudo-labels ({}) aren't equal." \
                .format(len(excerpt), len(pseudo_labels))
            self.pseudo_labels = pseudo_labels
            self.label_weight = l_tar_weight
        self.dataset = original_dataset # original_dataset: w_imges, s_imges
        self.excerpt = excerpt

    def __getitem__(self, index):
        """Get images and labels for data loader"""
        if self.label == True:
            w_images, s_images, clean_targets, clean_targets, idx = self.dataset[self.excerpt[index]]
            return (w_images, s_images, clean_targets, int(self.pseudo_labels[index]), float(self.label_weight[index])) 
        else:
            # without using
            w_images,s_images,_,_,_ = self.dataset[self.excerpt[index]]
            return (w_images,s_images)

    def __len__(self) -> int:
        return len(self.excerpt)

def get_dummy_tar(original_dataset,labelled_idx, unlabelled_idx,pseudo_labels,l_tar_weight,batch_size=128):
    """Get DummyDataset loader."""
    labeled_dataset = DummyDataset_tar(original_dataset,labelled_idx,pseudo_labels,l_tar_weight,label=True)
    unlabeled_dataset = DummyDataset_tar(original_dataset,unlabelled_idx)
    return labeled_dataset,unlabeled_dataset
    

class DummyDataset_sou(data.Dataset):
    def __init__(self,original_dataset,excerpt,pseudo_labels=[],l_tar_weight=[],label=False):
        """Init DummyDataset"""
        super(DummyDataset_sou,self).__init__()
        self.label = label
        assert len(excerpt) == len(l_tar_weight),\
            "Size of excerpt images({}) and pseudo-labels ({}) aren't equal." \
            .format(len(excerpt), len(l_tar_weight))
                
        self.dataset = original_dataset # original_dataset: w_imges, s_imges
        self.excerpt = excerpt
        self.label_weight = l_tar_weight
        if self.label == False:
            self.pseudo_labels = pseudo_labels
           

    def __getitem__(self, index):
        '''Get images and labels for data loader'''
        '''balanced sampling strategy'''
        if self.label == True:
            weak_img, strong_img, noise_target, clean_targets, _ = self.dataset[self.excerpt[index]] 
            return (weak_img, strong_img,clean_targets, noise_target, float(self.label_weight[index]))
        else:
            weak_img, strong_img, noise_target, clean_targets, _ = self.dataset[self.excerpt[index]]
            return (weak_img, strong_img,clean_targets, int(self.pseudo_labels[index]), float(self.label_weight[index]))

    def __len__(self) -> int:
        return len(self.excerpt)


def get_dummy_sou(original_dataset,labelled_idx,unlabelled_idx,l_weight,pse_weight,batch_size=None,pseudo_labels=None):
    """Get DummyDataset loader."""
    labeled_dataset = DummyDataset_sou(original_dataset,labelled_idx,l_tar_weight=l_weight,label=True)
    unlabeled_dataset = DummyDataset_sou(original_dataset,unlabelled_idx,pseudo_labels=pseudo_labels,l_tar_weight=pse_weight,label=False)

    return labeled_dataset,unlabeled_dataset

def get_balance_sou(sou_dataset,percentile_=None):
    '''re sampling source labelled data by ordering clean prob'''
    total_long = len(sou_dataset)
    dataloader = make_data_loader(sou_dataset,num_worker=8,batch=total_long,shuffle=False,drop_last=False)
    for step, (w_img, s_img, _, noise_target, clean_prob) in enumerate(dataloader):
        every_limit = (len(noise_target) * percentile_) / 31

        Final_Index = []
        clean_prob = clean_prob.cpu().numpy()
        sort_all_idx = np.argsort(-clean_prob)
        num_limit = np.zeros(31)
        for i in range(len(sort_all_idx)):
            class_ = noise_target[sort_all_idx[i]]
            if num_limit[class_] <= every_limit:
                Final_Index.append(sort_all_idx[i])
                a = num_limit[class_]
                a+=1
                num_limit[class_] = a
    print('num_limit',num_limit)        
    Final_Index = torch.tensor(Final_Index)
    sampler = SubsetSampler(Final_Index)
    return_loader = make_data_loader(sou_dataset,num_worker=8,batch=16,drop_last=True,sampler=sampler)
    '''check noise ratio and imbalance ratio'''
    # imbalance ratio && noise check
    num_iter = len(return_loader)
    data_iter = iter(return_loader)
    class_num = np.zeros(31)
    curr = 0.
    total_num = 0.
    for i in range(num_iter):
        batch = data_iter.next()
        _,_,ground,pseudo,_ = batch
        total_num += ground.size(0)
        pseudo = pseudo.tolist()
        ground = ground.tolist()
        for label in range(31):
            num = pseudo.count(label)
            a = class_num[label]
            a+=num
            class_num[label] = a
        for i in range(len(ground)):
            if int(ground[i]) == int(pseudo[i]):
                curr += 1.0
    acc_noise_check = (1*curr/total_num)
    imbalance_r = class_num # epcoh labelled true balance ratio
    print('imbalance',imbalance_r)
    return return_loader,imbalance_r.tolist(),acc_noise_check


def guess_pseudo_labels(args,out_1,out_2,out_F_1_s=None,out_F_2_s=None,IF_office=False,Ground_test=None,percentile_=None,domain=None,tar_class_ratio=None):
    """len(out) = len(target_dataset), shuffle=False"""
    '''
    sou 和 tar 采用不同的加pseudo-labels的策略
    '''
     # 每个类里
    Total_List = torch.arange(0,len(out_1)).cuda()
    out_sum = out_1 + out_2 # 不看两个网络的一致性，而是整合两个网络的意见
    soft_out = torch.softmax(out_sum, -1)
    max_class_probs, predict_class = torch.max(soft_out,-1)

    # # Ours -------------------------------------
    out_sum_s = out_F_1_s + out_F_2_s
    soft_out_s = torch.softmax(out_sum_s, -1)
    max_class_probs_s,predict_class_s = torch.max(soft_out_s,-1)
    idx = []
    for i in range(len(out_sum)):
        label1 = predict_class[i]
        label2 = predict_class_s[i]
        if label1 == label2:
            idx.append(i)
    total_equal_idx = torch.tensor(idx)
    #-------------- sample equally-----------------
    max_class_probs = (max_class_probs + max_class_probs_s) / 2.
    choose_limit = int(len(total_equal_idx)) * percentile_

    every_limit = choose_limit / args.num_class

    probs_np = max_class_probs.cpu().numpy()
    sort_all_index = np.argsort(-probs_np)
    Credi_idx = torch.tensor([i for i in sort_all_index if i in total_equal_idx]).cuda()
    labelled_index = []
    num_limit = np.zeros(args.num_class)
    for i in range(0, len(Credi_idx)):
        class_probs = max_class_probs[Credi_idx[i]]
        class_predict = predict_class[Credi_idx[i]]
        if (num_limit[class_predict] <= every_limit):
            labelled_index.append(Credi_idx[i])
            a = num_limit[class_predict]
            a+=1
            num_limit[class_predict] = a
        #total_equal_idx = Total_List
    labelled_index = torch.tensor(labelled_index)
    
    print('Final number of choosed:',len(labelled_index))
    Total_List = Total_List.cpu().numpy()
    unlabl_idx = torch.tensor([i for i in Total_List if i not in labelled_index])
    pseudo_labels = torch.tensor([predict_class[i] for i in labelled_index])
    l_tar_weight = torch.tensor([max_class_probs[i] for i in labelled_index])
    # print(pseudo_labels)
    # exit()
    return labelled_index, pseudo_labels, l_tar_weight, unlabl_idx#, Recall_scores


def generate_labels(args,F,C_1,C_2,dataset,domain='target',IF_office=None,sou_u_index=None,percentile_=None,tar_class_ratio=None):
    
    F.eval()
    C_1.eval()
    C_2.eval()
    IF_office = IF_office
    data_loader = make_data_loader(dataset,num_worker=8,batch=128,shuffle=False,drop_last=False)
    out_F_1_total = None
    out_F_2_total = None
    out_F_1_total_strong = None
    out_F_2_total_strong = None
    for step, (w_img,s_img,_,_,index) in enumerate(data_loader):
        w_img = make_cuda(w_img)
        s_img = make_cuda(s_img)
        with torch.no_grad():
            feature = F(w_img)
            out_F_1 = C_1(feature)
            out_F_2 = C_2(feature)
            feature_s = F(s_img)
            out_F_1_s = C_1(feature_s)
            out_F_2_s = C_2(feature_s)
            if step == 0:
                out_F_1_total = out_F_1.data.cuda()
                out_F_2_total = out_F_2.data.cuda()
                out_F_1_total_strong = out_F_1_s.data.cuda()
                out_F_2_total_strong = out_F_2_s.data.cuda()
            else:
                out_F_1_total = torch.cat( #concatenate two tensor
                    [out_F_1_total, out_F_1.data.cuda()], 0)
                out_F_2_total = torch.cat(
                    [out_F_2_total, out_F_2.data.cuda()], 0)
                out_F_1_total_strong = torch.cat(
                    [out_F_1_total_strong, out_F_1_s.data.cuda()], 0)
                out_F_2_total_strong = torch.cat(
                    [out_F_2_total_strong, out_F_2_s.data.cuda()], 0)
    if domain == 'source':
        print('unlabelled idx length:%d'%len(sou_u_index))
        out_F_1_total = out_F_1_total.tolist()
        out_F_2_total = out_F_2_total.tolist()
        out_F_1_total = [out_F_1_total[i] for i in sou_u_index]
        out_F_1_total = torch.tensor(out_F_1_total)
        out_F_2_total = [out_F_2_total[i] for i in sou_u_index]
        out_F_2_total = torch.tensor(out_F_2_total)

        out_F_1_total_strong = out_F_1_total_strong.tolist()
        out_F_2_total_strong = out_F_2_total_strong.tolist()
        out_F_1_total_strong = [out_F_1_total_strong[i] for i in sou_u_index]
        out_F_1_total_strong = torch.tensor(out_F_1_total_strong)
        out_F_2_total_strong = [out_F_2_total_strong[i] for i in sou_u_index]
        out_F_2_total_strong = torch.tensor(out_F_2_total_strong)


    print("Unlabelled process domain:{} Length:{}".format(domain,len(out_F_1_total)))
    
    label_idx, pseudo_labels, l_tar_weight , unlabl_idx = \
        guess_pseudo_labels(args,out_F_1_total,out_F_2_total,out_F_1_s=out_F_1_total_strong,out_F_2_s=out_F_2_total_strong,IF_office=IF_office,percentile_=percentile_,domain=domain,tar_class_ratio=tar_class_ratio)
    if domain=='source': # source just subset of unlabelled samples.
        label_idx = [sou_u_index[i] for i in label_idx]
        unlabl_idx = [sou_u_index[i] for i in unlabl_idx]
    return label_idx, pseudo_labels, l_tar_weight , unlabl_idx

#################################################################################################################################################
class NegEntropy(object): # -H
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


""" make preparation for office-31"""

def get_name(args):
    # --- office 31 ----
    if args.dset == 'office_a_w':
        source_name = 'amazon'
        target_name = 'webcam'
    elif args.dset == 'office_a_d':
        source_name = 'amazon'
        target_name = 'dslr'
    elif args.dset == 'office_w_d':
        source_name = 'webcam'
        target_name = 'dslr'
    elif args.dset == 'office_w_a':
        source_name = 'webcam'
        target_name = 'amazon'
    elif args.dset == 'office_d_w':
        source_name = 'dslr'
        target_name = 'webcam'
    elif args.dset == 'office_d_a':
        source_name = 'dslr'
        target_name = 'amazon'
    elif args.dset == 'office_a_a':
        source_name = 'amazon'
        target_name = 'amazon'
    ## 
    elif args.dset == 'M2Syn':
        source_name='Mnist'
        target_name='Syndigits'
    elif args.dset == 'Syn2M':
        source_name='Syndigits'
        target_name='Mnist'
    ## cifar and stl
    elif args.dset == 'cifar2stl':
        source_name='cifar'
        target_name='stl'
    elif args.dset == 'stl2cifar':
        source_name='stl'
        target_name='cifar'
    # --- office home ---
    # Art:Ar Clipart:Cl Product:Pr Real_World:Rw
    elif args.dset == 'home_Rw_Pr':
        source_name = 'Real_World'
        target_name = 'Product'
    else:
        sour = args.dset[5:7]
        print(sour)
        if sour == 'Rw':
            source_name = 'Real_World'
        elif sour == 'Ar':
            source_name = 'Art'
        elif sour == 'Cl':
            source_name = 'Clipart'
        elif sour == 'Pr':
            source_name = 'Product'
        tar = args.dset[8:10]
        if tar == 'Rw':
            target_name = 'Real_World'
        elif tar == 'Ar':
            target_name = 'Art'
        elif tar == 'Cl':
            target_name = 'Clipart'
        elif tar == 'Pr':
            target_name = 'Product'
    
    return source_name, target_name


def get_source_index(pred,probility):
    '''Get index of source labelled and unlabelled'''
    pred_l_idx = pred.nonzero()[0]
    pred_un_idx = (1-pred).nonzero()[0]
    prob = [probility[i] for i in pred_l_idx]
    return pred_l_idx, pred_un_idx, prob

def multiclass_noisify(y, T, random_state=0):
    """
    Flip classes according to transition probability matrix T.
    """
    y = np.asarray(y)
    #print(np.max(y), T.shape[0])
    assert T.shape[0] == T.shape[1]
    assert np.max(y) < T.shape[0]
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))
    assert (T >= 0.0).all()
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        #print(i, T[i,:])
        flipped = flipper.multinomial(1, T[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def sym_noise_new(noise_ratio,num_class=65,true_targets=None,random_state=None):
    '''
    flip y in the symmetric way
    '''
    r = noise_ratio
    ntm = np.ones((num_class,num_class))
    ntm = (r / (num_class - 1)) * ntm
    if r > 0.0:
        ntm[0,0] = 1. - r
        for i in range(1, num_class-1):
            ntm[i,i] = 1. - r
        ntm[num_class-1,num_class-1] = 1. - r
        y_noisy = multiclass_noisify(true_targets,ntm,random_state=random_state)
        actural_noise_rate = (true_targets != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noise rate is {}'.format(actural_noise_rate))
    else:
        y_noisy = true_targets
        return y_noisy

    return y_noisy.tolist()

def asym_noise_new(noise_ratio,num_class=10,true_targets=None,random_state=None):
    '''
    flip in the pair
    '''
    ntm = np.eye(num_class)
    r = noise_ratio
    if r > 0.0:
        ntm[0,0],ntm[0,1] = 1. - r, r
        for i in range(1, num_class-1):
            ntm[i,i],ntm[i,i+1] = 1. - r, r
        ntm[num_class-1,num_class-1], ntm[num_class-1,0] = 1. - r, r
        y_noisy = multiclass_noisify(true_targets,ntm,random_state=random_state)
        actural_noise_rate = (true_targets != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noise rate is {}'.format(actural_noise_rate))
    else:
        y_noisy = true_targets
        return y_noisy
        
    return y_noisy.tolist()





# Focal Loss for unimbalanced
class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
    
    def cal_α(self,class_num,target,focal_weight):
        number_class = np.zeros(class_num)
        #print(number_class)
        for i in range(len(target)):
            label = target[i]
            curr = int(number_class[label])
            curr+=1
            number_class[label] = curr
        #print(number_class)
        number_class = torch.tensor(number_class)
        
        a = np.zeros(len(target))
        for i in range(len(a)):
            class_ = int(target[i])
            # class_weight = soft_weight[class_]
            # # 这个权重，根据每次检测完之后，作为一个干净的检测
            # if class_ == 4:
            #     class_weight = 0.75
            # else:
            #     class_weight = 0.25
            class_weight = focal_weight[class_]
            a[i] = class_weight
        return torch.tensor(a)


    def forward(self, predict, target,focal_weight):
        #print(target)
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        #ids = target.view(-1, 1)
        #alpha = self.alpha[ids.data.view(-1)].view(-1,1).cuda() # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        # α是各个类别出现的比重，对每个batch各个类别出现的次数做一个统计
        alpha = self.cal_α(self.class_num,target,focal_weight)
        alpha = alpha.cuda()
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log().cuda()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def linear_rampup(now_iter, total_iter=200):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)
    
def get_focal_alpha(epoch,sour_dataset_l,num_class):
    #print(' === compute focal alpha in epoch:%d ==='%epoch)
    f_test_sou_loader = make_data_loader(sour_dataset_l,shuffle=False,num_worker=8,batch=len(sour_dataset_l),drop_last=False)
    iter_test = iter(f_test_sou_loader)
    _,_,targets,_ = iter_test.next()
    # print('Clean source targets\n',targets.tolist())
    # # 输出一个向量，对应着不同的类别的不同的weight
    epoch_sou_l_len = len(targets)  # 这是总的样本量
    class_number = np.zeros(num_class)
    # for i in range(len(targets)):
    #     label = targets[i]
    #     a = int(class_number[label])
    #     a+=1
    #     class_number[label] = a
    weight_class = np.zeros(num_class)
    for label in range(num_class):
        label_num = class_number[label]
        #label_weight = (1 - (label_num / epoch_sou_l_len))**3
        label_weight = label_num / epoch_sou_l_len
        weight_class[label] = label_weight
    print('weight class:\n',weight_class)
    return weight_class


def drawing(root,dis):
    csv_root = root+'/office_label_distribution.csv'
    df = pandas.read_csv(csv_root)
    amazon = dis[0]
    dslr = dis[1]
    webcam = dis[2]
    sns.histplot(amazon,bins=31,color='c',label='amazon')
    sns.histplot(dslr,bins=31,color='k',label='dslr')
    sns.histplot(webcam,bins=31,color='crimson',label='webcam')
    plt.legend(loc='upper right')
    plt.savefig(root+'/office31_label_dis.png')
    exit()


def cal_Recall_Precision(args,epoch,dataset):
    r'''
    caculate scores of recall and precision for source and target dataset pseudo-labels samples
    '''
    batch = len(dataset)
    data_loader = make_data_loader(dataset,num_worker=8,batch=batch,shuffle=False,drop_last=False)
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    class_num = np.zeros(args.num_class)
    for i in range(num_iter):
        batch = data_iter.next()
        _,_,ground,pseudo,_ = batch # all index
        ground = ground.tolist()
        pseudo = pseudo.tolist()
        # ----- Recall for every class -----
        Recall_scores = np.zeros(args.num_class)
        num = np.zeros(args.num_class)
        for i in range(len(ground)):
            label = int(ground[i])
            label_pse = int(pseudo[i])
            if label == label_pse:
                a = num[label]
                a+=1
                num[label]=a
        for label in range(args.num_class):
            class_num[label] = ground.count(label)
            if class_num[label] == 0:
                print('without predict class{}'.format(label))
                Recall_scores[label] = 0
            else:
                Recall_scores[label] = num[label] / class_num[label]
        # ---- precision for every class -----
        Precision_scores = np.zeros(args.num_class)
        class_num = np.zeros(args.num_class) # 预测为c类的全部个数
        for label in range(args.num_class):
            pre_c_num = pseudo.count(label) # total num of class c prediction
            class_num[label] = pre_c_num # total num
            num = 0.
            for i in range(len(ground)):
                if int(pseudo[i]) == label:
                    if int(ground[i])== label:
                        num+=1
            if class_num[label] == 0:
                Precision_scores[label] = 0
            else:
                Precision_scores[label] = num / class_num[label]
        
        Recall_scores = Recall_scores.tolist()
        Precision_scores = Precision_scores.tolist()

    return Recall_scores, Precision_scores


def cal_pse_acc(args,dataset):
    batch = len(dataset)
    data_loader = make_data_loader(dataset, batch=batch,shuffle=False,drop_last=False)
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    totalnum = batch
    for i in range(num_iter):
        batch = data_iter.next()
        _,_,ground,pseudo,_ = batch # all index
        ground = ground.tolist()
        pseudo = pseudo.tolist()
        # print(ground)
        # print(pseudo)
        # ---- pseudo acc in every epoch ---- 
        curr = 0.
        for i in range(len(ground)):
            if int(ground[i]) == int(pseudo[i]):
                curr += 1.
        acc_pseudo = (1*curr/totalnum)

    return acc_pseudo


def detect_acc(epoch,source_dataset):
    '''evaluate how many noise labels could be checked out?'''
    batch_num = len(source_dataset)
    data_loader = make_data_loader(source_dataset,num_worker=8,batch=batch_num,shuffle=False,drop_last=False)
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        batch = data_iter.next()
        _,_,ground,pseudo,w_x = batch
        # ---- pseudo acc in every epoch ---- 
        ground = ground.tolist()
        pseudo = pseudo.tolist()
        curr = 0.
        for i in range(len(ground)):
            if int(ground[i]) == int(pseudo[i]):
                curr += 1.
        acc_noise_check = (1*curr/batch_num)
        
    return acc_noise_check

def get_imbalance_ratio(args,dataset):
    length = len(dataset)
    data_loader = make_data_loader(dataset,num_worker=8,batch=length,shuffle=False,drop_last=False)
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        batch = data_iter.next()
        _,_,_,pseudo,_ = batch
        pseudo = pseudo.tolist()
        #print(pseudo)
        class_num = np.zeros(args.num_class)
        for label in range(args.num_class):
            class_num[label] = pseudo.count(label)
        # max_num = 0
        # min_num = 0
        # for i in range(len(class_num)):
        #     if i==0:
        #         a = int(class_num[i])
        #         max_num = a
        #         min_num = a
        #     else:
        #         a = int(class_num[i])
        #         if a > max_num:
        #             max_num = a
        #         elif a < min_num:
        #             min_num = a
        # if min_num==0:
        #     min_num = 1 # ??
        #imbalance_r = max_num / min_num
        imbalance_r = class_num
    return imbalance_r.tolist()