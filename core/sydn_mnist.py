import random
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import sys
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.optim as optim
import numpy as np
from torch.utils.data import dataloader,ConcatDataset
# network
import argparse
from Config.utils import creat_model, random_seed, enable_cudnn_benchmark,make_cuda, record_config, \
    creat_model, NegEntropy,get_name,Labeled_Loss,make_data_loader,get_dummy_tar,get_dummy_sou,generate_labels, \
        make_data_loader, get_source_index,cal_Recall_Precision,detect_acc,get_imbalance_ratio,cal_pse_acc,get_balance_sou
import Config.config as cf
import torch
from sklearn.mixture import GaussianMixture
from dataloader import mnist_noise_loader as loader_target
from dataloader import syndigits_noise_loader as loader_source
import matplotlib.pyplot as plt
import torch.nn as nn
import json
from tensorboardX import SummaryWriter
from torchvision import utils
parser = argparse.ArgumentParser(description=' -- NoiseUDA  mnist -> synthDigits-- ')
#parser.add_argument('--beta', default=1.0, type=float, choices=[0.0, 0.1 ,0.2, 0.3, 0.45, 0.6, 1.0])
parser.add_argument('--lr', default=0.02, type=float, choices=[1e-3, 1e-2, 1e-1, 0.002, 0.02])
parser.add_argument('--batch_size_s', default=128, type=int, choices=[32,64,128,256])
parser.add_argument('--batch_size_t', default=128, type=int, choices=[32,64,128,256])
parser.add_argument('--image_size', default=32, type=int, choices=[28, 32])
parser.add_argument('--Epoch', default=100, type=int, choices=[1, 2, 3, 30,  50, 100])
parser.add_argument('--Noise', default='sym', type=str, choices=['asym','sym'])
parser.add_argument('--dset', default='Syn2M', type=str, choices=['S2M','M2m', 'Syn2SVHN', 'Syn2M', 'M2Syn'])
parser.add_argument('--warmup', default=5, type=int, choices=[0, 1, 2, 3, 5, 10, 15, 20, 30, 35, 40])
parser.add_argument('--phase', default='noise', type=str, choices=['clean', 'noise'])
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--corruption_level', default=0.1, type=float, help='Noise level', choices=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
parser.add_argument('--noise_ratio', default=0.45, type=float, help='Proportion of noise data', choices=[0.0,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.7,0.8])
parser.add_argument('--num_class', default=10, type=int, help='Class number', choices=[10])
# parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--datapath', default='../dataset', type=str, help='data root path')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--pse_threshold', default=0.95, type=float,help='pseudo label threshold')
parser.add_argument('--Tem', default=1.0, type=float, help='pseudo label temperature')
parser.add_argument('--shapern_T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--Log', default='no', type=str, help='Logging',choices=['yes','no'])
parser.add_argument('--RecordFolder',type=str)
args = parser.parse_args()

# torch.cuda.set_device(args.local_rank)
# torch.is_distributed.init_process_group(backend='nccl')

def test(Feature_L,Class_1,Class_2,test_Target):
    """Test Accuracy"""
    Feature_L.eval()
    Class_1.eval()
    Class_2.eval()

    correct = 0
    total = 0
    Test_confidence = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_Target):
            inputs, targets = inputs.cuda(), targets.cuda()
            feature = Feature_L(inputs)
            c1_out = Class_1(feature)
            c2_out = Class_2(feature)
            outputs = c1_out+c2_out
            conficence,prediction = torch.max(torch.softmax(outputs,-1), 1) #dim=1，每行取最大值
            total+=targets.size(0)
            correct+=prediction.eq(targets).cpu().sum().item()
            Test_confidence.append(conficence)
        acc = 1.*correct / total
        return acc,Test_confidence

def SNetTest(Feature,cnn,Test_data):
    cnn.eval()
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(Test_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            feature = Feature(inputs)
            c_out = cnn(feature)
            outputs = c_out
            _,prediction = torch.max(outputs, 1) #dim=1，每行取最大值
            total+=targets.size(0)
            correct+=prediction.eq(targets).cpu().sum().item()
            
        acc = 1.*correct / total
        return acc

def noise_train(Feature,Cl_1,Cl_2,epoch,opt_F,opt_C,set_s_l,set_s_pse,set_t_l,tar_class_ratio=None,percentile_=None):
    
    Feature.train()
    Cl_1.train()
    Cl_2.eval()

    print('Length of Source Labeled:%d'%len(set_s_l))
    print('Length of Target Labeled:%d'%len(set_t_l))

    
    #sou_l_dataloader,imbalance_r,acc_noise_check = get_balance_sou(set_s_l,percentile_=percentile_)
    sou_l_dataloader = make_data_loader(set_s_l,num_worker=0,batch=args.batch_size_s,drop_last=True)
    tar_l_dataloader = make_data_loader(set_t_l,num_worker=0,batch=args.batch_size_s,drop_last=True)
    sou_pse_dataloader = make_data_loader(set_s_pse,num_worker=0,batch=args.batch_size_s,drop_last=True)

    #num_iter = max(len(sou_l_dataloader), len(tar_l_dataloader))
    num_iter = len(sou_l_dataloader)
    # Class_1, Class_2
    iter_sou_l = iter(sou_l_dataloader)
    iter_tar_l = iter(tar_l_dataloader)
    iter_sou_pse = iter(sou_pse_dataloader)

    for iteration in range(num_iter):
        try:
            batch_sou_pse = iter_sou_pse.next()
        except:
            iter_sou_pse = iter(sou_pse_dataloader)
            batch_sou_pse = iter_sou_pse.next()
        try:
            batch_sou_l = iter_sou_l.next() # all weak augmentation
        except:
            iter_sou_l = iter(sou_l_dataloader)
            batch_sou_l = iter_sou_l.next()
        try:
            batch_tar_l = iter_tar_l.next() # all weak augmentation
        except:
            iter_tar_l = iter(tar_l_dataloader)
            batch_tar_l = iter_tar_l.next()
        
        # load batch
        sou_img,_,_,sou_label,sou_w_x = batch_sou_l
        #tar_img,tar_ground,tar_label,tar_w_x = batch_tar_l
        w_t_img,s_t_img,_,pseudo_labels_t,tar_w_xt = batch_tar_l
        w_s_img,s_s_img,_,pseudo_labels_s,tar_w_xs = batch_sou_pse
        # transfer labels
        batch_size = sou_img.size(0)
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, sou_label.view(-1,1),1)
        sou_w_x = sou_w_x.view(-1,1).type(torch.FloatTensor)
        # make cuda
        sou_img,labels_x,sou_w_x = make_cuda(sou_img),make_cuda(labels_x),make_cuda(sou_w_x)
        s_s_img,s_t_img,pseudo_labels_s,pseudo_labels_t = make_cuda(s_s_img),make_cuda(s_t_img),make_cuda(pseudo_labels_s),make_cuda(pseudo_labels_t)

        opt_F.zero_grad()
        opt_C.zero_grad()

        with torch.no_grad():
            # ---- Mixed data --- 
            # re-define for merged data
            feature = Feature(sou_img)
            out_merge = Cl_1(feature)
            px = torch.softmax(out_merge,-1)
            px = sou_w_x * labels_x + (1-sou_w_x) * px   
            ptx = px**(1/args.shapern_T)
            targets_merge = ptx/ptx.sum(dim=1,keepdim=True)
            targets_merge = targets_merge.detach()
        # forward 
        # loss for sou labelled
        feature = Feature(sou_img)
        predict_sl = Cl_1(feature)
        loss_sl = criterion(predict_sl,targets_merge)
        feature = Feature(s_s_img) # predict for strong unlabelled source
        predict_sou1 = Cl_1(feature)
        predict_ss = predict_sou1
        loss_su = CEloss(predict_ss,pseudo_labels_s)
        # loss for unlablled target data 
        feature = Feature(s_t_img)
        predict_1 = Cl_1(feature)
        predict_tar = predict_1
        loss_tu = CEloss(predict_tar,pseudo_labels_t)
        # Total Loss
        loss = loss_sl + loss_su + loss_tu #+ loss_skl + loss_tkl
        #loss = loss_sl + loss_tu
        loss.backward()
        opt_F.step()
        opt_C.step()

        # sys.stdout.write('\r%s:%.1f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]|Labeled(sou) Loss:%.4f|SourUnloss:%.4f |TarUnloss:%.4f, TotalLoss:%.4f'
        #      %(args.dset,args.corruption_level,args.Noise,epoch,args.Epoch,iteration,num_iter,loss_sl.item(),loss_su.item(),loss_tu.item(),loss.item()))
        # sys.stdout.flush()
        sys.stdout.write('\r%s:%.1f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]|Labeled(sou) Loss:%.4f| TarUnloss:%.4f, TotalLoss:%.4f'
             %(args.dset,args.noise_ratio,args.Noise,epoch,args.Epoch,iteration,num_iter,loss_sl.item(),loss_tu.item(),loss.item()))
        sys.stdout.flush()

    return loss_sl.item(),loss_tu.item(),loss.item()#,imbalance_r,acc_noise_check   

def warm_up(Feature,Classifier,epoch,opt_F,opt_C,warm_sou,len_sour):
    Feature.train()
    Classifier.train()
    #num_iter = (get)
    iter_num =  len_sour / args.batch_size_s #100 # len(warm up dataset)
    for batch_idx,(img,_,label,_,index) in enumerate(warm_sou):
        img=make_cuda(img)
        label=make_cuda(label)
        opt_F.zero_grad()
        opt_C.zero_grad()
        feature = Feature(img)
        out = Classifier(feature)
        loss = CEloss(out,label)
        if args.Noise=='asym':
            pass
        else:
            loss = loss
        loss.backward()
        opt_F.step()
        opt_C.step()

        sys.stdout.write('\r %s : %.2f - Noise: %s | Epoch:[%3d/%3d] Iter:[%3d/%3d] | Loss:%.4f'
            %(args.dset,args.noise_ratio,args.Noise,epoch,args.warmup,batch_idx+1,iter_num,loss.item()))
        sys.stdout.flush()
    # return loss
    return loss.item()

def eval_train(Feature,Classifier,eval_data,sou_length):
    print("The preparation before eval_train is OK, now we are eval the samples up!\n")
    Feature.eval()
    Classifier.eval()
    total_num = sou_length
    losses = torch.zeros(total_num)
    with torch.no_grad():
        for batch_idx, (inputs, _,labels,_,index) in enumerate(eval_data):
            inputs = make_cuda(inputs)
            labels = make_cuda(labels)
            f_out = Feature(inputs)
            output = Classifier(f_out)
             # loss of every sample
            loss = CE(output,labels)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    #all_loss.append(losses)
    if args.corruption_level==0.9:
        pass
    else:
        input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=30,tol=1e-2,reg_covar=5e-4) # max_iter = 20 noise:0.2 max_iter = 20 noise:0.45
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    print("Eval have complicated!\n")
    return prob


def specific_train(epoch,Feature,CNN_TarNet,optimizer_F,opt_cnn,set_t_l,set_t_un):
    # initialization
    Feature.train()
    #Feature.eval()
    CNN_TarNet.train()

    l_loader = make_data_loader(set_t_l,num_worker=0,batch=args.batch_size_s,drop_last=True)
    #un_loader = make_data_loader(set_t_un,batch=args.batch_size_s,drop_last=True)

    print('length of target labelled data',len(l_loader))
    #print('length of target unlabelled data',len(un_loader))
    num_iter = len(l_loader)
    #num_iter = 100
    #un_iter = iter(un_loader)
    l_iter = iter(l_loader)
    #for iteration in range(num_iter):
    for iteration in range(num_iter):
        try:
            l_batch = l_iter.next()
        except:
            l_iter = iter(l_loader)
            l_batch = l_iter.next()
        # try:
        #     un_batch = un_iter.next()
        # except:
        #     un_iter = iter(un_loader)
        #     un_batch = un_iter.next()

        # batch
        #img, _,label, w_x = l_batch 
        img,_,_,label,w_x = l_batch 
        #w_t_img, s_t_img = un_batch
        #batch_size = img.size(0)
        #labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, label.view(-1,1),1)
        #w_x = w_x.view(-1,1).type(torch.FloatTensor)
        # make cuda
        label = label.cuda()
        #labels_x = labels_x.cuda()
        img, w_x = img.cuda(), w_x.cuda()
        #w_t_img, s_t_img = w_t_img.cuda(), s_t_img.cuda()
        #for unlabelled Reg
        # with torch.no_grad():
        #     # --- unlabelled target data ---
        #     feature = Feature(w_t_img)
        #     out_un = CNN_TarNet(feature) # ！ Regu
        #     pu_t = torch.softmax(out_un.detach()/args.Tem,-1)
        #     max_probs,targets_un = torch.max(pu_t,-1)
        #     mask = max_probs.ge(args.pse_threshold).float()
        #     # loss for labelled tar
        
        feature = Feature(img)
        predict_l = CNN_TarNet(feature)
        #loss_l = criterion(predict_l,targets_l)
        loss_l = CEloss(predict_l,label)
        #loss for unlabelled tar
        # feature = Feature(s_t_img)
        # predict_un = CNN_TarNet(feature)
        # loss_un = (F.cross_entropy(predict_un,targets_un,reduction='none')*mask).mean()
        
        loss = loss_l #+ loss_un
        optimizer_F.zero_grad()
        opt_cnn.zero_grad()

        loss.backward()
        
        optimizer_F.step()
        opt_cnn.step()

        sys.stdout.write('\r%s|%s:%.2f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]| LabeledLoss:%.4f | TotalLoss:%.4f'
                %("Target Specific",args.dset,args.noise_ratio,args.Noise,epoch,args.Epoch,iteration+1,num_iter,loss_l.item(),loss.item()))
        sys.stdout.flush()

    return loss.item()



print("===========  Training Start(SYDN-->MNIST) Noise(ratio:{},type:{})  ===========\n".format(args.noise_ratio,args.Noise))

enable_cudnn_benchmark()
seed = random_seed()
seed.init_random_seed
print(seed.seed)

if args.phase == 'clean': 
    # 1.clean training phase
    #clean()
    pass
else:
    # # 当前目录的绝对路径
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # 上一级目录的绝对路径
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    source_name,target_name = get_name(args)
    # Log
    if args.Log=='yes':
        log_name = 'SYDN2MNIST_Noise_(New_paper_data){}_{}'.format(args.Noise,args.noise_ratio)
        parse = record_config(dir_path,log_name,dset='Syn2M')
        parse.make_dir()
        writer = SummaryWriter(parse.log_root)
    # SynthDigits -- Source 
    loader_source = loader_source.syndigits_dataloader(root=root,
                                                        ratio=args.noise_ratio,
                                                        level=args.corruption_level,
                                                        type=args.Noise,
                                                        batch_size=args.batch_size_s,
                                                        img_size=args.image_size,
                                                        noise_file='%s/%.2f_%s.json'%(root+'/SynDigits_NoiseLabels',args.noise_ratio,args.Noise)
    )
    target_noise_ratio = 0.0
    # MNIST -- Target
    loader_target = loader_target.mnist_dataloader(root=root,ratio=target_noise_ratio,level=0.0,type=args.Noise,batch_size=args.batch_size_t,img_size=args.image_size,
                                                    noise_file='%s/%.1f_%s.json'%(root+'/MNIST_NoiseLabels',target_noise_ratio,args.Noise) # ！
    )
    print("Loader Completed!\n")
    # 定义网络
    Feature,Class_1,Class_2,cnn_tar = creat_model(model='mnist_syndigits',args=args)
    # optimizer
    optimizer_F=optim.SGD(Feature.module.feature.parameters(),lr=args.lr*0.1,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_C_1=optim.SGD(Class_1.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_C_2=optim.SGD(Class_2.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_cnn=optim.SGD(cnn_tar.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    # loss
    criterion = Labeled_Loss()
    CEloss = nn.CrossEntropyLoss().cuda()
    CE = nn.CrossEntropyLoss(reduction='none')
    ################# 
    # Define Dataset
    #################
    # source 
    Source_Dataset = loader_source.run('ALL_Data') # return weak_img, strong_img, targets, index
    print("length of source:%d"%len(Source_Dataset))
    # target 
    Target_Dataset = loader_target.run('ALL_Data') # return weak_img, strong_img, targets, index
    print("length of target:%d"%len(Target_Dataset))
    # test data
    length_tar,test_target = loader_target.run('test')
    length_sou,test_source = loader_source.run('test')
    warmup_source = make_data_loader(Source_Dataset,batch=args.batch_size_s,shuffle=True,drop_last=True)
    
    #---- test ---- 
    '''pseudo-labels Recall, Precision, Imbalance Ratio(ρ=Nummax/Nummin)'''
    NoiseCleanRatio_1 = [] # noise learning 
    NoiseCleanRatio_2 = []
    Recall_source1 = []
    Precision_source1 = []
    Recall_source2 = []
    Precision_source2 = []
    Recall_target = []
    Precision_target = []
    Acc_t = [] # pseudo-labbels acc
    Acc_s1 = []
    Acc_s2 = []
    ImbalanceRatio_t = [] # equals Num(max class) / Num(min class)
    ImbalanceRatio_s1 = []
    ImbalanceRatio_s2 = []
    ImbalanceRatio_s1_l = []
    ImbalanceRatio_s2_l = []
    
    for epoch in range(args.Epoch):
        print(" --------------------------------------------- This is %d epoch  --------------------------------------------- \n"%epoch)
        all_loss = [[], []]
        if epoch < args.warmup:
            print(" ===  Warm up  ===  \n")
            print("Warmming up the <<first>> Classifier...\n")
            W1_loss = warm_up(Feature,Class_1,epoch,\
                optimizer_F,optimizer_C_1,warmup_source,len(Source_Dataset))
            print("\n")
            print("Warmming up the <<second>> Classifier...\n")
            W2_loss = warm_up(Feature,Class_2,epoch,\
                optimizer_F,optimizer_C_2,warmup_source,len(Source_Dataset))
        else:
            if epoch > 40:
                lr = 0.001
                for param_group in optimizer_F.param_groups:
                    param_group['lr'] = lr*0.1
                for param_group in optimizer_C_1.param_groups:
                    param_group['lr'] = lr
                for param_group in optimizer_C_2.param_groups:
                    param_group['lr'] = lr
                for param_group in optimizer_cnn.param_groups:
                    param_group['lr'] = lr
                
            if epoch-args.warmup == 0:
                percentile_ = 0.1
            else:
                percentile_ = 0.1 + (epoch-args.warmup) * 0.05
                percentile_ = min(0.8, percentile_)

            
            eval_train_source = make_data_loader(Source_Dataset,batch=args.batch_size_s,shuffle=False)
            prob1 = eval_train(Feature,Class_1,eval_train_source,len(Source_Dataset)) # net1 GMM 
            prob2 = eval_train(Feature,Class_2,eval_train_source,len(Source_Dataset)) # net2 GMM
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)
            
            
            # get dataset of target
            tar_l_idx, pseudo_labels,l_tar_weight, tar_un_idx = generate_labels(args,Feature,Class_1,Class_2,Target_Dataset,IF_office=True,percentile_=percentile_)
            tar_dataset_l, tar_dataset_un = get_dummy_tar(Target_Dataset,tar_l_idx,tar_un_idx, pseudo_labels,l_tar_weight,batch_size=args.batch_size_t)
            if epoch >= args.warmup:
                imbalance_r = get_imbalance_ratio(args,tar_dataset_l)
                recall_t,precision_t = cal_Recall_Precision(args,epoch,tar_dataset_l)
                ImbalanceRatio_t.append(imbalance_r)
                Recall_target.append(recall_t)
                Precision_target.append(precision_t)
            acc_t = cal_pse_acc(args,tar_dataset_l)
            print('--------epoch:%d,Acc Tar:%.4f'%(epoch,acc_t))
            print(imbalance_r)
            Acc_t.append(acc_t)
            # ---------------  process for Source -------------- 
            # classifier 1 -------------------------------------
            sou_l_index, sou_u_index, prob = get_source_index(pred2,prob2)
            sou_pse_idx1,pseudo_s1_labels,pse_weight1,_ = generate_labels(args,Feature,Class_1,Class_2,Source_Dataset,domain='source',IF_office=True,sou_u_index=sou_u_index,percentile_=percentile_) # add pseudo-labels for source
            sou_dataset_l, sou_dataset_pse = get_dummy_sou(Source_Dataset,sou_l_index,sou_pse_idx1,prob,pse_weight1,batch_size=args.batch_size_s,pseudo_labels=pseudo_s1_labels)
            if epoch >= args.warmup:
                # test
                imbalance_r1 = get_imbalance_ratio(args,sou_dataset_pse)
                imbalance_l_r1 = get_imbalance_ratio(args,sou_dataset_l)
                recall_s_1,precision_s_1 = cal_Recall_Precision(args,epoch,sou_dataset_pse)
                ImbalanceRatio_s1.append(imbalance_r1)
                ImbalanceRatio_s1_l.append(imbalance_l_r1)
                Recall_source1.append(recall_s_1)
                Precision_source1.append(precision_s_1)
            acc_s_1 = cal_pse_acc(args,sou_dataset_pse)
            print('--------epoch:%d,Acc sou1:%.4f'%(epoch,acc_s_1))
            Acc_s1.append(acc_s_1)
            detect_acc_1 = detect_acc(epoch,sou_dataset_l)
            NoiseCleanRatio_1.append(detect_acc_1)
            print('Noise Check Acc:%.4f'%detect_acc_1)
            print("=== Traing Phase <1> net===")
            loss_sou_1,loss_t_pse1,loss_1= noise_train(Feature,Class_1,Class_2,epoch,optimizer_F,optimizer_C_1,sou_dataset_l,sou_dataset_pse,tar_dataset_l,percentile_=percentile_)
            # ImbalanceRatio_s1_l.append(imbalance_l_r1)
            # NoiseCleanRatio_1.append(detect_acc_1)
            print('\n')
            # classifier 2-------------------------------------
            sou_l_index, sou_u_index, prob = get_source_index(pred1,prob1)
            sou_pse_idx2,pseudo_s2_labels,pse_weight2,_ = generate_labels(args,Feature,Class_1,Class_2,Source_Dataset,domain='source',IF_office=True,sou_u_index=sou_u_index,percentile_=percentile_) # add pseudo-labels for source
            sou_dataset_l, sou_dataset_pse = get_dummy_sou(Source_Dataset,sou_l_index,sou_pse_idx2,prob,pse_weight2,batch_size=args.batch_size_s,pseudo_labels=pseudo_s2_labels)
            if epoch >= args.warmup:
                # test
                imbalance_r2 = get_imbalance_ratio(args,sou_dataset_pse)
                imbalance_l_r2 = get_imbalance_ratio(args,sou_dataset_l)
                recall_s_2,precision_s_2 = cal_Recall_Precision(args,epoch,sou_dataset_pse)
                ImbalanceRatio_s2.append(imbalance_r2)
                ImbalanceRatio_s2_l.append(imbalance_l_r2)
                Recall_source2.append(recall_s_2)
                Precision_source2.append(precision_s_2)
            acc_s_2 = cal_pse_acc(args,sou_dataset_pse)
            print('\n')
            print('--------epoch:%d,Acc sou2:%.4f'%(epoch,acc_s_2))
            Acc_s2.append(acc_s_2)
            detect_acc_2 = detect_acc(epoch,sou_dataset_l)    
            NoiseCleanRatio_2.append(detect_acc_2)
            print('Noise Check Acc:%.4f'%detect_acc_2)
            print('\n')
            print("=== Traing Phase <2> net===")
            loss_sou_2,loss_t_pse2,loss_2 = noise_train(Feature,Class_2,Class_1,epoch,optimizer_F,optimizer_C_2,sou_dataset_l,sou_dataset_pse,tar_dataset_l,percentile_=percentile_)
            loss_l = specific_train(epoch,Feature,cnn_tar,optimizer_F,optimizer_cnn,sou_dataset_l if (epoch - args.warmup)== 0 else tar_dataset_l, tar_dataset_un)
        print('\n')
        """Test Acc"""
        #if epoch >= args.warmup:
        AccTar,confidence = test(Feature,Class_1,Class_2,test_target)
        AccSou,confidence = test(Feature,Class_1,Class_2,test_source)
        AccTarget = SNetTest(Feature,cnn_tar,test_target)
        print("== Target:%s| Epoch:%d | LNet Accuracy:%.4f ==\n" % (target_name,epoch, AccTar))
        print("== Source:%s| Epoch:%d | LNet Accuracy:%.4f ==\n" % (source_name,epoch, AccSou))
        print("== Target:%s| Epoch:%d | SNet Accuracy:%.4f ==\n" % (target_name,epoch, AccTarget))

        if args.Log=='yes':
            writer.add_scalar('Test/Target Acc(Lnet)', AccTar, epoch)
            writer.add_scalar('Test/Target Acc', AccTarget, epoch)
            writer.add_scalar('Test/Source Acc(LNet)', AccSou, epoch)
            if epoch < args.warmup:
                writer.add_scalar('Train/C1_warmup_TotalLoss', W1_loss,epoch) # C1 Loss ,,,
                writer.add_scalar('Train/C2_warmup_TotalLoss',W2_loss,epoch)
            else:
                writer.add_scalar('Train/Specific_LabelledLoss',loss_l,epoch-args.warmup)
                # writer.add_scalar('Train/Specific_TotalLoss',t_loss,epoch-args.warmup)
                writer.add_scalar('Train/C1_train_Labeled(sou)Loss',loss_sou_1,epoch-args.warmup)
                writer.add_scalar('Train/C1_train_pseLabeled(tar)Loss',loss_t_pse1,epoch-args.warmup)
                # writer.add_scalar('Train/C1_train_pseLabeled(sou)Loss',loss_s_pse1,epoch-args.warmup)
                # writer.add_scalar('Train/C2_train_pseLabeled(tar)Loss',loss_s_pse2,epoch-args.warmup)
                writer.add_scalar('Train/C1_train_Total',loss_1,epoch-args.warmup)
                writer.add_scalar('Train/C2_train_Labeled(sou)Loss',loss_sou_2,epoch-args.warmup)
                writer.add_scalar('Train/C2_train_pseLabeled(tar)Loss',loss_t_pse2,epoch-args.warmup)
                writer.add_scalar('Train/C2_train_Total',loss_2,epoch-args.warmup)
        
        json.dump(NoiseCleanRatio_1, open(dir_path+'/Record/{}/Noisecheck1.json'.format(args.RecordFolder), "w"))
        json.dump(NoiseCleanRatio_2, open(dir_path+'/Record/{}/Noisecheck2.json'.format(args.RecordFolder), "w"))
        json.dump(Recall_source1, open(dir_path+'/Record/{}/recall_s1.json'.format(args.RecordFolder), "w"))
        json.dump(Precision_source1, open(dir_path+'/Record/{}/pre_s1.json'.format(args.RecordFolder), "w"))
        json.dump(Recall_source2, open(dir_path+'/Record/{}/recall_s2.json'.format(args.RecordFolder), "w"))
        json.dump(Precision_source2, open(dir_path+'/Record/{}/pre_s2.json'.format(args.RecordFolder), "w"))
        json.dump(Recall_target, open(dir_path+'/Record/{}/recall_t.json'.format(args.RecordFolder), "w"))
        json.dump(Precision_target, open(dir_path+'/Record/{}/pre_t.json'.format(args.RecordFolder), "w"))
        json.dump(Acc_t, open(dir_path+'/Record/{}/Acc_t.json'.format(args.RecordFolder), "w"))
        json.dump(Acc_s1, open(dir_path+'/Record/{}/Acc_s1.json'.format(args.RecordFolder), "w"))
        json.dump(Acc_s2, open(dir_path+'/Record/{}/Acc_s2.json'.format(args.RecordFolder), "w"))
        json.dump(ImbalanceRatio_t, open(dir_path+'/Record/{}/ImbalanceRatio_t.json'.format(args.RecordFolder), "w"))
        json.dump(ImbalanceRatio_s1, open(dir_path+'/Record/{}/ImbalanceRatio_s1.json'.format(args.RecordFolder), "w"))
        json.dump(ImbalanceRatio_s2, open(dir_path+'/Record/{}/ImbalanceRatio_s2.json'.format(args.RecordFolder), "w"))
        json.dump(ImbalanceRatio_s1_l, open(dir_path+'/Record/{}/ImbalanceRatio_s1_l.json'.format(args.RecordFolder), "w"))
        json.dump(ImbalanceRatio_s2_l, open(dir_path+'/Record/{}/ImbalanceRatio_s2_l.json'.format(args.RecordFolder), "w"))
        print("\n")
        print("END! Seed:%d"%seed.seed)


