'''
Amazon <-> Webcam | Amazon <-> Dslr | Dslr <-> Webcam
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.optim as optim
# network
import argparse
from Config.utils import creat_model, random_seed, enable_cudnn_benchmark,make_cuda, record_config, \
    creat_model, NegEntropy,get_name,Labeled_Loss,make_data_loader,get_dummy_tar,get_dummy_sou,generate_labels, \
        make_data_loader, get_source_index,cal_Recall_Precision,detect_acc,get_imbalance_ratio,cal_pse_acc,get_balance_sou
import torch

from sklearn.mixture import GaussianMixture
from dataloader import office_noise_loader_new as office_loader
import torch.nn as nn
from tensorboardX import SummaryWriter
import json
import numpy as np

parser = argparse.ArgumentParser(description=' -- NoiseUDA office -- ')
parser.add_argument('--lr', default=0.01, type=float, choices=[1e-3, 1e-2, 1e-1, 0.005, 0.02, 0.002])
parser.add_argument('--batch_size_s', default=32, type=int, choices=[2,4,8,12,16,32,64,128])
parser.add_argument('--batch_size_t', default=32, type=int, choices=[2,4,8,12,16,32,64,128])
parser.add_argument('--image_size', default=256, type=int, choices=[28, 32, 64, 227, 256])
parser.add_argument('--Epoch', default=60, type=int)
parser.add_argument('--Noise', default='sym', type=str, choices=['asym','sym'])
parser.add_argument('--dset', default='office_d_w', type=str)
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--phase', default='clean', type=str, choices=['clean', 'noise'])
parser.add_argument('--p_threshold', default=0.7, type=float, help='clean probability threshold')
parser.add_argument('--corruption_level', default=0.1, type=float, help='Noise level', choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
parser.add_argument('--noise_ratio', default=0.4, type=float, help='Proportion of noise data')
parser.add_argument('--num_class', default=31, type=int, help='Class number', choices=[31])
parser.add_argument('--Tem', default=1.0, type=float, help='pseudo label temperature')
parser.add_argument('--shapern_T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--datapath', default='../dataset', type=str, help='data root path')
parser.add_argument('--lambda_u', default=0.9, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--pse_threshold', default=0.9, type=float,help='pseudo label threshold')
parser.add_argument('--Log',default='no',type=str,help='if record test result this time',choices=['yes','no'])
parser.add_argument('--RecordFolder',type=str)
args = parser.parse_args()


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

def SNetTest(Feature,cnn,Test_data,best_acc):
    Feature.eval()
    cnn.eval()
    correct = 0
    total = 0
    first_Test = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(Test_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            feature = Feature(inputs)
            if first_Test:
                Features = feature
                first_Test = False
            else:
                Features = torch.cat((Features, feature), 0)
            c_out = cnn(feature)
            outputs = c_out
            _,prediction = torch.max(outputs, 1) #dim=1，每行取最大值
            total+=targets.size(0)
            correct+=prediction.eq(targets).cpu().sum().item()
            
        acc = 1.*correct / total
        
        return acc, Features

def noise_train(Feature,Cl_1,Cl_2,epoch,opt_F,opt_C,set_s_l,set_s_pse,set_t_l,tar_class_ratio=None,percentile_=None):
    
    Feature.train()
    Cl_1.train()
    Cl_2.eval()

    print('Length of Source Labeled:%d'%len(set_s_l))
    print('Length of Target Labeled:%d'%len(set_t_l))

    
    #sou_l_dataloader,imbalance_r,acc_noise_check = get_balance_sou(set_s_l,percentile_=percentile_)
    sou_l_dataloader = make_data_loader(set_s_l,num_worker=8,batch=args.batch_size_s,drop_last=True)
    tar_l_dataloader = make_data_loader(set_t_l,num_worker=8,batch=args.batch_size_s,drop_last=True)
    sou_pse_dataloader = make_data_loader(set_s_pse,num_worker=8,batch=args.batch_size_s,drop_last=True)

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
            # px = torch.softmax(px.detach(),-1)
            # _,hard_label = torch.max(px,-1)
        # forward 
        # loss for sou labelled
        feature = Feature(sou_img)
        predict_sl = Cl_1(feature)
        loss_sl = criterion(predict_sl,targets_merge)
        # FixMatch for pseudo-labels
        # loss for unlabelled source data
        # feature = Feature(w_s_img)
        # predict_1 = Cl_1(feature)
        # predict_2 = Cl_2(feature)
        # predict_sou = predict_1 + predict_2
        # loss_su = CEloss(predict_sou,pseudo_labels_s) # CELoss
        ### Ablation Study 
        # feature = Feature(s_s_img) # predict for strong unlabelled source
        # predict_sou1 = Cl_1(feature)
        # predict_ss = predict_sou1
        # loss_su = CEloss(predict_ss,pseudo_labels_s)
        feature = Feature(s_s_img) # predict for strong unlabelled source
        predict_sou1 = Cl_1(feature)
        predict_ss = predict_sou1
        loss_su = CEloss(predict_ss,pseudo_labels_s)
        # loss for unlablled target data 
        feature = Feature(s_t_img)
        predict_1 = Cl_1(feature)
        predict_tar = predict_1
        loss_tu = CEloss(predict_tar,pseudo_labels_t)
        # feature = Feature(s_t_img)
        # predict_tar1 = Cl_1(feature)
        # predict_tar2 = Cl_2(feature)
        # predict_ts = (predict_tar1 + predict_tar2) #/ 2.
        # loss_tkl = KLloss(predict_ts,predict_tar)
        
        # regloss
        # prior = torch.ones(args.num_class)/args.num_class
        # prior = prior.cuda()        
        # pred_mean = torch.softmax(predict_sl, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))
        # Total Loss
        #loss = loss_sl + loss_su + loss_tu #+ penalty #+ loss_skl + loss_tkl 
        loss = loss_sl + loss_su + loss_tu
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
    
    Feature.train() # output: feature, classout
    Classifier.train()
    iter_num =  len_sour / args.batch_size_s #100 # len(warm up dataset)
    for batch_idx,(img,_,label,_,_) in enumerate(warm_sou):
        img=make_cuda(img)
        label=make_cuda(label)
        opt_F.zero_grad()
        opt_C.zero_grad()
        feature = Feature(img)
        out = Classifier(feature)
        loss = CEloss(out,label)
        # total loss       
        if args.Noise=='asym':
            #penalty = conf_penalty(src_class_output)
            # loss = src_loss_class #+ src_loss_domain + tar_loss_domain + penalty
            pass
        else:
            #penalty = conf_penalty(out) # 在类别不平衡
            Loss = loss #+ penalty
        Loss.backward()
        opt_F.step()
        opt_C.step()

        sys.stdout.write('\r %s : %.1f - Noise: %s | Epoch:[%3d/%3d] Iter:[%3d/%3d]|Loss:%.4f'
            %(args.dset,args.noise_ratio,args.Noise,epoch,args.warmup,batch_idx+1,iter_num,Loss.item()))
        sys.stdout.flush()

    return loss.item()


def eval_train(Feature,Classifier,eval_data,sou_length):

    Feature.eval()
    Classifier.eval()
    total_num = sou_length
    losses = torch.zeros(total_num)
    with torch.no_grad():
        for batch_idx, (inputs,_,labels,_,index) in enumerate(eval_data):
            inputs = make_cuda(inputs)
            labels = make_cuda(labels)
            f_out = Feature(inputs)
            output = Classifier(f_out)
             # loss of every sample
            loss = CE(output,labels)
            '''
            修改为每个class单独进行建模，原来的阈值进行判断，
            改为现在的按照排序取一定的比例决定为干净还是脏标记
            '''
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses-losses.min())/(losses.max()-losses.min())
    if args.noise_ratio==0.9:
        # history = torch.stack(all_loss)
        # input_loss = history[-5: ].mean(0)
        # input_loss = input_loss.reshape(-1,1)
        pass
    else:
        input_loss = losses.reshape(-1,1)
    
    class_ = args.num_class
    gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()] #属于min的probability
    print("Eval have complicated!\n")
    return prob 

def specific_train(epoch,Feature,CNN_TarNet,optimizer_F,opt_cnn,set_t_l,set_t_un):
    # initialization
    Feature.train()
    #Feature.eval()
    CNN_TarNet.train()

    l_loader = make_data_loader(set_t_l,num_worker=8,batch=args.batch_size_s,drop_last=True)
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

        sys.stdout.write('\r%s|%s:%.1f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]| LabeledLoss:%.4f | TotalLoss:%.4f'
                %("Target Specific",args.dset,args.noise_ratio,args.Noise,epoch,args.Epoch,iteration+1,num_iter,loss_l.item(),loss.item()))
        sys.stdout.flush()

    return loss.item() 



print("===========  Training Start(office_{}) Noise ratio:{}===========\n".format(args.dset,args.noise_ratio))
'''2021-11-9'''
enable_cudnn_benchmark()
seed = random_seed(1234) #7312
seed.init_random_seed()

if args.phase == 'clean':
    pass
else:
    '''Noise training with noise processing'''
    #torch.multiprocessing.set_start_method('spawn')
    print(" === Starting training noise is %f === \n"%args.noise_ratio)
    # root 
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset/office31'
    # label = []
    # amazon_label = json.load(open(root+'/Noise_labels/amazon/0.0_sym.json', "r"))
    # label.append(amazon_label)
    # dslr_label = json.load(open(root+'/Noise_labels/dslr/0.0_sym.json', "r"))
    # label.append(dslr_label)
    # webcam_label = json.load(open(root+'/Noise_labels/webcam/0.0_sym.json', "r"))
    # label.append(webcam_label)
    # drawing(root,label)
    # exit()
    # get sourcename and target name
    source_name,target_name = get_name(args) # amazon,webcam
    print('source:',source_name)
    print('target',target_name)
    # Log
    if args.Log=='yes':
        print(args.Log)
        log_name = 'Office_{}2{}_(New_paperdata)Noise_{}'.format(source_name,target_name,args.noise_ratio)
        parse = record_config(dir_path,log_name,args.dset)
        parse.make_dir()
        writer = SummaryWriter(parse.log_root)
    #Source Office
    Source_loader = office_loader.office_31(root=root,
                                            type=args.Noise,
                                            domain='source',
                                            dataset_name=source_name,
                                            batch_size=args.batch_size_s, #2
                                            img_size=args.image_size
    )
    target_noise_ratio = 0.0
    # Test Data 
    Target_loader = office_loader.office_31(root=root,
                                            type=args.Noise,
                                            domain='target',
                                            dataset_name=target_name,
                                            batch_size=args.batch_size_t, #8
                                            img_size=args.image_size          
    )
    # 拿到 Target的class ratio
    print("Loader Completed!\n")
    #定义office网络
    Feature_L,Class_1,Class_2,CNN_TarNet = creat_model(model='office31',args=args)
    # optimizer
    optimizer_F = optim.SGD(Feature_L.module.feature_extractor.parameters(),lr=args.lr*0.1,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_C_1 = optim.SGD(Class_1.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_C_2 = optim.SGD(Class_2.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer_CNN_T = optim.SGD(CNN_TarNet.module.classifier.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
    # loss 
    criterion = Labeled_Loss()
    CEloss = nn.CrossEntropyLoss().cuda()
    CE = nn.CrossEntropyLoss(reduction='none')
    #if args.Noise == 'asym': 
    conf_penalty = NegEntropy()
    ################# 
    # Define Dataset
    #################
    # source 
    Source_Dataset = Source_loader.run('ALL_Data',data_path=root+'/%s_uniform_noisy_%.1f.txt'%(source_name,args.noise_ratio)) # return weak_img, strong_img, targets, index
    print("length of source:%d"%len(Source_Dataset))
    # target 
    Target_Dataset = Target_loader.run('ALL_Data',data_path=root+'/%s.txt'%(target_name)) # return weak_img, strong_img, targets, index
    print("length of target:%d"%len(Target_Dataset))
    # test data
    length_tar,test_Target = Target_loader.run('test',data_path=root+'/%s.txt'%(target_name))
    length_tar,test_Source = Source_loader.run('test',data_path=root+'/%s.txt'%(source_name))
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
    Acc_source = []
    Acc_target_S = []
    Acc_target_T = [] # 教师
    best_acc = 0.
    for epoch in range(args.Epoch):
        print(" ------- This is %d epoch ------- \n"%epoch)
       #all_loss = [[],[]]
        if epoch < args.warmup: 
            print("Warmming up 1")
            W1_total = warm_up(Feature_L,Class_1,epoch,\
                optimizer_F,optimizer_C_1,warmup_source,len(Source_Dataset))
            print('\n')
            print("Warmming up 2")
            W2_total = warm_up(Feature_L,Class_2,epoch,\
                optimizer_F,optimizer_C_2,warmup_source,len(Source_Dataset))
        else:
            # if epoch > 40:
            #     lr = 0.001
            #     for param_group in optimizer_F.param_groups:
            #         param_group['lr'] = lr*0.1
            #     for param_group in optimizer_C_1.param_groups:
            #         param_group['lr'] = lr
            #     for param_group in optimizer_C_2.param_groups:
            #         param_group['lr'] = lr
            #     for param_group in optimizer_CNN_T.param_groups:
            #         param_group['lr'] = lr

            # if epoch-args.warmup == 0:
            #     percentile_ = 0.4
            # else:
            #     percentile_ = 0.4 + (epoch-args.warmup) * 0.05
            #     percentile_ = min(0.8, percentile_)
            if epoch-args.warmup == 0:
                percentile_ = 0.2
            else:
                percentile_ = 0.2 + (epoch-args.warmup) * 0.05
                percentile_ = min(0.7, percentile_)
                
            eval_train_source = make_data_loader(Source_Dataset,batch=args.batch_size_s,shuffle=False)
            prob1 = eval_train(Feature_L,Class_1,eval_train_source,len(Source_Dataset)) # net1 GMM 
            prob2 = eval_train(Feature_L,Class_2,eval_train_source,len(Source_Dataset)) # net2 GMM
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)
            # ---------------  process for target -------------- 
            # get dataset of target
            # if target_name == 'webcam':
            #     tar_class_ratio = json.load(open(root+'/class_ratio_webcam','r'))
            tar_l_idx, pseudo_labels,l_tar_weight, tar_un_idx = generate_labels(args,Feature_L,Class_1,Class_2,Target_Dataset,IF_office=True,percentile_=percentile_)
            tar_dataset_l, tar_dataset_un = get_dummy_tar(Target_Dataset,tar_l_idx,tar_un_idx, pseudo_labels,l_tar_weight,batch_size=args.batch_size_t)
            # if epoch >= args.warmup:
            #     imbalance_r = get_imbalance_ratio(args,tar_dataset_l)
            #     recall_t,precision_t = cal_Recall_Precision(args,epoch,tar_dataset_l)
            #     ImbalanceRatio_t.append(imbalance_r)
            #     Recall_target.append(recall_t)
            #     Precision_target.append(precision_t)
            acc_t = cal_pse_acc(args,tar_dataset_l)
            print('--------epoch:%d,Acc Tar:%.4f'%(epoch,acc_t))
            #print(imbalance_r)
            Acc_t.append(acc_t)
            # ---------------  process for Source -------------- 
            # classifier 1 -------------------------------------
            sou_l_index, sou_u_index, prob = get_source_index(pred2,prob2)
            sou_pse_idx1,pseudo_s1_labels,pse_weight1,_ = generate_labels(args,Feature_L,Class_1,Class_2,Source_Dataset,domain='source',IF_office=True,sou_u_index=sou_u_index,percentile_=percentile_) # add pseudo-labels for source
            sou_dataset_l, sou_dataset_pse = get_dummy_sou(Source_Dataset,sou_l_index,sou_pse_idx1,prob,pse_weight1,batch_size=args.batch_size_s,pseudo_labels=pseudo_s1_labels)
            # if epoch >= args.warmup:
            #     # test
            #     imbalance_r1 = get_imbalance_ratio(args,sou_dataset_pse)
            #     imbalance_l_r1 = get_imbalance_ratio(args,sou_dataset_l)
            #     recall_s_1,precision_s_1 = cal_Recall_Precision(args,epoch,sou_dataset_pse)
            #     ImbalanceRatio_s1.append(imbalance_r1)
            #     ImbalanceRatio_s1_l.append(imbalance_l_r1)
            #     Recall_source1.append(recall_s_1)
            #     Precision_source1.append(precision_s_1)
            # acc_s_1 = cal_pse_acc(args,sou_dataset_pse)
            # print('--------epoch:%d,Acc sou1:%.4f'%(epoch,acc_s_1))
            # Acc_s1.append(acc_s_1)
            # detect_acc_1 = detect_acc(epoch,sou_dataset_l)
            # NoiseCleanRatio_1.append(detect_acc_1)
            # print('Noise Check Acc:%.4f'%detect_acc_1)
            print("=== Traing Phase <1> net===")
            loss_sou_1,loss_t_pse1,loss_1= noise_train(Feature_L,Class_1,Class_2,epoch,optimizer_F,optimizer_C_1,sou_dataset_l,sou_dataset_pse,tar_dataset_l,percentile_=percentile_)
            # ImbalanceRatio_s1_l.append(imbalance_l_r1)
            # NoiseCleanRatio_1.append(detect_acc_1)
            print('\n')
            # classifier 2-------------------------------------
            sou_l_index, sou_u_index, prob = get_source_index(pred1,prob1)
            sou_pse_idx2,pseudo_s2_labels,pse_weight2,_ = generate_labels(args,Feature_L,Class_1,Class_2,Source_Dataset,domain='source',IF_office=True,sou_u_index=sou_u_index,percentile_=percentile_) # add pseudo-labels for source
            sou_dataset_l, sou_dataset_pse = get_dummy_sou(Source_Dataset,sou_l_index,sou_pse_idx2,prob,pse_weight2,batch_size=args.batch_size_s,pseudo_labels=pseudo_s2_labels)
            # if epoch >= args.warmup:
            #     # test
            #     imbalance_r2 = get_imbalance_ratio(args,sou_dataset_pse)
            #     imbalance_l_r2 = get_imbalance_ratio(args,sou_dataset_l)
            #     recall_s_2,precision_s_2 = cal_Recall_Precision(args,epoch,sou_dataset_pse)
            #     ImbalanceRatio_s2.append(imbalance_r2)
            #     ImbalanceRatio_s2_l.append(imbalance_l_r2)
            #     Recall_source2.append(recall_s_2)
            #     Precision_source2.append(precision_s_2)
            # acc_s_2 = cal_pse_acc(args,sou_dataset_pse)
            # print('\n')
            # print('--------epoch:%d,Acc sou2:%.4f'%(epoch,acc_s_2))
            # Acc_s2.append(acc_s_2)
            # detect_acc_2 = detect_acc(epoch,sou_dataset_l)    
            # NoiseCleanRatio_2.append(detect_acc_2)
            # print('Noise Check Acc:%.4f'%detect_acc_2)
            print('\n')
            print("=== Traing Phase <2> net===")
            loss_sou_2,loss_t_pse2,loss_2 = noise_train(Feature_L,Class_2,Class_1,epoch,optimizer_F,optimizer_C_2,sou_dataset_l,sou_dataset_pse,tar_dataset_l,percentile_=percentile_)
            loss_l = specific_train(epoch,Feature_L,CNN_TarNet,optimizer_F,optimizer_CNN_T,sou_dataset_l if (epoch - args.warmup)== 0 else tar_dataset_l, tar_dataset_un)

        print('\n')
        """Test Acc"""
        AccTar,confidence = test(Feature_L,Class_1,Class_2,test_Target)
        AccSou,confidence = test(Feature_L,Class_1,Class_2,test_Source)
        AccTarget, FeaturesTarget = SNetTest(Feature_L,CNN_TarNet,test_Target,best_acc)
        # if AccTarget > best_acc:
        #     best_acc = AccTarget
        #     np.save(dir_path+'/Record/{}/RSLR_features_target_office31_{}.npy'.format(args.RecordFolder, args.dset), FeaturesTarget.cpu().numpy())
        
        Acc_source.append(AccSou)
        Acc_target_S.append(AccTarget)
        Acc_target_T.append(AccTar) # 教师
        json.dump(Acc_source, open(dir_path+'/Record/{}/Source_Acc_{}_Threshold{}.json'.format(args.RecordFolder,args.noise_ratio,args.pse_threshold), "w"))
        json.dump(Acc_target_S, open(dir_path+'/Record/{}/Student_Acc_{}_Threshold{}.json'.format(args.RecordFolder,args.noise_ratio,args.pse_threshold), "w"))
        json.dump(Acc_target_T, open(dir_path+'/Record/{}/Teacher_Acc_{}_Threshold{}.json'.format(args.RecordFolder,args.noise_ratio,args.pse_threshold), "w"))
        #print("Confidence:",confidence)
        print("== Target:%s| Epoch:%d | LNet Accuracy:%.4f ==\n" % (target_name,epoch, AccTar))
        print("== Source:%s| Epoch:%d | LNet Accuracy:%.4f ==\n" % (source_name,epoch, AccSou))
        print("== Target:%s| Epoch:%d | SNet Accuracy:%.4f ==\n" % (target_name,epoch, AccTarget))
        if args.Log=='yes':
            writer.add_scalar('Test/Target Acc(Lnet)', AccTar, epoch)
            writer.add_scalar('Test/Target Acc', AccTarget, epoch)
            writer.add_scalar('Test/Source Acc(LNet)', AccSou, epoch)
            if epoch < args.warmup:
                writer.add_scalar('Train/C1_warmup_TotalLoss', W1_total,epoch) # C1 Loss ,,,
                writer.add_scalar('Train/C2_warmup_TotalLoss',W2_total,epoch)
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
    #---- test ----  
    '''drawing trainging result'''
    # json.dump(,open(dir_path+'/Record/office31/  .json',"w"))
    # json.dump(NoiseCleanRatio_1, open(dir_path+'/Record/office31/{}/Noisecheck1.json'.format(args.RecordFolder), "w"))
    # json.dump(NoiseCleanRatio_2, open(dir_path+'/Record/office31/{}/Noisecheck2.json'.format(args.RecordFolder), "w"))
    # json.dump(Recall_source1, open(dir_path+'/Record/office31/{}/recall_s1.json'.format(args.RecordFolder), "w"))
    # json.dump(Precision_source1, open(dir_path+'/Record/office31/{}/pre_s1.json'.format(args.RecordFolder), "w"))
    # json.dump(Recall_source2, open(dir_path+'/Record/office31/{}/recall_s2.json'.format(args.RecordFolder), "w"))
    # json.dump(Precision_source2, open(dir_path+'/Record/office31/{}/pre_s2.json'.format(args.RecordFolder), "w"))
    # json.dump(Recall_target, open(dir_path+'/Record/office31/{}/recall_t.json'.format(args.RecordFolder), "w"))
    # json.dump(Precision_target, open(dir_path+'/Record/office31/{}/pre_t.json'.format(args.RecordFolder), "w"))
    json.dump(Acc_t, open(dir_path+'/Record/{}/Acc_t_Threshold{}.json'.format(args.RecordFolder, args.pse_threshold), "w"))
    # json.dump(Acc_s1, open(dir_path+'/Record/office31/{}/Acc_s1.json'.format(args.RecordFolder), "w"))
    # json.dump(Acc_s2, open(dir_path+'/Record/office31/{}/Acc_s2.json'.format(args.RecordFolder), "w"))
    # json.dump(ImbalanceRatio_t, open(dir_path+'/Record/office31/{}/ImbalanceRatio_t.json'.format(args.RecordFolder), "w"))
    # json.dump(ImbalanceRatio_s1, open(dir_path+'/Record/office31/{}/ImbalanceRatio_s1.json'.format(args.RecordFolder), "w"))
    # json.dump(ImbalanceRatio_s2, open(dir_path+'/Record/office31/{}/ImbalanceRatio_s2.json'.format(args.RecordFolder), "w"))
    # json.dump(ImbalanceRatio_s1_l, open(dir_path+'/Record/office31/{}/ImbalanceRatio_s1_l.json'.format(args.RecordFolder), "w"))
    # json.dump(ImbalanceRatio_s2_l, open(dir_path+'/Record/office31/{}/ImbalanceRatio_s2_l.json'.format(args.RecordFolder), "w"))
    print("\n")
    print("END! Seed:%d"%seed.seed)
