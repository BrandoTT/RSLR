"""
Mnist --> Mnist_m
"""
import random
import os
import sys
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchvision.datasets import mnist
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision import transforms
from model import Feature, Classifier, Discriminator, MNISTmodel
import argparse
from Config.utils import creat_model, random_seed, enable_cudnn_benchmark, get_data_loader, make_cuda, record_config, \
    creat_model, SemiLoss, NegEntropy, writter_Log
import Config.config as cf
import torch
from sklearn.mixture import GaussianMixture
from dataloader import mnist_noise_loader as loader_source
from dataloader import mnist_m_noise_loader as loader_target
import torch.nn as nn
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description=' -- NoiseUDA mnist_mnist_m -- ')
#parser.add_argument('--beta', default=1.0, type=float, choices=[0.0, 0.1 ,0.2, 0.3, 0.45, 0.6, 1.0])
parser.add_argument('--lr', default=1e-3, type=float, choices=[1e-3, 1e-2, 1e-1])
parser.add_argument('--batch_size_s', default=32, type=int, choices=[32,64,128])
parser.add_argument('--batch_size_t', default=128, type=int, choices=[32,64,128])
parser.add_argument('--image_size', default=28, type=int, choices=[28, 32])
parser.add_argument('--Epoch', default=100, type=int, choices=[1, 3, 50, 100])
parser.add_argument('--Noise', default='sym', type=str, choices=['asym','sym'])
parser.add_argument('--dset', default='M2m', type=str, choices=['S2M','M2m', 'Syn2SVHN', 'Syn2M'])
parser.add_argument('--warmup', default=20, type=int, choices=[0, 15, 20, 30, 35, 40])
parser.add_argument('--phase', default='clean', type=str, choices=['clean', 'noise'])
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--corruption_level', default=0.1, type=float, help='Noise level', choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
parser.add_argument('--noise_ratio', default=0.0, type=float, help='Proportion of noise data', choices=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
parser.add_argument('--num_class', default=10, type=int, help='Class number', choices=[10])
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--datapath', default='../dataset', type=str, help='data root path')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
args = parser.parse_args()

def clean():
    """干净源域样本训练,baseline上限"""
    print("=== baseline training noise ratio %f==\n"%args.noise_ratio)
    # load data
    train=True # If train phase
    get_dataset=False
    # dataloader_mnist =  get_data_loader(cf.mnist_dataset_name, train, \
    #     get_dataset, args.batch_size_t, args.image_size)
    # dataloader_mnist_m = get_data_loader(cf.mnist_m_dataset_name, train, \
    #     get_dataset, args.batch_size_t, args.image_size)

    # Log
    # log_name = 'MNIST2MNIST_m_Noise_{}'.format(args.noise_ratio)
    # parse = record_config(log_name)
    # parse.make_dir()
    # writer = SummaryWriter(parse.log_root)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    # 上一级目录的绝对路径
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    Source_loader = loader_source.mnist_dataloader(root=root,
                                                   ratio=args.noise_ratio,
                                                   level=args.corruption_level,
                                                   type=args.Noise,
                                                   batch_size=args.batch_size_s,
                                                   img_size=args.image_size,
                                                   noise_file='%s/%.1f_%s.json'%(root+'/MNIST_NoiseLabels',args.noise_ratio, args.Noise)
    )
    Target_loader = loader_target.mnist_m_dataloader(root=root,batch_size=args.batch_size_t,img_size=args.image_size)
    


    # my_net = MNISTmodel()
    # my_net.train()
    # parameter_list = [{"params": my_net.feature.parameters()},
    #                   {"params": my_net.classifier.parameters()},
    #                  {"params": my_net.discriminator.parameters()}]
    #new NNStructure
    F=Feature()
    C=Classifier()
    D=Discriminator()
    F.train()
    C.train()
    D.train()
    parameter_list = [{"params": F.feature.parameters()},
                    {"params": C.classifier.parameters()},
                    {"params": D.discriminator.parameters()}]
    optimizer = optim.SGD(
        parameter_list,
        lr=args.lr
    )

    # log_name = 'M2m(clean)'
    # parse = record_config(log_name)
    # parse.make_dir()
    #writer = SummaryWriter(parse.log_root)

    # classLoss and DomainLoss
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    #my_net=make_cuda(my_net)
    F=make_cuda(F)
    C=make_cuda(C)
    D=make_cuda(D)
    loss_class=make_cuda(loss_class)
    loss_domain=make_cuda(loss_domain)
    best_accu_t = 0.0
    # 2.Noise training
    for epoch in range(args.Epoch):
        
        dataloader_mnist = Source_loader.run('warmup')
        dataloader_mnist_m = Target_loader.run('warmup')

        test_source = Source_loader.run('test')
        test_target = Target_loader.run('test')

        len_dataloader = min(len(dataloader_mnist), len(dataloader_mnist_m)) #461 468
        data_source_iter = iter(dataloader_mnist)
        data_target_iter = iter(dataloader_mnist_m)
        for i in range(len_dataloader): #461
           
            p = float(i + epoch * len_dataloader) / args.Epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # 1. training model using source data
            data_source = data_source_iter.next()
            s_img, s_label,_ = data_source
            batch_size = len(s_label)
            domain_label = torch.zeros(batch_size).long()
            
            s_img=make_cuda(s_img)
            s_label = make_cuda(s_label)
            domain_label = make_cuda(domain_label)
            
            feature = F(input_data=s_img)
            class_output = C(feature=feature)
            domain_output = D(feature=feature, alpha=alpha)
            #class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            # error of Label & Domain
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)
            # 2. training model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target
            batch_size = len(t_img)
            domain_label = torch.ones(batch_size).long()
            t_img=make_cuda(t_img)
            t_label = make_cuda(t_label)
            domain_label = make_cuda(domain_label)
            
            target_feature = F(input_data=t_img)
            target_output = C(feature=target_feature)
            domain_output = D(feature=target_feature, alpha=alpha)
            #target_output, domain_output = my_net(input_data=t_img, alpha=alpha)
            # target domain error
            err_t_domain = loss_domain(domain_output, domain_label)
            err_t_label = loss_class(target_output, t_label)
            # Total error
            err = err_s_label + err_t_domain + err_s_domain

            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            
            sys.stdout.write('\r Epoch:[%3d/%3d] Iter:[%3d/%3d] | TotalLoss:%.4f'
                %(epoch,args.Epoch,i,len_dataloader,err))
            sys.stdout.flush()
        #     torch.save(my_net, '{0}/{1}(clean).pth'.format(cf.model_root, args.dset))
        print("\n")
        # test
        accu_s = TestBaseline(epoch,F,C,D,test_source)#mnist
        print('== Accuracy of the %s %f ==\n' % ('MNIST', accu_s))
        accu_t = TestBaseline(epoch,F,C,D,test_target) #mnist_m
        print('== Accuracy of the %s %f ==\n' % ('mnist_m', accu_t))

        # writer.add_scalar('Train/Source Class Loss', err_s_label.data.cpu(), epoch)
        # writer.add_scalar('Train/Target Class Loss', err_t_label.data.cpu(), epoch)
        # writer.add_scalar('Train/Source Domain Loss', err_s_domain.data.cpu(), epoch)
        # writer.add_scalar('Train/Target Domain Loss', err_s_domain.data.cpu(), epoch)
        # writer.add_scalar('Train/Total Loss',err.data.cpu(),epoch)
        # writer.add_scalar('Test/Source Acc', accu_s, epoch)
        # writer.add_scalar('Test/Target Acc', accu_t, epoch)

        
def TestBaseline(epoch,F,C,D,Test_data):
    """Test Accuracy"""
    F.eval()
    C.eval()
    D.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(Test_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            test_feature = F(inputs)
            c_out = C(test_feature)
            outputs = c_out
            _,prediction = torch.max(outputs, 1) #dim=1，每行取最大值
            total+=targets.size(0)
            correct+=prediction.eq(targets).cpu().sum().item()
        acc = 1.*correct / total
        return acc

def test(epoch,F,C_1,C_2,D,data_loader):
    """Test Accuracy"""
    F.eval()
    C_1.eval()
    C_2.eval()
    D.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            test_feature = F(inputs)
            c1_out = C_1(test_feature)
            c2_out = C_2(test_feature)
            outputs = c1_out + c2_out
            _,prediction = torch.max(outputs, 1) #dim=1，每行取最大值
            total+=targets.size(0)
            correct+=prediction.eq(targets).cpu().sum().item()
        acc = 1.*correct / total
        return acc



def noise_train(Feature,Classifier_1,Classifier_2,Discriminator,epoch,optimizer_F,optimizer_C,optimizer_D,source_labeled,source_unlabeled,target_loader):
    """正式训练"""  
    print("Now We are starting noise training!\n")
    Feature.train()
    Classifier_1.train()
    Classifier_2.eval() # fix one network and train another
    Discriminator.train()
    
    length_source = len(source_labeled) + len(source_unlabeled)
    num_iter = min(len(target_loader), length_source)
    data_target_iter = iter(target_loader)
    data_source_labeled_iter = iter(source_labeled)
    data_source_unlabeled_iter = iter(source_unlabeled)
    """"
    现在的问题就是在评估labeled和unlabeled的时候，实验显示某一个数量很少，甚至
    不如target，这样的话，就会报错，比如，.next()后边没有了，这样就不能用source总长度来
    计算num_iter 
    这样就很难实现
    """
    # alignment training
    for iteration in range(num_iter):
        p = float(iteration + epoch * num_iter) / args.Epoch / num_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # class loss and noise learning
        source_labeled_batch = data_source_labeled_iter.next() 
        source_unlabeled_batch = data_source_unlabeled_iter.next()
        """ source_labeled_batch: inputs1, inputs2, labels, w_x"""
        """ source_unlabeled_batch: inputs1, inputs2"""
        inputs_x1, inputs_x2, labels_x, w_x = source_labeled_batch
        inputs_u1, inputs_u2 = source_unlabeled_batch
        batch_size = inputs_x1.size(0)
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1),1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)
        
        inputs_x1, inputs_x2, labels_x, w_x = inputs_x1.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()
        
        
        # source or  target dis
        # operations for source data
        with torch.no_grad():
            # label co-guessing for unlabeled source data ??? 
            f_out_u1 = Feature(inputs_u1)
            f_out_u2 = Feature(inputs_u2)
            outputs_u11 = Classifier_1(f_out_u1)
            outputs_u12 = Classifier_1(f_out_u2)
            outputs_u21 = Classifier_2(f_out_u1)
            outputs_u22 = Classifier_2(f_out_u2)
            
            pu = (torch.softmax(outputs_u11,dim=1) + torch.softmax(outputs_u12,dim=1) + torch.softmax(outputs_u21,dim=1) +torch.softmax(outputs_u22,dim=1)) / 4
            ptu = pu**(1/args.T)
            targets_source_u = ptu / ptu.sum(dim=1,keepdim=True)
            targets_source_u.detach()
            
            # label co-refinement for labeled source data 
            f_out_x1 = Feature(inputs_x1)
            f_out_x2 = Feature(inputs_x2)
            outputs_x1 = Classifier_1(f_out_x1)
            outputs_x2 = Classifier_2(f_out_x2)

            px = (torch.softmax(outputs_x1,dim=1) + torch.softmax(outputs_x2,dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px**(1/args.T)
            targets_source_x = ptx / ptx.sum(dim=1,keepdim=True)
            targets_source_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha) # λ ~ Beta(α, α)    
        l = max(l, 1-l) # λ' = max(λ, 1-λ)
        
        # W 
        all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0) # 4 * batch_size_s = batch_size_t
        all_targets = torch.cat([targets_source_x, targets_source_x, targets_source_u, targets_source_u],dim=0)
        idx = torch.randperm(all_inputs.size(0)) # return list [2, 1, 0, 3 ... ]
        
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b #mixed_input实际上就是[X',X',U',U']混合后的样本以及对应的标记
        
        size_batch_source = all_inputs.size(0)
        domain_label = torch.zeros(size_batch_source).long() # source label: 0
        domain_label = domain_label.cuda()

        # forward
        f_out_source = Feature(mixed_input)
        c_out_source = Classifier_1(f_out_source)
        d_out_source = Discriminator(f_out_source,alpha)
        
        # c_out_source [:args.batch_size_s*2] 
        logits_x = c_out_source[:args.batch_size_s*2]
        logits_u = c_out_source[args.batch_size_s*2:]
        Lx, Lu, lamb = criterion(logits_x,mixed_target[:args.batch_size_s*2],logits_u,mixed_target[args.batch_size_s*2:],epoch+iteration,args.warmup)
        loss_s_domain = CEloss(d_out_source, domain_label)
        
        # operations for target data 
        target_batch = data_target_iter.next()
        t_img, t_label = target_batch
        size_batch_target = t_img.size(0)
        domain_label = torch.ones(size_batch_target).long()
        
        t_img,t_label = t_img.cuda(),t_label.cuda()
        domain_label = domain_label.cuda()

        f_out_target = Feature(t_img)
        c_out_target = Classifier_1(f_out_target)
        d_out_target = Discriminator(f_out_target,alpha)
        # Target error
        loss_t_domain = CEloss(d_out_target,domain_label)
        loss_t_labels = CEloss(c_out_target, t_label)
        
        # regularization 
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(c_out_source,dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        # Total Loss
        Loss = Lx + lamb * Lu + penalty + loss_s_domain + loss_t_domain
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        optimizer_D.zero_grad()
        
        Loss.backward()
        
        optimizer_F.step()
        optimizer_C.step()
        optimizer_D.step()
        # sys.stdout.write('\r%s:%.1f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]|Labeled Loss:%.4f, Unlabeled Loss:%.4f, Source Domian_loss:%.4f, Target Domain_loss:%.4f, Target Class_loss:%.4f, TotalLoss:%.4f'
        #     %(args.dset,args.corruption_level,args.Noise,epoch,args.Epoch,iteration,num_iter,Lx.item(),Lu.item(), loss_s_domain.data.cpu().numpy(), loss_t_domain.data.cpu().numpy(), loss_t_labels.data.cpu().numpy(),Loss.item()))
        sys.stdout.write('\r%s:%.1f-%s|Epoch:[%3d/%3d]Iter:[%3d/%3d]|Labeled Loss:%.4f,Unlabeled Loss:%.4f,TotalLoss:%.4f'
             %(args.dset,args.corruption_level,args.Noise,epoch,args.Epoch,iteration,num_iter,Lx.item(),Lu.item(),Loss.item()))
        sys.stdout.flush()

    #return loss
    return Lx.item(),Lu.item(),loss_s_domain.item(),loss_t_domain.item(),Loss.item()




def warm_up(Feature,Classifier,Discriminator,epoch,optimizer_F,optimizer_C,optimizer_D,warmup_source, warmup_target):
    """warm up之后，warm up就是普通的DANN训练"""
    print("The preparation before warm up is OK, now we are warmming up!\n")
    Feature.train()
    Classifier.train()
    Discriminator.train()

    iter_num = min(len(warmup_source),len(warmup_target)) # 461
    data_source_iter = iter(warmup_source) 
    data_target_iter = iter(warmup_target)
    
    for iteration in range(iter_num):
        
        p = float(iteration + epoch * iter_num) / args.warmup / iter_num
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # >> Source <<
        data_source = data_source_iter.next()
        s_img, s_label, index = data_source

        batch_size = len(s_label)
        domain_label =  torch.zeros(batch_size).long()
        s_img = make_cuda(s_img)
        s_label = make_cuda(s_label)
        domain_label = make_cuda(domain_label)

        #forward network
        f_out = Feature(input_data=s_img)
        source_class_output = Classifier(feature=f_out)
        domain_output = Discriminator(feature=f_out,alpha=alpha)
        
        # error of source
        loss_s_label = CEloss(source_class_output, s_label)
        loss_s_domain = CEloss(domain_output, domain_label)

        # >> Target <<
        data_target = data_target_iter.next()
        t_img, t_label = data_target
        batch_size = len(t_label)
        domain_label = torch.ones(batch_size).long()
        
        t_img = make_cuda(t_img)
        t_label = make_cuda(t_label)
        domain_label = make_cuda(domain_label)

        #forward network
        f_out = Feature(input_data=t_img)
        target_class_output = Classifier(feature=f_out)
        domain_output = Discriminator(feature=f_out,alpha=alpha)

        # error of target
        loss_t_label = CEloss(target_class_output,t_label)
        loss_t_domain = CEloss(domain_output, domain_label)
        
        # >> Total error <<
        if args.Noise=='asym': # flip is asym; uniform is sym
            penalty = conf_penalty(source_class_output)
            total_err = loss_s_label + loss_s_domain + loss_t_domain + penalty
        else:
            total_err = loss_s_label + loss_s_domain + loss_t_domain
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        optimizer_D.zero_grad()

        total_err.backward()
        # update
        optimizer_F.step()
        optimizer_C.step()
        optimizer_D.step()
        
        
        sys.stdout.write('\r %s : %.1f - Noise: %s | Epoch:[%3d/%3d] Iter:[%3d/%3d] | ClassLoss:%.4f, TotalLoss:%.4f'
            %(args.dset,args.noise_ratio,args.Noise,epoch,args.warmup,iteration,iter_num,loss_s_label.item(), total_err.item()))
        sys.stdout.flush()
    
    # return loss
    return loss_s_label.item(), loss_s_domain.item(), loss_t_domain.item(),total_err.item()

def eval_train(Feature,Classifier,all_loss,eval_data):
    """通过对Loss分布做GMM分析，检测出最有可能是噪声的标记"""
    print("The preparation before eval_train is OK, now we are eval the samples up!\n")
    Feature.eval()
    Classifier.eval()
    total_num = 60000
    losses = torch.zeros(total_num)
    with torch.no_grad():
        #alpha = 0.
        for batch_idx, (inputs, labels, index) in enumerate(eval_data):
            inputs = make_cuda(inputs)
            labels = make_cuda(labels)
            f_out = Feature(inputs)
            output = Classifier(f_out)
             # loss of every sample
            loss = CE(output,labels)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    if args.corruption_level==0.9:
        history = torch.stack(all_loss)
        input_loss = history[-5: ].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    print("Eval have complicated!\n")
    return prob, all_loss


#if __name__ == '__main__':

print("===========  Training Start Noise ratio:{}===========\n".format(args.noise_ratio))

enable_cudnn_benchmark()
seed = random_seed()
seed.init_random_seed

if args.phase == 'clean': 
    # 1.clean training phase
    clean()
else:
    # # 当前目录的绝对路径
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # 上一级目录的绝对路径
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = dir_path+'/dataset'
    # Log
    log_name = 'MNIST2MNIST_m_Noise(Divide)_{}'.format(args.noise_ratio)
    parse = record_config(log_name)
    parse.make_dir()
    writer = SummaryWriter(parse.log_root)

    # 2.Noise training
    loader_source = loader_source.mnist_dataloader(root=root,
                                                   ratio=args.noise_ratio,
                                                   level=args.corruption_level,
                                                   type=args.Noise,
                                                   batch_size=args.batch_size_s,
                                                   img_size=args.image_size,
                                                   noise_file='%s/%.1f_%s.json'%(root+'/MNIST_NoiseLabels',args.noise_ratio, args.Noise)
                                                   )
    loader_target = loader_target.mnist_m_dataloader(root=root,batch_size=args.batch_size_t,img_size=args.image_size)
    print("Loader completed!\n")
    #定义网络
    model='mnist_mnist_m'
    F,C_1,C_2,D = creat_model(model=model)
    
    # optimizer
    optimizer_F=optim.SGD(F.parameters(),lr=args.lr,momentum=0.9)
    optimizer_C_1=optim.SGD(C_1.parameters(),lr=args.lr,momentum=0.9)
    optimizer_C_2=optim.SGD(C_2.parameters(),lr=args.lr,momentum=0.9)
    optimizer_D=optim.SGD(D.parameters(),lr=args.lr,momentum=0.9)
    # loss
    criterion = SemiLoss()
    CEloss = nn.CrossEntropyLoss()
    CE = nn.CrossEntropyLoss(reduction='none')
    if args.Noise == 'asym': 
        conf_penalty = NegEntropy()
    all_loss = [[], []] #源域
    #acc = [[], []] #源域和目标域的acc
    for epoch in range(args.Epoch):
        lr = args.lr
        if epoch >= 80:
            lr /= 10
        for param_group in optimizer_F.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_C_1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_C_2.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = lr
        
        eval_train_source = loader_source.run('eval_train')
        test_source = loader_source.run('test')
        test_target = loader_target.run('test')
        if epoch < args.warmup:
            """warm up阶段，只用噪声源域数据训练分类器"""
            print("This is %d epoch\n"%epoch)
            print(" ===  Warm up  ===  \n")
            warmup_source = loader_source.run('warmup')
            warmup_target = loader_target.run('warmup')
            print("Warmming up the <<first>> Classifier...\n")
            W1_Class,W1_s_d,W1_t_d,W1_total = warm_up(Feature=F,Classifier=C_1,Discriminator=D,epoch=epoch,optimizer_F=optimizer_F,optimizer_C=optimizer_C_1,optimizer_D=optimizer_D,\
                warmup_source=warmup_source, warmup_target=warmup_target)
            print("\n")
            print("Warmming up the <<second>> Classifier...\n")
            W2_Class,W2_s_d,W2_t_d,W2_total = warm_up(Feature=F,Classifier=C_2,Discriminator=D,epoch=epoch,optimizer_F=optimizer_F,optimizer_C=optimizer_C_2,optimizer_D=optimizer_D,\
                warmup_source=warmup_source, warmup_target=warmup_target)
            # if epoch == args.warmup-1:
            #     """最后一轮，给所有的目标域样本加上初始的伪标记"""
            #     print("--- Generating Pseudo labels --- \n")
            #     generate_labels(net)
            #     excerpt, pseudo_labels = \
            #         generate_labels(net, target_dataset, num) #为Target Data 加入pseudo-labels Dataset
            #     target_dataset_labelled = get_dummy(target_dataset,)
        
        else:
            prob1,all_loss[0] = eval_train(Feature=F,Classifier=C_1,all_loss=all_loss[0],eval_data=eval_train_source) # net1 GMM 
            prob2,all_loss[1] = eval_train(Feature=F,Classifier=C_2,all_loss=all_loss[1],eval_data=eval_train_source) # net2 GMM
            
            """正式训练阶段，用噪声源域"""
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)
            print("=== Traing Phase <1> net===")
            """每个epoch为Source划分标记数据和无标记数据"""
            source_labeled_data, source_unlabeled_data \
                = loader_source.run('train',pred2,prob2)
            train_target = loader_target.run('train')
            """noise training 阶段，用到源域和目标域（有标记数据和无标记数据做MixMatch）"""
            Tr1_Lx,Tr1_Lu,Tr1_sd,Tr1_td,Tr1_Total = noise_train(F,C_1,C_2,D,epoch,optimizer_F,optimizer_C_1,optimizer_D,source_labeled_data,source_unlabeled_data,train_target)
            
            print("=== Traing Phase <2> net===")
            source_labeled_data, source_unlabeled_data \
                = loader_source.run('train',pred1,prob1)
            train_target = loader_target.run('train')
            """noise training 阶段，用到源域和目标域（有标记数据和无标记数据做MixMatch）"""
            Tr2_Lx,Tr2_Lu,Tr2_sd,Tr2_td,Tr2_Total = noise_train(F,C_2,C_1,D,epoch,optimizer_F,optimizer_C_2,optimizer_D,source_labeled_data,source_unlabeled_data,train_target)
        print('\n')
        """Test Acc"""
        AccSource = test(epoch,F,C_1,C_2,D,test_source) #每个Epoch之后，进行一轮测试
        print("== Source:%s| Epoch:%d | Accuracy:%.10f ==\n" % ('Mnist',epoch,AccSource))
        AccTarget = test(epoch,F,C_1,C_2,D,test_target)
        print("== Target:%s| Epoch:%d | Accuracy:%.10f ==\n" % ('Mnist_m',epoch,AccTarget))
        # acc[0].append(AccSource)
        # acc[1].append(AccTarget)
        writer.add_scalar('Test/Source Acc', AccSource, epoch) #Acc
        writer.add_scalar('Test/Target Acc', AccTarget, epoch)
        if epoch < args.warmup:
            writer.add_scalar('Train/C1_warmup_ClassLoss', W1_Class,epoch) # C1 Loss ,,,
            writer.add_scalar('Train/C1_warmup_DomainSLoss',W1_s_d,epoch)
            writer.add_scalar('Train/C1_warmup_DomainTLoss',W1_t_d,epoch)
            writer.add_scalar('Train/C1_warmup_TotalLoss',W1_total,epoch)
            writer.add_scalar('Train/C2_warmup_ClassLoss', W2_Class,epoch) # C1 Loss ,,,
            writer.add_scalar('Train/C2_warmup_DomainSLoss',W2_s_d,epoch)
            writer.add_scalar('Train/C2_warmup_DomainTLoss',W2_t_d,epoch)
            writer.add_scalar('Train/C2_warmup_TotalLoss',W2_total,epoch)
        else:
            writer.add_scalar('Train/C1_train_LabeledLoss',Tr1_Lx)
            writer.add_scalar('Train/C1_train_UnLabeledLoss',Tr1_Lu)
            writer.add_scalar('Train/C1_train_DomainSLoss',Tr1_sd)
            writer.add_scalar('Train/C1_train_DomainTLoss',Tr1_td)
            writer.add_scalar('Train/C1_train_Total',Tr1_Total)
            writer.add_scalar('Train/C2_train_LabeledLoss',Tr2_Lx)
            writer.add_scalar('Train/C2_train_UnLabeledLoss',Tr2_Lu)
            writer.add_scalar('Train/C2_train_DomainSLoss',Tr2_sd)
            writer.add_scalar('Train/C2_train_DomainTLoss',Tr2_td)
            writer.add_scalar('Train/C2_train_Total',Tr2_Total)


