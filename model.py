import torch.nn as nn
from torch.nn.modules import utils
from torch.nn.modules.linear import Linear
from functions import ReverseLayerF
import math
import torch 
#import network
from network import resnet
from torchvision import models

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

# Basic Model
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))   
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

#MNIST Model
class MNISTmodel(nn.Module):
    """ MNIST architecture"""
    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output

"""MNIST --> Mnist_m的三段Structure"""

#F
class Feature(nn.Module):
    """Feature class for MNIST -> MNIST-M"""
    def __init__(self):
        super(Feature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28,kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(28), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=28, out_channels=48,kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )
    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        return feature
#C
class Classifier(nn.Module):
    """classifier class for MNIST -> MNIST-M"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

    def forward(self, feature):
        class_output = self.classifier(feature)
        return class_output
#D
class Discriminator(nn.Module):
    """classifier class for MNIST -> MNIST-M"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.discriminator(reverse_feature)
        return domain_output

"""MNIST <--SynDigits的三段Structure"""

class cnn_tar_ms(nn.Module):
    def __init__(self):
        super(cnn_tar_ms, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),  # 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),  # 9
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 4
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1) # ???
        class_output = self.classifier(feature)
        return feature, class_output
    

class Feature_Synth(nn.Module):
    def __init__(self):
        super(Feature_Synth, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),  # 28
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),  # 9
            #nn.BatchNorm2d(64),
            #nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 4
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),  # 1
        )
    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1) # ???
        return feature

class Classifier_Synth(nn.Module):
    def __init__(self):
        super(Classifier_Synth, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            # nn.Linear(128 * 1 * 1, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(2048, 10)
        )

    def forward(self, feature):
        class_output = self.classifier(feature)
        return class_output

class Discriminator_Synth(nn.Module):
    def __init__(self):
        super(Discriminator_Synth,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.discriminator(reverse_feature)
        return domain_output

'''New Structure Office'''
class ResNet50(nn.Module):
    def __init__(self,args):
        super(ResNet50, self).__init__()
        self.img_size = args.image_size
        resnetModel = models.resnet50(pretrained=True)
        feature_map = list(resnetModel.children())
        feature_map.pop()
        self.feature_extractor = nn.Sequential(*feature_map)
        
        self.classifier = nn.Sequential(
            #nn.Linear(2048, 31),
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,31)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, self.img_size, self.img_size)
        feature = self.feature_extractor(input_data)
        feature = feature.view(-1, 2048)
        reverse_bottleneck = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_bottleneck)

        return class_output, domain_output

# 三段式结构，拆分后 office-31
##################
# Feature Extrator & classifier 
##################

# # backbone
# class ResNet50Fc(nn.Module):
#     def __init__(self):
#         super(ResNet50Fc, self).__init__()
#         model_resnet50 = models.resnet50(pretrained=True)
#         self.conv1 = model_resnet50.conv1
#         self.bn1 = model_resnet50.bn1
#         self.relu = model_resnet50.relu
#         self.maxpool = model_resnet50.maxpool
#         self.layer1 = model_resnet50.layer1
#         self.layer2 = model_resnet50.layer2
#         self.layer3 = model_resnet50.layer3
#         self.layer4 = model_resnet50.layer4
#         self.avgpool = model_resnet50.avgpool
#         self.__in_features = model_resnet50.fc.in_features

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return x

#     def output_num(self):
#         return self.__in_features

# class office_Feature(nn.Module):
#     def __init__(self,args):
#         super(office_Feature,self).__init__()
#         self.base_network = ResNet50Fc()
#         self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(),1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.5)]
#         self.feature_extractor = nn.Sequential(*self.bottleneck_layer_list)
        
#     def forward(self,input):
#         input_data = input.expand(input.data.shape[0],3,224,224)
#         x_base = self.base_network(input_data)
#         x = self.feature_extractor(x_base)
#         feature = x.view(-1, 1024)
#         return feature

# ##################
# # Label Classifier
# ###################
# class office_classifier(nn.Module):
#     def __init__(self,args):
#         super(office_classifier,self).__init__()
#         self.img_size = args.image_size
#         self.classifier = nn.Sequential(
#             nn.Linear(1024,256),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(256,65)
#         )
#     def forward(self, feature):
#         class_output = self.classifier(feature)
#         return class_output

###############################
##################
# Feature Extrator & classifier 
##################
class office_Feature(nn.Module):
    def __init__(self,args):
        super(office_Feature,self).__init__()
        self.img_size = args.image_size
        resnetModel = models.resnet50(pretrained=True)
        feature_map = list(resnetModel.children())
        feature_map.pop() # 用于移除最后一个元素
        self.feature_extractor = nn.Sequential(*feature_map)
        
    def forward(self,input):
        input_data = input.expand(input.data.shape[0],3,256,256)
        x = self.feature_extractor(input_data)
        feature = x.view(-1, 2048)
        return feature

##################
# Label Classifier
###################
class office_classifier(nn.Module):
    def __init__(self,args):
        super(office_classifier,self).__init__()
        self.img_size = args.image_size
        self.classifier = nn.Sequential(
            nn.Linear(2048,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,31)
            #nn.Linear(256,65)
        )
    def forward(self, feature):
        class_output = self.classifier(feature)
        return class_output

###################
# Domain Classifier
###################
class office_discriminator(nn.Module):
    def __init__(self,args):
        super(office_discriminator,self).__init__()
        self.img_size = args.image_size
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 2),
        )
    def forward(self,feature,alpha):
        reverse_bottleneck = ReverseLayerF.apply(feature, alpha)
        domain_output = self.discriminator(reverse_bottleneck)
        return domain_output



# cifar and slt ResNet 18
class cifar_feature(nn.Module):
    def __init__(self,args):
        super(cifar_feature,self).__init__()
        self.img_size = args.image_size
        resnetModel = models.resnet50(pretrained=True)
        feature_map = list(resnetModel.children())
        feature_map.pop() # 用于移除最后一个元素
        self.feature_extractor = nn.Sequential(*feature_map)
        
    def forward(self,input):
        input_data = input.expand(input.data.shape[0],3,32,32)
        x = self.feature_extractor(input_data)
        feature = x.view(-1,2048)
        return feature

class cifar_classifier(nn.Module):
    def __init__(self,args):
        super(cifar_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,9)
        )
    def forward(self, feature):
        class_output = self.classifier(feature)
        return class_output

