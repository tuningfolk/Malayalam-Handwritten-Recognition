import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms, datasets

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import logging

height,weight = 30,30
channels = 1
class ConvNet(nn.Module):
    #initialize the class & the parameters
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.n1 = height
        
        self.f1 = 3 #Random
        self.k1 = 64
        self.O1 = self.n1 - self.f1 + 1
        
        self.f2 = 3
        self.s2 = 2
        self.O2 = np.floor((self.O1 - self.f2)/(self.s2)) + 1
        
        self.f3 = 3
        self.k3 = 32
        self.O3 = self.O2 - self.f3 + 1
        
        self.f4 = 3
        self.s4 = 2
        self.O4 = int(np.floor((self.O3 - self.f4)/(self.s4)) + 1)
        
        self.n5 = num_classes
        #conv layer 1 + max pool
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.k1, kernel_size=self.f1),
            nn.MaxPool2d(kernel_size=self.f2, stride=self.s2)
        )
        
        #conv layer 2 + max pool 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.k1, out_channels=self.k3, kernel_size=self.f3),
            nn.MaxPool2d(kernel_size=self.f4, stride=self.s4)
        )
        
        #Avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
        #FC layer
        self.fc = nn.Linear(in_features=self.k3*9, out_features=num_classes)
        
    def forward(self, x, visualize=False):
        out = self.layer1(x)
        if visualize:
            print("Layer 1 output: ", out[:,0:3,:,:].shape)
            # show_batch(out[:,0:3,:,:])
        out = self.layer2(out)
        if visualize:
            print("Layer 2 output:", out.shape)
            # show_batch(out[:,0:3,:,:])
        out = self.avgpool(out)
        if visualize:
            print("Avg pool output: ", out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# print(model)

def train_model(data, num_classes):
    
    model = ConvNet(num_classes)
    print(model)
    return model