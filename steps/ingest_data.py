import logging
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

import os
# for dirname, _, filenames in os.walk('/kaggle/input/custom-digital-malayalam/digital_malayalam'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         break
    # break

from zenml import step
class IngestData:
    '''
        Ingesting data from the data_path
    '''
    def __init__(self, data_path) -> None:
        '''
        Args:
            data_path: path to the data
        '''
        self.data_path = data_path 
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        height,width = 30,30
        channels = 1
        
        class SingleChannel(object):
            def __call__(self,img):
                img_array = np.array(img)
                img_single_channel = img_array[:,:,0]
                negated = 255-img_single_channel
                return Image.fromarray(negated)

        transformer = transforms.Compose([
            SingleChannel(),
            transforms.ToTensor(),
            transforms.Resize((height,width)),
        ])

        real_dataset = datasets.ImageFolder(self.data_path, transform=transformer)

        real_train_size = int(0.8 * len(real_dataset))
        real_test_size = len(real_dataset) - real_train_size

        real_train_dataset, real_test_dataset = torch.utils.data.random_split(real_dataset, [real_train_size, real_test_size])

        real_trainloader = torch.utils.data.DataLoader(real_train_dataset, batch_size = 64, shuffle=True)
        real_testloader = torch.utils.data.DataLoader(real_test_dataset, batch_size = 64, shuffle=True)
        char_to_label = {}
        label_to_char = {}

        for i, (k,v) in enumerate(real_dataset.class_to_idx.items()):
            char_to_label[k] = v
            label_to_char[v] = k
        num_classes = len(char_to_label)
        print("Number of classes: ", num_classes)

        return real_trainloader, real_testloader, char_to_label, label_to_char, num_classes
        

# @step
def ingest_data(data_path: str):
    '''
    Ingesting the data from the data path

    Args:
        data_path: path to the data
    '''
    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.get_data()
        print("LENGTH OF DATA :", len(data))
        return data
    except Exception as e:
        logging.error("Error while ingesting data: {e}")
        raise e