# -*- coding: utf-8 -*-
"""
File containing the CNN to train. There are two network inside this file. We use the second (CNNModel)
The class is an extension of the Dataset class provided by Pytorch.
To optimize memory consumption the dataset doesn't store the images but only the path to them. Images are read on fly when you access to an element of the dataset

Work in similar way to EgoDataset and MyDataset

@author: Alberto Zancanaro (Jesus)
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

import json
import pickle
import pandas as pd
import os

    
#%% Neural Network

class handNet(nn.Module):
    
    
    def __init__(self, activation_function = 0, image_size = (140, 140), kernel_size = 3, pool_var = False):
        super().__init__()
        
        self.pool_var = pool_var
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        #Define the activation function
        self.act = []
        self.act.append(nn.Sigmoid()) #0
        self.act.append(nn.ReLU()) #1
        self.act.append(nn.LeakyReLU()) #2
        self.act.append(nn.Tanh()) #3
        self.act.append(nn.SELU()) #4
        self.act.append(nn.Hardshrink()) #5
        self.act.append(nn.Hardtanh()) #6
        
        self.activation = self.act[activation_function]
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.conv1 = nn.Conv2d(3, 6, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.n_neurons_fc = self.evaluateNeuronFCLayer(image_size)
        # self.n_neurons_fc = 25088
        
        self.fc1 = nn.Linear(self.n_neurons_fc, 2048)
        self.fc2 = nn.Linear(2048, 256)
        # self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 1)
        self.softmax = nn.Softmax(dim = 1)
        
        
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
        
    def forward(self, x,):
        if(self.pool_var):
            x = self.pool(self.activation(self.conv1(x)))
            x = self.pool(self.activation(self.conv2(x)))
            x = self.pool(self.activation(self.conv3(x)))
        else:
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
        x = x.view(x.size()[0], self.n_neurons_fc)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = (self.act[1](x))
        
        return x
    
        
    def evaluateNeuronFCLayer(self, img_input_size):
        x = torch.ones(1, 3, img_input_size[0], img_input_size[1])
        x = self.conv1(x)
        if(self.pool_var): x = self.pool(x)
        x = self.conv2(x)
        if(self.pool_var): x = self.pool(x)
        x = self.conv3(x)
        if(self.pool_var): x = self.pool(x)
        
        # print(x.size())
        # print("new size: ", x.size()[1] * x.size()[2] * x.size()[3])
        
        return x.size()[1] * x.size()[2] * x.size()[3]
        

#%%

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        # conv 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Maxpool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # conv 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Maxpool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # conv 3
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        # Maxpool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # conv 4
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.relu4 = nn.ReLU()

        # Maxpool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # FC 1
        # self.fc1=nn.Linear(128*20*20,512)
        self.fc1 = nn.Linear(12800,512)
        self.relu5 = nn.ReLU()
        
        # FC 2
        self.fc2 = nn.Linear(512,6)
        
        # self.softmax = nn.Softmax()

    def forward(self,x):
        # conv1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # max poo1 1
        out = self.maxpool1(out)
        
        # conv2
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # max poo1 2
        out = self.maxpool2(out)
        
        # conv3
        out = self.cnn3(out)
        out = self.relu3(out)
        
        # max poo1 3
        out = self.maxpool3(out)
        
        # conv4
        out = self.cnn4(out)
        out = self.relu4(out)
        
        # max poo1 4
        out = self.maxpool4(out)
           
        out = out.view(out.size(0),-1)
                
        # fc1
        out=self.relu5(self.fc1(out))
        
        # fc2
        out=self.fc2(out)
        
        # out = self.softmax(out)
        
        return out

#%%
            
def trainEpoch(network, data_train, optimizer, loss_fn):
    x_train = data_train[0]
    y_train = data_train[1]
 
    # Zeroes the gradient buffers of all parameters
    optimizer.zero_grad()     
    
    # Forward pass
    out = network(x_train)
    
    # Evaluate loss
    train_loss = loss_fn(out, y_train)
    
    # Backward pass
    train_loss.backward()
    
    # Update
    optimizer.step()
        
    #Return loss
    return train_loss.data


def saveObject(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def loadObject(filename):
    with open (filename, 'rb') as fp:
        return pickle.load(fp)
    
    
def loadDataset(path, reduce_size = (64, 64)):
    element_list = [element for element in os.walk(path)][0][2]
    
    x = torch.ones(len(element_list) - 1, 3, reduce_size[0], reduce_size[1])
    y = torch.zeros(len(element_list) - 1, 6)
    
    i = 0
    for element in element_list:
        if('jpg' in element):
            tmp_x  = cv2.imread(path + "/" + element)
            tmp_x = cv2.resize(tmp_x, reduce_size, interpolation = cv2.INTER_CUBIC)
            x[i, :, :, :] = torch.from_numpy(tmp_x).permute(2,1,0).float()
            i += 1
            
            # if(i == 13):
            #     cv2.imshow("x" + str(i), tmp_x)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
        else:
            with open(path + "/" + element, "rb") as fp:
                y_list = pickle.load(fp)
                
    # One hot encoding
    # for label, i in zip(y_list, range(len(y_list))): y[i, label] = 1
    y = torch.FloatTensor(y_list)
            
    # x normalization between 0 and 1                
    x = x / 255

    return x, y      
