# -*- coding: utf-8 -*-
"""
File containing the dataset class used in the train of the CNN. 
The class is an extension of the Dataset class provided by Pytorch.
To optimize memory consumption the dataset doesn't store the images but only the path to them. Images are read on fly when you access to an element of the dataset

Work in similar way to EgoDataset and MyDataset

@author: Alberto Zancanaro (Jesus)
"""

#%%

import os
import numpy as np
from PIL import Image
import pickle
import random
import cv2

#from support_function import *

import torch
import torch.utils.data
import torchvision.transforms as T

#%%

class HandClassificationDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, n_elements = -1, transforms = None):
        if(n_elements > 200 or n_elements< 0): n_elements = 200
                
        tmp_list = [element for element in os.walk(path)][0][2]
        elements_list = []
        
        for element in tmp_list:
            if('jpg' in element):
                elements_list.append(element)
            else:
                with open(path + "/" + element, "rb") as fp:
                    y_list = pickle.load(fp)
                
        elements_list.sort()
        # tmp_index_list = np.linspace(0,len(elements_list) - 1, len(elements_list)).astype(int)
        # np.random.shuffle(tmp_index_list)
        # elements_list = [elements_list[i] for i in tmp_index_list]
        # y_list = [y_list[j] for j in tmp_index_list]
                
        # c = list(zip(elements_list, boxes_list))
        # random.shuffle(c)
        # elements_list, boxes_list = zip(*c)
            
                
        self.transforms = transforms
        self.elements = elements_list
        self.y_list = list(np.float_(y_list))
        # self.y_list = y_list
        self.path = path
        
        self.get_image_PIL = False

    def __getitem__(self, idx):
        # Retrieve image
        img = Image.open(self.path + "/" + self.elements[idx]).convert("RGB")
    
        label = self.y_list[int(self.elements[idx][:-4])]
        
        if ((self.transforms is not None) and (self.get_image_PIL == False)):
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.elements)
    

#%%
        
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_transform_2():
    transforms = []
    
    transforms.append(TransformOpenCV())
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

    
class TransformOpenCV(object):

    def __call__(self, img):
        kernel = np.ones((4,4))
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, np.array([2, 0, 0]), np.array([20, int(255 * 0.68), 255]))
        img = cv2.dilate(img, kernel, iterations = 2)
        img = cv2.erode(img, kernel, iterations = 2)
        
        return img