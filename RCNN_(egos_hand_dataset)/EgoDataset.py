# -*- coding: utf-8 -*-
"""
File containing the dataset class used in the train of the RCNN. 
This dataset is the EgoHands dataset.
The class is an extension of the Dataset class provided by Pytorch.
To optimize memory consumption the dataset doesn't store the images but only the path to them. Images are read on fly when you access to an element of the dataset

Based on the script and tutorial on the Pytorch website (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

N.B. Inside path you must have the folders that you find inside _LABELLED_SAMPLES.

@author: Alberto Zancanaro (Jesus)
"""

#%%

import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import scipy
from PIL import Image

import random
import cv2
from support_function import *

#%%

class EgoDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, n_elements = -1, transforms = None):
        if(n_elements > 2400): n_elements = 450
        entire_folder = int(n_elements / 100) # n_elements will become the nearest lower multiple of 100
        sample_last_Folder = n_elements % 100 # Not used
        
        # Work similar to MyDataset
        folder_list = []
        for element in os.walk(path): folder_list.append(element)
        del folder_list[0]
        
        folder_list = random.sample(folder_list, entire_folder + 1)
        
        elements_list = []
        boxes_list = []
        for folder in folder_list:
            boxes = np.squeeze(scipy.io.loadmat(folder[0] + "/" + folder[2][100])['polygons'])
            boxes = getListBoxes(boxes)
            folder[2].sort()
            for i, photo_name in zip(range(100), folder[2]):
                elements_list.append(folder[0] + "/" + photo_name)
                boxes_list.append(getBoxes(i, boxes))
        
        # Shuffle elements
        tmp_index_list = np.linspace(0,len(elements_list) - 1, len(elements_list)).astype(int)
        np.random.shuffle(tmp_index_list)
        elements_list = [elements_list[i] for i in tmp_index_list]
        boxes_list = [boxes_list[j] for j in tmp_index_list]
                            
                
        self.transforms = transforms
        self.boxes_list = boxes_list
        self.elements = elements_list
        
        self.get_image_PIL = False

    def __getitem__(self, idx):
        # Retrieve image
        img = Image.open(self.elements[idx]).convert("RGB")
        
        # Retrieve boxes and convert into tensor
        boxes = self.boxes_list[idx]
        boxes = convertBoxes(boxes)
        self.last_boxes = boxes
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Things use in the original script
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        # Parameters needed for RCNN training
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Return the image in PIL format or applied the transform 
        if ((self.transforms is not None) and (self.get_image_PIL == False)):
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.elements)
    
    def checkBoxes(self, idx, color = (255, 0, 128), thickness = 2):
        tmp_bool = self.get_image_PIL
        self.get_image_PIL = True
        img, target = self[idx]
        self.get_image_PIL = tmp_bool
        
        img_opencv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = target['boxes'].numpy()
        
        for box in boxes:
            cv2.rectangle(img_opencv, (box[0], box[1]), (box[2], box[3]), color = color, thickness = thickness)
            
        cv2.imshow("check box "  + str(idx) + " - " + self.elements[idx], img_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return img, boxes
    
    def checkBoxes_2(self, idx, color = (255, 0, 128), thickness = 2):
        img_opencv = cv2.imread(self.elements[idx])
        boxes = self.boxes_list[idx]
        
        for box in boxes:
            cv2.rectangle(img_opencv, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color = color, thickness = thickness)
        
        
        cv2.imshow("check box "  + str(idx) + " - " + self.elements[idx], img_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
#%%
        
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_transform_2():
    # Kernel for morphological transformation    
   
        
    transforms = []
    
    transforms.append(TransformOpenCV())
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


#%%
    
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