# -*- coding: utf-8 -*-
"""
File containing various function used by the script dataset_creator.py, MyDataset.py and train_RCNN.py

@author: Alberto Zancanaro (Jesus)
"""

#%%

import os
import numpy as np
import cv2
import random

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#%% Function for the dataset creation

def generateBoxPoints(frame_resolution, min_dim_rect = 80, max_dim_rect = 160, limit_x = (-1, -1), limit_y = (-1, -1)):
    """
    Generate the pair of point that define the box

    Parameters
    ----------
    frame_resolution : tuple
        Tuple containing the resolution of the image in wich generate the boxes.
    min_dim_rect : int, optional
        DESCRIPTION. The default is 80.
    max_dim_rect : int, optional
        DESCRIPTION. The default is 160.
    limit_x : tuple, optional
        Set the minimum and maximum value of the coordinate x for the box. If it's (-1, -1) the script use the frame width as limit. The default is (-1, -1).
    limit_y : tuple, optional
        Set the minimum and maximum value of the coordinate y for the box. If it's (-1, -1) the script use the frame height as limit. The default is (-1, -1).

    Returns
    -------
    pt1 : tuple
        Point 1 of the boxes in the form (x, y).
    pt2 : tuple
        Point 1 of the boxes in the form (x, y).

    """
    
    randint = np.random.randint
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Generate point 1 (pt1)
    
    if(limit_x != (-1, -1)): x1 = randint(limit_x[0], limit_x[1])
    else: x1 = randint(0, frame_resolution[0])
    
    if(limit_y != (-1, -1)): y1 = randint(limit_y[0], limit_y[1])
    else: y1 = randint(0, frame_resolution[1])
    
    pt1 = (x1, y1)
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Generate point 2 (pt2)
    
    bad_point = True
    
    # Since the random generation pt2 can have invalid coordinate. So the script continue to generat point until a valid point is generated
    while(bad_point):
        x2 = x1 + random.choice((-1, 1)) * randint(min_dim_rect, max_dim_rect)
        y2 = y1 + random.choice((-1, 1)) * randint(min_dim_rect, max_dim_rect)
        
        if not (x2 > frame_resolution[0] or x2 < 0 or y2 > frame_resolution[1] or y2 < 0): bad_point = False
            
        if(limit_x != (-1, -1) and (x2 < limit_x[0] or x2 > limit_x[1])): bad_point = True
        if(limit_y != (-1, -1) and (y2 < limit_y[0] or y2 > limit_y[1])): bad_point = True
            
    pt2 = (x2, y2)
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    return pt1, pt2


def saveElements(path, filename, frame, n_rect, rect_list):
    """
    Save the element

    Parameters
    ----------
    path : string
        Path of the file. WITHOUT the name of the file
    filename : string
        Name of the file. WITHOUT extension.
    frame : Numpy matrix
        Image to save.
    n_rect : int
        Number of boxes in the image.
    rect_list : list
        List of boxes in the image.

    """
    
    boxes = np.zeros((4, n_rect))
    
    for i in range(n_rect): boxes[:, i] = convertRect(rect_list[i])
    
    # boxes[:, 0] = convertRect(rect_1)
    # boxes[:, 1] = convertRect(rect_2)
    
    with open(path + "/Boxes/" + filename + '.npy', 'wb') as f:
        np.save(f, boxes)
        # np.save(f, rect_2_vet)
    
    cv2.imwrite(path + "/Image/" + filename + '.jpg', frame)
    
    # print("saved")
    # print(path + "/Boxes/" + filename + '.npy')
    # print(path + "/Train/" + filename + '.jpg')
    # print("-----------------------------")
    
    
    
def convertRect(rect):
    """
    Used in saveElements() function. Convert the boxes format in a Numpy array of 1 x 4.
    The value are [x1, y1, x2, y2]

    """
    tmp_vet = np.zeros(4)
    
    tmp_vet[0] = rect[0][0]
    tmp_vet[1] = rect[0][1]
    tmp_vet[2] = rect[1][0]
    tmp_vet[3] = rect[1][1]
    
    return tmp_vet

#%% Function for the dataset reading and training

def convertBoxes(boxes_path):
    """
    Read the npy file containing the boxes matrix.
    Since the boxes are randomly generated the function check that (x1, y1) is the upper left corner and (x2, y2) is the down right corner
    """
    tmp_matrix = np.load(boxes_path).T
    
    for box, i in zip(tmp_matrix, range(tmp_matrix.shape[0])):
        if(box[0] > box[2]):
            tmp_matrix[i, 0], tmp_matrix[i, 2] = box[2], box[0]
            
        if(box[1] > box[3]):
            tmp_matrix[i, 1], tmp_matrix[i, 3] = box[3], box[1]
            
    return tmp_matrix

def get_model():
    """
    Get the model from torchvision library

    Returns
    -------
    model : Pytorch model
        Return the fasterrcnn_resnet50 trained on COCO.

    """
    
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
        
    