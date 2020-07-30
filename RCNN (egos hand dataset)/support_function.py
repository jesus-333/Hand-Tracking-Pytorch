import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

import scipy.io
import cv2

#%%

def getListBoxes(boxes):
    boxes_tmp = np.zeros((100, 4))
    
    a = boxes['myleft']
    b = boxes['myright']
    c = boxes['yourleft']
    d = boxes['yourright']
    
    return [a, b, c, d]

def getBoxes(idx, boxes):
    tmp_list = []
    
    for i in range(4):
        points_list = boxes[i][idx].squeeze()
        if (points_list.shape != (0,) and points_list.shape != (0, 0)):
            # print(points_list.shape)
            tmp_list.append(getCorner(points_list))
            
    return tmp_list
    
    
def getCorner(points_list):
    min_x = int(min(points_list[:, 0]))
    min_y = int(min(points_list[:, 1]))
    max_x = int(max(points_list[:, 0]))
    max_y = int(max(points_list[:, 1]))
    
    return (min_x, min_y), (max_x, max_y)

def convertBoxes(boxes):
    """
    Convert the boxes of the dataset in the format used in the network
    i.e.: boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
    """
    
    boxes_matrix = np.zeros((len(boxes), 4))
    
    for i, box in zip(range(len(boxes)), boxes):
        boxes_matrix[i, :] = [box[0][0], box[0][1], box[1][0], box[1][1]]
        
    return boxes_matrix


def showBoxes_1(idx, filepath_list, boxes_list, color = (255, 0, 128), thickness = 2):
    img = cv2.imread(filepath_list[idx])
    
    for box in boxes_list[idx]:
        cv2.rectangle(img, box[0], box[1], color, thickness)
        
        
    cv2.imshow("Test Boxes" + filepath_list[idx], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def get_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
  
    
    
    
    
        