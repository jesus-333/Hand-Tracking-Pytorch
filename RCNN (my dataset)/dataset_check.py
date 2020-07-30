# -*- coding: utf-8 -*-
"""
Simple file to check the dataset

@author: Alberto Zancanaro (Jesus)
"""
#%%

import numpy as np
import cv2
import math
import random

from support_function import *

import torch

#%%

image_path = "Train/Image/"
boxes_path = "Train/Boxes/"

n_example = 2

image = cv2.imread(image_path + str(n_example) + '.jpg')
boxes = np.load(boxes_path + str(n_example) + '.npy')


#%%
for i in range(2):
    pt1 = (int(boxes[0, i]), int(boxes[1, i]))
    pt2 = (int(boxes[2, i]), int(boxes[3, i]))
    
    cv2.rectangle(image, pt1, pt2, color = (0, 255, 0), thickness = 4)

cv2.imshow("Test Boxes " + str(n_example),image)

cv2.waitKey(0)
cv2.destroyAllWindows()