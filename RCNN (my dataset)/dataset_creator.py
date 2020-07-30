# -*- coding: utf-8 -*-
"""
Script used to create the database to train the RCNN.
It visualize n_boxes in the image. You should position the hand inside the boxes and after n_seconds the script save a copy of the image and the position of the boxes.
After that new boxes appear and so on. A good value of n_seconds is between 4 and 5.5

N.B. The automatic folder creation was implemented but not tested. If it create problem comment that part and create the folder by yourself.
     The folder must be 'Image', 'Boxes' and 'Check' (if you set the save_check variable to True)

@author: Alberto Zancanaro (Jesus)
"""
#%%

import numpy as np
import cv2
import math
import time
import random

import os

from support_function import *

#%% Variables

# Max and min dimension of boxes
max_dim_rect = 160
min_dim_rect = 90

# Number of rect in the image
n_rect = 2

# Offset from the border
offset = 15

# Frame (camera) resolution
frame_shape = (640, 480)

# Save additional image with the boxes drawn in the check folder.  
save_check = True

# Path where save the various elements
path = 'Train'

# Number of second 
n_seconds = 4.5

# variables used in the code
draw_new_box = True
pt_list = []

#%% Read counter (USED TO GIVE AN UNIQUE ID TO EACH IMAGE)

file1 = open('counter.txt', 'r')
counter = int(file1.read())
file1.close()

#%% Create the folder

try:
    os.makedirs(path)
    os.makedirs(path + "/Image")
    os.makedirs(path + "/Boxes")
    if(save_check): os.makedirs(path + "/Check")
except:
    print("ERROR DURING FOLDER CREATION")

#%%

# Open Camera
try:
    default = 0 # Try Changing it to 1 if webcam not found
    capture = cv2.VideoCapture(default)
except:
    print("No Camera Source Found!")

while capture.isOpened():
    
    # Capture frames from the camera
    ret, frame = capture.read()
    
    frame_copy = frame.copy()
        
    if(draw_new_box):
        start = time.time()
        
        pt_list = []
        if(n_rect == 1):
            pt_list.append(generateBoxPoints(frame_shape, min_dim_rect, max_dim_rect, limit_x = (offset, 640 - offset)))
        elif(n_rect == 2):
            pt_list.append(generateBoxPoints(frame_shape, min_dim_rect, max_dim_rect, limit_x = (0 + offset, 320 - offset)))
            pt_list.append(generateBoxPoints(frame_shape, min_dim_rect, max_dim_rect, limit_x = (320 + offset, 640 - offset)))
        else:
            for i in range(n_rect): pt_list.append(generateBoxPoints(frame_shape, min_dim_rect, max_dim_rect, limit_x = (offset, 640 - offset)))
        
        draw_new_box = False
    
    for i in range(n_rect):
        # clr = tuple(np.random.randint(0, 255, 3))
        # cv2.rectangle(img = frame, pt1 = pt_list[i][0], pt2 = pt_list[i][1], color = clr, thickness = 4)
        cv2.rectangle(frame, pt_list[i][0], pt_list[i][1], color = (0, 0, 255), thickness = 4)
    
    # cv2.line(frame, (320, 0), (320, 480), color = (255, 0, 0), thickness = 4)
    
    frame = np.fliplr(frame)
    cv2.imshow("Frame", frame)
    
    end = time.time()
    
    if(end - start > n_seconds):
        print("saved element " + str(counter))
        print("- - - - - - - - - - - - - - - - - - - - \n")
        
        # Save the original frame, the boxes file
        saveElements(path, str(counter), frame_copy, n_rect, pt_list)
        
        # Save image with the boxes drawn to check later (OPTION)
        if(save_check): cv2.imwrite(path + "/Check/" + str(counter) + '.jpg', frame)
        
        counter += 1
        n_rect = random.choice([1, 2])
        
        draw_new_box = True
    
    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
    # if(counter >= 4):
    #     break

capture.release()
cv2.destroyAllWindows()

#%%

file1 = open('counter.txt', 'w')
file1.write(str(counter))
# file1.write(str(0))
file1.close()
