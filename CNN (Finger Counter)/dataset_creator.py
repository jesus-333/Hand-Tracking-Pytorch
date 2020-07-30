# -*- coding: utf-8 -*-
"""
Script used to create the database to train the CNN.
It visualize 1 box in the image with the number of fingers to shown. 
You should position the hand inside the boxes and after n_seconds the script save a copy of the image of the hand and the relative number of fingers.
After that anew box appear and so on. A good value of n_seconds is between 4 and 5.5

N.B. The automatic folder creation was implemented but not tested. If it create problem comment that part and create the folder by yourself.

@author: Alberto Zancanaro (Jesus)

"""
#%%

import numpy as np
import cv2
import math
import time
import random
import pickle
import os

#%% Variables

# Dimensions in pixel of the hand
square_dim = 140

# Save additional image with the boxes drawn in the check folder.  
save_check = True

# Path where save the various elements
path = 'Train'

# Variables used in the script
n_fingers = random.randint(0, 5)
pt1 = (440, 230)
pt2 = (pt1[0] + square_dim, pt1[1] + square_dim)

text_dict = {0:'zero', 1:'uno', 2:'due', 3:'tre', 4:'quattro', 5:'cinque'}
fingers_list = []

take_new_photo = True

#%% Read counter

file1 = open('counter.txt', 'r')
counter = int(file1.read())
file1.close()

#%%

try:
    os.makedirs(path)
    if(save_check): os.makedirs("Check")
except:
    print("ERROR DURING FOLDER CREATION")

# Open Camera
try:
    default = 0 # Try Changing it to 1 if webcam not found
    capture = cv2.VideoCapture(default)
    
    with open(path + "/fingers_list.txt", "rb") as fp:
        fingers_list = pickle.load(fp)
except:
    print("No Camera Source Found!")

while capture.isOpened():
    
    # Capture frames from the camera
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
        
    if(take_new_photo):
        start = time.time()
        n_fingers = random.randint(0, 5)
        take_new_photo = False
    
        if(int((time.time())) % 2 == 0):
            pt1 = (440, 230)
            pt2 = (pt1[0] + square_dim, pt1[1] + square_dim)
        else:
            pt1 = (50, 230)
            pt2 = (pt1[0] + square_dim, pt1[1] + square_dim)
            
        clr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    cv2.putText(frame, text_dict[n_fingers], (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = clr, thickness = 4)
    cv2.rectangle(frame, pt1, pt2, color = (0, 0, 255), thickness = 4)
    
    cv2.imshow("Frame", frame)
        
    end = time.time()
    
    if(end - start > 3.5):
        # print(end - start)
        print("saved element " + str(counter))
        print("- - - - - - - - - - - - - - - - - - - - \n")
        
        hand = frame_copy[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        
        # saveElements(path, str(counter), frame_copy, n_rect, pt_list)
        # cv2.imshow("hand", hand)
        cv2.imwrite(path + "/" + str(counter) + ".jpg", hand)
        if(save_check): cv2.imwrite("Check/" + str(counter) + ".jpg", frame)
        fingers_list.append(n_fingers)
        
        counter += 1
        take_new_photo = True
    
    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
    # if(counter >= 220):
    #     break

capture.release()
cv2.destroyAllWindows()

#%%

with open(path + "/fingers_list.txt", "wb") as fp:
    pickle.dump(fingers_list, fp)

file1 = open('counter.txt', 'w')
file1.write(str(counter))
# file1.write(str(0))
file1.close()
