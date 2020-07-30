import numpy as np
import cv2
import math
import socket
import time

from hand_detector_utils import *
# from hand_classification_net import *

import torch

#%% Settings

# Use 0 if you use the laptot camera. 
#If you use a software to split the video source or similar try 1, 2 etc. (Start with 1)
default_device = 1 

# If you only want to test this program without the Unity application set to True to obtain a mirrored feedback
# If you use a third party application to split the video source use that application to obtain the mirror effetct and set this variable to False
flip_frame = True
if(default_device == 1): flip_frame = False

#%% Other variables (don't modify)

# Variables for UPD use
UDP_IP = "127.0.0.1"
UDP_PORT = 5065

# Socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Counter use to FPS improvement
counter = 0

# Variable to save the hand image
hand1 = np.zeros((10, 10, 3))
hand2 = np.zeros((10, 10, 3))

# Predictions regarding the actual frame of the number of finger per hand
finger_predicts_1 = -1
finger_predicts_2 = -1

# Tracking of fingers counter of the two hands. Use to decide wich command send
finger_list_1 = []
finger_list_2 = []

# Flip the image of left hand to improve classifier performance
flip_hand_1 = True

#%%
# Load NN
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hand_tracker = torch.load("model/model_all_40.pth")
hand_tracker.to(device)
hand_tracker.eval()

finger_counter = torch.load("model/CNN_250_140.pth")
# finger_counter = CNNModel()
# finger_counter.load_state_dict(torch.load("model/model_chk.pt"))
finger_counter.to(device)
finger_counter.eval()

#%%

# Open Camera
try:
    capture = cv2.VideoCapture(default_device)
    # capture = cv2.VideoCapture(cv2.CAP_DSHOW)
except:
    print("No Camera Source Found!")
    

for i in range(15): ret, frame = capture.read()


while capture.isOpened():
    
    # Capture frames from the camera
    ret, frame = capture.read()
    # frame = cv2.resize(frame, (320, 160))
    if(flip_frame):  frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Neural network section (Hand tracking and finger counter)
    # To improve performance the two network where activated every tot frame
    
    if(counter %  3 == 0):
        # Predict hand position
        boxes_predict = trackingHandWithRCNN(hand_tracker, frame, device)
        
        # Reset counter
        counter = 1
        
        # Performe other action only if at least 1 hand is detected
        if(len(boxes_predict) > 0): 
            
            if(len(boxes_predict) == 2):
                # If the net detect 2 hands, hand1 will becoome the left one and hand2 the right one
                if(boxes_predict[0,0] < boxes_predict[1,0]):
                    hand1 = frame_copy[boxes_predict[0,1]:boxes_predict[0,3], boxes_predict[0,0]:boxes_predict[0,2]]
                    hand2 = frame_copy[boxes_predict[1,1]:boxes_predict[1,3], boxes_predict[1,0]:boxes_predict[1,2]]
                else:
                    hand1 = frame_copy[boxes_predict[1,1]:boxes_predict[1,3], boxes_predict[1,0]:boxes_predict[1,2]]
                    hand2 = frame_copy[boxes_predict[0,1]:boxes_predict[0,3], boxes_predict[0,0]:boxes_predict[0,2]]
                
                # Denoise both hand
                if(flip_hand_1): hand1 = cv2.flip(hand1, 1)
                hand1 = cv2.GaussianBlur(hand1, (3,3), 0)
                hand2 = cv2.GaussianBlur(hand2, (3,3), 0)
                
                # Predict finger for both hand
                finger_predicts_1 = predictFingers(finger_counter, hand1, device)
                finger_predicts_2 = predictFingers(finger_counter, hand2, device)
            else:
                # If only 1 (or more than 2) hand are detected take only the first
                hand1 = frame_copy[boxes_predict[0,1]:boxes_predict[0,3], boxes_predict[0,0]:boxes_predict[0,2]]
                hand2 = np.zeros((10, 10, 3))
                
                # Denoise hand image
                if(flip_hand_1 and boxes_predict[0, 2] < frame.shape[1] / 2): hand1 = cv2.flip(hand1, 1)
                hand1 = cv2.GaussianBlur(hand1, (3,3), 0)
                
                # Predict fingers for the hand and set the other finger counter to -1
                finger_predicts_1 = predictFingers(finger_counter, hand1, device)
                finger_predicts_2 = -1
                
            
            # Add finger number to the list
            finger_list_1.append(finger_predicts_1)
            finger_list_2.append(finger_predicts_2)
            
            # Mantain only the last five element
            if(len(finger_list_1) > 5):
                finger_list_1 = finger_list_1[-5:]
            if(len(finger_list_2) > 5):
                finger_list_2 = finger_list_2[-5:]
                
    # print("finger 1: ", finger_predicts_1)
    cv2.putText(frame, "finger 1: " + str(finger_predicts_1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # if(finger_predicts_2 != -1): 
        # print("   finger 2: ", finger_predicts_2)
    cv2.putText(frame, "finger 2: " + str(finger_predicts_2), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Draw box around hand(s)
    if(-1 not in boxes_predict):
        for line in boxes_predict:
            # Extract point
            pt1 = (int(line[0]), int(line[1]))
            pt2 = (int(line[2]), int(line[3]))
            
            # Draw rectangle
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), thickness = 4)
               
            # Draw central point of the rectangle
            cv2.circle(frame, centralPointInBox(line) , 10, [255,0,255], -1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Show the frame (and optionally the hand image)
    cv2.imshow("Full Frame", frame)
    # cv2.imshow("Frame copy", frame_copy)
    cv2.imshow("Hand 1 (SX)", cv2.resize(hand1, (140, 140)))
    # cv2.imshow("Hand 2 (DX)", hand2)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create and send command
    
    if(len(boxes_predict) > 0):
        if(-1 not in finger_list_2):  # If both hands are on the screen
            # Evaluate central points for both hand
            central_point_1 = centralPointInBox(boxes_predict[0])
            central_point_2 = centralPointInBox(boxes_predict[1])
            
            # Create the command with the position of the central point
            command_hand_position_1 = "h 1 " +  str(central_point_1[0]) + " " + str(central_point_1[1])
            command_hand_position_2 = "h 2 " +  str(central_point_2[0]) + " " + str(central_point_2[1])
            
            # Ration of finger counter necessary to obtain hand close or open
            ratio = 3/5
            
            # Creation finger command for hand 1
            if(finger_list_1.count(5)/5 >= ratio or finger_list_1.count(4)/5 >= ratio or (finger_list_1.count(4) + finger_list_1.count(5))/5):
                command_finger_1 = "f 1 OPEN"
            elif(finger_list_1.count(0)/5 >= ratio or finger_list_1.count(1)/5 >= ratio or (finger_list_1.count(0) + finger_list_1.count(1))/5):
                command_finger_1 = "f 1 CLOSE"
            else: command_finger_1 = "f 1 IDDLE"
            
            # Creation finger command for hand 2
            if(finger_list_2.count(5)/5 >= ratio or finger_list_2.count(4)/5 >= ratio or (finger_list_2.count(4) + finger_list_2.count(5))/5):
                command_finger_2 = "f 2 OPEN"
            elif(finger_list_2.count(0)/5 >= ratio or finger_list_2.count(1)/5 >= ratio or (finger_list_2.count(0) + finger_list_2.count(1))/5):
                command_finger_2 = "f 2 CLOSE"
            else: command_finger_2 = "f 2 IDDLE"
                    
        else:  # If only one hand is on the screen (i.e. at least 1 element in finger_list_2 is -1)
            # Evaluate central points for both hand
            central_point_1 = centralPointInBox(boxes_predict[0])
            
            # Create the command with the position of the central point
            command_hand_position_1 = "h 1 " +  str(central_point_1[0]) + " " + str(central_point_1[1])
            command_hand_position_2 = "h 2 " +  str(-1) + " " + str(-1)
            
            # Ration of finger counter necessary to obtain hand close or open
            ratio = 3/5
            
            # Creation finger command for hand 1
            if(finger_list_1.count(5)/5 >= ratio or finger_list_1.count(4)/5 >= ratio or (finger_list_1.count(4) + finger_list_1.count(5))/5):
                command_finger_1 = "f 1 OPEN"
            elif(finger_list_1.count(0)/5 >= ratio or finger_list_1.count(1)/5 >= ratio or (finger_list_1.count(0) + finger_list_1.count(1))/5):
                command_finger_1 = "f 1 CLOSE"
            else: command_finger_1 = "f 1 IDDLE"
            
            # Creation finger command for hand 2
            command_finger_2 = "f 2 IDDLE"
            
    else:
         # Create the command with the position of the central point
         command_hand_position_1 = "h 1 "  +  str(-1) + " " + str(-1)
         command_hand_position_2 = "h 2 "  +  str(-1) + " " + str(-1)
         command_finger_1 = "f 1 IDDLE"
         command_finger_2 = "f 2 IDDLE"
        
    # Send command
    sendCommand(sock, UDP_IP, UDP_PORT, command_hand_position_1, debug_var = False)
    sendCommand(sock, UDP_IP, UDP_PORT, command_hand_position_2, debug_var = False)
    sendCommand(sock, UDP_IP, UDP_PORT, command_finger_1, debug_var = False)
    sendCommand(sock, UDP_IP, UDP_PORT, command_finger_2, debug_var = False)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Advance counter
    counter += 1

capture.release()
cv2.destroyAllWindows()
