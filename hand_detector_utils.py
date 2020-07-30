# -*- coding: utf-8 -*-
"""
File containing function for the hand detector. There are some method

@author: Alberto Zancanaro (Jesus)
"""
#%%

import numpy as np
import cv2
import math
import socket
import time
import torch

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    
def sendCommand(sock, UDP_IP, UDP_PORT, command, debug_var = True):
    sock.sendto((command).encode(), (UDP_IP, UDP_PORT) )
    
    if(debug_var): print("_"*10, command, " sent!", "_"*10)
        
def getBiggestBox(matrix_boxes_predicted):
    """
    Return the biggest or the two biggest boxes.
    
    """
    
    tmp_mat = np.zeros((2, 4))
    
    vet_area = np.zeros(matrix_boxes_predicted.shape[0])
    
    for i in range(len(vet_area)):
        vet_area[i] = (matrix_boxes_predicted[i, 2] - matrix_boxes_predicted[i, 0]) * matrix_boxes_predicted[i, 3] - matrix_boxes_predicted[i, 1]
        
    # Get the index of the two biggest box
    ind = vet_area.argsort()[-2:][::-1]
    
    tmp_mat[:, :] = matrix_boxes_predicted[ind, :]
    
    if(checkIfInside(tmp_mat[0,:], tmp_mat[1,:])): # If one rect is inside another I return the biggest one
        # print(tmp_mat)
        # print("argmax: ", np.argmax(vet_area))
        # print("vet_area: ", vet_area)
        # print(matrix_boxes_predicted[np.argmax(vet_area), :])
        tmp_mat = np.zeros((1, 4))
        
        # tmp_mat[0, :] = matrix_boxes_predicted[np.argmax(vet_area), :]
        tmp_mat[0, :] = matrix_boxes_predicted[ind[0], :]
        return tmp_mat.astype(int)
    else: 
        # print("tmp_mat: ", tmp_mat)
        return tmp_mat.astype(int) # Otherwise I return both

def checkIfInside(rect1, rect2):
    """
    Check if rect2 is inside rect1. The two rect are specified by a vector of 4 elements. 
    The first two are the upper left corner and the last two are the down right corner
    """
    
    # Check if the upper left corner of rect2 is inside rect1
    if(rect2[0] >= rect1[0] and rect2[0] <= rect1[2]):
        if(rect2[1] >= rect1[1] and rect2[1] <= rect1[3]):
            return True
        
    # Check if the upper right corner of rect2 is inside rect1
    if(rect2[2] >= rect1[0] and rect2[2] <= rect1[2]):
        if(rect2[1] >= rect1[1] and rect2[1] <= rect1[3]):
            return True
        
    # Check if the down left corner of rect2 is inside rect1
    if(rect2[0] >= rect1[0] and rect2[0] <= rect1[2]):
        if(rect2[3] >= rect1[1] and rect2[3] <= rect1[3]):
            return True
        
    # Check if the down right corner of rect2 is inside rect1
    if(rect2[2] >= rect1[0] and rect2[2] <= rect1[2]):
        if(rect2[3] >= rect1[1] and rect2[3] <= rect1[3]):
            return True
        
        
    return False
    

def trackingHandWithRCNN(model, img, device):
    img = torch.from_numpy(img).float().to(device)
    img = img / 255
    img = img.permute(2, 0, 1)
    img = img.to(device)
    predictions = model([img])
    
    boxes_predict = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    # print(boxes_predict)
    
    if(boxes_predict.shape[0] >= 2):
        return getBiggestBox(boxes_predict)
    else: return boxes_predict
    
    
def centralPointInBox(box):
    x = int(box[0] + (box[2] - box[0])/2)
    y = int(box[1] + (box[3] - box[1])/2)
    
    return (x, y)

def predictFingers(model, img, device, model_input_size = (140, 140)):
    # Resize to the necessary input size
    hand = cv2.resize(img, model_input_size)
    
    # Convert the image in a Pytorch tensor, normalize, swap axis and move to GPU (if present)
    hand_tensor = torch.from_numpy(hand).float().to(device)
    hand_tensor = hand_tensor / 255
    hand_tensor = hand_tensor.permute(2, 0, 1)
    hand_tensor.to(device)
    
    # Predict number of fingers and convert result in a numpy array
    finger_predict = model(hand_tensor.unsqueeze(0))
    finger_predict = np.argmax(finger_predict.cpu().detach().numpy())
    
    # cv2.putText(frame, str(finger_predict), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
    # print(finger_predict)
    
    return finger_predict

