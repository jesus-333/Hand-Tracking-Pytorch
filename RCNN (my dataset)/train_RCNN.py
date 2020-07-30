# -*- coding: utf-8 -*-
"""
Script to train the RCNN. 

Based on the script and tutorial on the Pytorch website (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

N.B. This script use a netwtork that require torchvision. All the this stuff is inside the folder vision.
     You can aslo download eveythong from the github of Pytorch.
     To some unknown reasons some imports doesn't work well in the original file so I edited something here and there (ONLY ON IMPORT LEVEL)

@author: Alberto Zancanaro (Jesus)
"""

#%%

import torch
import torchvision.transforms as transforms

from vision.references.detection import utils
from vision.references.detection.engine import train_one_epoch, evaluate

import numpy as np
import cv2
import random

from support_function import get_model

import matplotlib.pyplot as plt

#%% Dataset creation

from MyDataset import MyDataset, get_transform, get_transform_2

dataset_train = MyDataset(path = "Train", n_elements = 450, shuffle = True, transforms = get_transform(False))
dataset_test = MyDataset(path = "Test", n_elements = 23, transforms = get_transform(False))


#%% Check dataset

n_example = 3
for i in range(0, n_example): dataset_train.checkBoxes(i)

# Check dataset
# dataset_train.checkBoxes(4)


# Check for the boxes coordinates (UNCOMMENT)
# for i in range(len(dataset_train)):
#     img, target = dataset_train[i]
#     boxes = dataset_train.last_boxes
    
#     print("Element: ", i)
    
#     for row in boxes:
#         diff_x = row[2] - row[0]
#         diff_y = row[3] - row[1]
#         print("  ", row)
#         print("   x1 - x2: ", diff_x)
#         print("   y1 - y2: ", diff_y)
        
        
#         if(diff_x <= 0 or diff_y <= 0): break
#         else: print("ALL OK")
    
#     print("- - - - - - - - - - - - - - - - - - - - - - - -")
    
#     if(diff_x <= 0 or diff_y <= 0): 
#         print("ERROR")
#         break


#%% Define model (get from orchvision modelzoo)
model = get_model()

#%% Test forward (OPTION)

if(True):
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size = 2, shuffle = True, num_workers=4, collate_fn=utils.collate_fn)
    
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images,targets)   # Returns losses and detections
    
    # # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x) 
    
    print("NO EXECUTION ERROR")

#%% Training settings
    
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Cuda is available: ", torch.cuda.is_available())

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset_train, batch_size = 2, shuffle = True, num_workers=4, collate_fn=utils.collate_fn)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

# let's train it for 10 epochs
num_epochs = 50

#%%

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 10)
    # update the learning rate
    lr_scheduler.step()

    # Uncomment if you want to save your model every 10 epocj
    # if(epoch % 10 == 0): torch.save(model, "model_all_" + str(epoch) +".pth")

print("That's it!")
torch.save(model, "model_all_" + str(num_epochs) +".pth")

#%% Test on the dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

idx_test = [random.randint(0, 16) for iter in range(8)]
# idx_test = np.linspace(0, 15, 16)
# idx = np.unique(idx)
idx = np.unique(idx_test)

# model.load_state_dict(torch.load("model_only_weight.pt"))
model = torch.load("model_all_20.pth")
model.to(device)
model.eval()


for idx in range(18, 23):
    img, target = dataset_test[idx]
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    
    # print(img.is_cuda)
    print(img.shape)
    
    predictions = model(img)
    
    boxes_predict = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    
    
    dataset_test.get_image_PIL = True
    img_opencv = dataset_test[idx][0]
    img_opencv = cv2.cvtColor(np.array(img_opencv), cv2.COLOR_RGB2BGR)
    dataset_test.get_image_PIL = False
    
    for line in boxes_predict:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        
        cv2.rectangle(img_opencv, pt1, pt2, (255, 0, 0), thickness = 1)
        
        print("pt1: ", pt1, " - pt2: ", pt2)
    print("- - - - - ")
    
    cv2.imshow("TEST PREDICTIONS " + str(idx), img_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#%%
img_1, target = dataset_test[16]
img_1 = torch.unsqueeze(img_1, 0)
img_1 = img.to(device)

# print(img.is_cuda)
print(img_1.shape)

predictions = model(img_1)

boxes_predict = predictions[0]['boxes'].cpu().detach().numpy().astype(int)


dataset_test.get_image_PIL = True
img_opencv = dataset_test[16][0]
img_opencv = cv2.cvtColor(np.array(img_opencv), cv2.COLOR_RGB2BGR)
dataset_test.get_image_PIL = False

for line in boxes_predict:
    pt1 = (line[0], line[1])
    pt2 = (line[2], line[3])
    
    cv2.rectangle(img_opencv, pt1, pt2, (255, 0, 0), thickness = 1)
    
    print("pt1: ", pt1, " - pt2: ", pt2)
print("- - - - - ")

cv2.imshow("TEST PREDICTIONS", img_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()