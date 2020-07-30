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

#%% Dataset creation (and check)

from EgoDataset import EgoDataset, get_transform, get_transform_2

path = "Train"

dataset_train = EgoDataset(path, n_elements = 450, transforms = get_transform(False))
dataset_test = EgoDataset(path, n_elements = 70, transforms = get_transform(False))

# Since the large number of elements I randomly choose a subset of the folder of the dataset. So I use the same list of folder for train and test


#%% Check dataset

n_example = -3
for i in range(0, n_example): dataset_train.checkBoxes(i)

# a, b = dataset_train[4]
# a = a.squeeze().numpy()

# cv2.imshow("test", a)
# dataset_train.checkBoxes(4)


#%% Define model (get from torchvision modelzoo)
model = get_model()

#%% Test forward (OPTIONAL)

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

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset_train, batch_size = 2, shuffle = True, num_workers=4, collate_fn=utils.collate_fn)


# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 20

#%% Train the model

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset

print("That's it!")


#%% Test the model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

idx_test = [random.randint(0, 69) for iter in range(15)]
# idx = np.unique(idx)
idx = np.unique(idx_test)

# model.load_state_dict(torch.load("model_only_weight.pt"))
model.to(device)
model.eval()

for idx in idx_test:
    img, target = dataset_test[idx]
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    print(img.is_cuda)
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
    
    cv2.imshow("TEST PREDICTIONS", img_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
