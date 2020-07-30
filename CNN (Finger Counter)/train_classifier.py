#%% Imports
from hand_classification_net import handNet, handNet2, trainEpoch, loadDataset, saveObject, loadObject
from HandClassificationDataset import HandClassificationDataset, get_transform

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import random

# torch.cuda.empty_cache()

#%%
reduce_size = (140, 140)

learning_rate = 0.005
n_epoch = 250

#%%

dataset_train = HandClassificationDataset('Train', transforms = get_transform(False))
dataset_test = HandClassificationDataset('Test', transforms = get_transform(False))

b1, b2 = dataset_train[4]

dataset_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)
# dataset_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)


#%% Training setup

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = handNet2()

net.train() 
net.to(device)

# optimizer = optim.Adam(nn.ParameterList(net.parameters()), lr = learning_rate, weight_decay = 1e-2)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters())

# loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

# net.computeVisualWeigth()
# net.to(device)

# a = torch.cuda.memory_summary()

#%% Training

test_loss_log = []
train_loss_log = []
train_loss = 0

for epoch in range(n_epoch):
    for local_batch, local_labels in dataset_loader_train:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.long().to(device)
        
        train_loss += trainEpoch(net, [local_batch, local_labels], optimizer, loss_fn)
        
    train_loss_log.append(float(train_loss))
    if(epoch % 10 == 0): print('EPOCH: ', epoch, ' - Train loss: ', train_loss)
    
    if(train_loss < 1e0-5): break
    
    train_loss = 0
           
#%%

# Plot losses
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
# plt.semilogy(test_loss_log, label='Test loss')
# plt.semilogy(np.ones(len(test_loss_log)) * 0.8, label = '0.8 line')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
# plt.xlim(0, 75)
# plt.ylim(0.65, 10)
plt.grid()
plt.legend()
plt.tight_layout()
# plt.savefig("Plot/test_train_error_" + characteristic + "_" + str(ind) + "_" + str(learning_rate) + ".png", bbox_inches  = 'tight')
plt.show()
