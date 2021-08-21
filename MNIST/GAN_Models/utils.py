# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:20:57 2021

@author: mpatt
"""

#Import the libraries 
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np
import matplotlib.pyplot as plt




def imshow(imgs):
    imgs = torchvision.utils.make_grid(imgs) # makes grid of the tensor images passed
    npimgs = imgs.numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()