# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:33:45 2021

@author: mpatt
"""


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F


class CNN(nn.Module):
	
	def __init__(self):
		super(CNN, self).__init__()
		self.main = nn.Sequential(
			
			# input is Z, going into a convolution
			nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),
			
			nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),

			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),

			nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Dropout2d(p=0.2),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
		)
			
		self.classifier = nn.Sequential(
			nn.Linear(128, 10),
		)
		
	def forward(self, x):   
		x = self.main(x)
		x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
		logits = self.classifier(x)
		return logits, x