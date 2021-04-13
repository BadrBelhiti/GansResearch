from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
data_root = "../data/CelebA/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4

# Output directory
output_dir = '../data/CelebA/tensors/data/'

# Initialize output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

transform = transforms.Compose([    
                transforms.CenterCrop(178),
                transforms.Resize(128),   
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ])
trainset = ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

iters = 0

print("Processing images...")

# Iterate through each batch
for i, data in enumerate(dataloader, 0):
    
    # Iterate through each sample
    for sample in data[0]:
        torch.save(sample, output_dir + str(iters) + '.pt')
        iters += 1

        if iters % 100 == 0:
            print('Built %d tensors' % iters)

