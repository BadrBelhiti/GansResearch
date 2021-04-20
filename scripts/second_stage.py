from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Model name
model_name = 'second_stage'

# Root directory for dataset
edges_root = "../data/CelebA/edges_grayscale/large/edges/data/"
grayscale_root = "../data/CelebA/edges_grayscale/large/grayscale/data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Mapping weight (cGAN lambda)
mapping_weight = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Size of feature maps in conditional input layers
ncf = 32

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
g_lr = 2e-4

d_lr = 2e-4

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4

# Initialize model working directory
if not os.path.exists('../models/'):
    os.mkdir('../models/')

model_dir = '../models/' + model_name + '/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_dir += str(num_epochs) + '/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

class EdgesToGrayscaleDataset(Dataset):

    def __init__(self, edges_dir, grayscale_dir, transform=None):
        assert len(os.listdir(edges_dir)) == len(os.listdir(grayscale_dir))
        self.edges = edges_dir
        self.grayscale = grayscale_dir
        self.transform = transform
        self.length = len(os.listdir(edges_dir))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        edges_image = Image.open(os.path.join(self.edges, str(idx) + '.jpg'))
        grayscale_image = Image.open(os.path.join(self.grayscale, str(idx) + '.jpg'))

        if self.transform:
            edges_image = self.transform(edges_image)
            grayscale_image = self.transform(grayscale_image)

        return {'edges': edges_image, 'grayscale': grayscale_image}

transform = transforms.Compose([    
                transforms.Grayscale(),   # Converts image into single channel
                transforms.ToTensor(),    # Converts image into tensor
                transforms.Normalize((0.5,),(0.5,)) # Normalizes pixel values to [0, 1]
                ])

# Use custom dataset to load in conditional input and desired output
dataset = EdgesToGrayscaleDataset(edges_root, grayscale_root, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print('Training on:', device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Convolutional layers before latent vector is concatenated
        self.conditional = nn.Sequential(
            nn.Conv2d(nc, ncf, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ncf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ncf, ncf * 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ncf * 2, ncf * 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ncf * 4, ncf * 8, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ncf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ncf * 8, ncf * 16, 5, 1, 2, bias=False),
            nn.ReLU(True)
        )

        # Beginning of actual generator
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ncf * 16, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, conditional_input):
        cond = self.conditional(conditional_input)
        print(cond.shape)
        # return self.main(cond)
        return cond

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
# print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        edges, grayscale = input
        # Concatenate images on channel dimension
        new_input = torch.cat((edges, grayscale), dim=1)
        return self.main(new_input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)

# Initialize loss functions
criterion = nn.BCELoss()
conditional_criterion = nn.L1Loss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()

        # Format batch
        real_edges = data['edges'].to(device)
        real_grayscale = data['grayscale'].to(device)
        
        # Skip iterations with incomplete batch
        if (len(real_grayscale) < batch_size):
            continue

        b_size = real_grayscale.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD((real_edges, real_grayscale)).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Edges
        conditional_input = real_edges

        # Generate fake image batch with G
        fake = netG(conditional_input)
        print(fake.shape)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = netD((real_edges, fake.detach())).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD((real_edges, fake)).view(-1)

        # Calculate G's realism loss
        errG_realism = criterion(output, label)

        # Calculate G's mapping loss
        errG_mapping = conditional_criterion(fake, data['grayscale'].to(device))

        # Calculate G's total loss
        errG = errG_realism + mapping_weight * errG_mapping

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()
        

        # Output training stats
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

    # Save generated sample from each epoch
    # fake_image = netG((torch.randn(16, nz, 1, 1, device=device), dataset[0]['edges']))
    # img_list.append(np.transpose(vutils.make_grid(fake_image, padding = 5, normalize=True), (1, 2, 0)))

print('Finished training. Saving results...')

plt.figure(figsize=(10,5))
plt.title('Generator and Discriminator Loss During Training')
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(model_dir + 'loss.png')

# Save model
if device.type == 'cuda':
    netG = netG.module
torch.save(netG.state_dict(), model_dir + 'model.pth')
