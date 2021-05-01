from __future__ import print_function
# %matplotlib inline
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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import normal

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Model name
model_name = 'baseline_dcgan'
arch_name = 'noisy_disc'

# Root directory for dataset
data_root = "../data/CelebA/training_edges/large/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = ngf // 4

# Number of training epochs
num_epochs = 20

# Learning rate for optimizers
g_lr = 5e-5

d_lr = 5e-5

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

model_dir += arch_name + '/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

transform = transforms.Compose([
    transforms.Grayscale(),   # Make sure to change the parameters accordingly
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

print('Training on:', device)

# Function to initialize weights using a normal distribution of some mean and variance


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
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
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

    def forward(self, input):
        return self.main(input)


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
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=d_lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=g_lr)

# Function to save current GAN as a checkpoint


def save_checkpoint(checkpoint_name):
    save_dir = '%s%s/' % (model_dir, checkpoint_name)
    model_G = netG.module if device.type == 'cuda' else netG
    model_D = netD.module if device.type == 'cuda' else netD

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model_G.state_dict(), save_dir + 'netG.pth')
    torch.save(model_D.state_dict(), save_dir + 'netD.pth')


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

        D_loss = None

        ### WGAN 5 iterations for critic ###
        for _ in range(5):
            # Reset critic gradients
            netD.zero_grad()

            ### Forward pass through critic using real images ###

            # Get real images
            real = data[0].to(device)

            # Forward pass real images through critic and flattern result
            output_real = netD(real).view(-1)

            ### Forward pass through critic using fake images ###

            # Get fake images
            fake = netG(torch.randn(batch_size, nz, 1,
                        1, device=device)).to(device)

            # Forward pass fake images through critic and flattern result
            output_fake = netD(fake).view(-1)

            ### Calculate loss and update critic ###

            # Critic loss
            D_loss = -(torch.mean(output_real) - torch.mean(output_fake))

            # Calculate critic gradients
            D_loss.backward()

            # Update critic weights
            optimizerD.step()

            # Clip weights
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

        ### Update generator ###

        # Get fake images
        fake = netG(torch.randn(batch_size, nz, 1,
                    1, device=device)).to(device)

        # Forward pass fake images through critic and flattern result
        output_fake = netD(fake).view(-1)

        # Calculate generator loss
        G_loss = -(torch.mean(output_fake))

        # Calculate generator gradients
        G_loss.backward()

        # Update weights
        optimizerG.step()

        # Output training stats
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, i, len(dataloader), D_loss, G_loss))

        G_losses.append(G_loss)
        D_losses.append(D_loss)

        iters += 1

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0 and epoch != 0:
        save_checkpoint('%d-epochs' % epoch)

# Plot and save loss data
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(model_dir + 'loss.png')

# Save model
save_checkpoint('final-%d' % iters)
