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
from torchsummary import summary

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Model name
model_name = "second_stage"

# Root directory for dataset
edges_root = "../data/CelebA/edges_grayscale/large/edges/data/"
grayscale_root = "../data/CelebA/edges_grayscale/large/grayscale/data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 1

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
ndf = 64

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
if not os.path.exists("../models/"):
    os.mkdir("../models/")

model_dir = "../models/" + model_name + "/"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_dir += str(num_epochs) + "/"

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

        edges_image = Image.open(os.path.join(self.edges, str(idx) + ".jpg"))
        grayscale_image = Image.open(os.path.join(self.grayscale, str(idx) + ".jpg"))

        if self.transform:
            edges_image = self.transform(edges_image)
            grayscale_image = self.transform(grayscale_image)

        return {"edges": edges_image, "grayscale": grayscale_image}


transform = transforms.Compose(
    [
        transforms.Grayscale(),  # Converts image into single channel
        transforms.ToTensor(),  # Converts image into tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalizes pixel values to [0, 1]
    ]
)

# Use custom dataset to load in conditional input and desired output
dataset = EdgesToGrayscaleDataset(edges_root, grayscale_root, transform)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print("Training on:", device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(9):  # ResNet-9 requires 9 blocks

            model += [
                ResnetBlock(ngf * mult)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # Skip connection
        return x + self.conv_block(x)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(2 * nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        edges, grayscale = input
        return self.model(torch.cat([edges, grayscale], dim=1))


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize loss functions
criterion = nn.BCELoss()
conditional_criterion = nn.L1Loss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

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
        real_edges = data["edges"].to(device)
        real_grayscale = data["grayscale"].to(device)

        # Skip iterations with incomplete batch
        if len(real_grayscale) < batch_size:
            continue

        b_size = real_grayscale.size(0)
        label = torch.full((b_size, 1, 14, 14), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD((real_edges, real_grayscale))

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Edges
        conditional_input = real_edges

        # Generate fake image batch with G
        fake = netG(conditional_input)
        label = torch.full((b_size, 1, 14, 14), fake_label, dtype=torch.float, device=device)

        # Classify all fake batch with D
        output = netD((real_edges, fake.detach()))

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
        label = torch.full((b_size, 1, 14, 14), real_label, dtype=torch.float, device=device)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD((real_edges, fake))

        # Calculate G's realism loss
        errG_realism = criterion(output, label)

        # Calculate G's mapping loss
        errG_mapping = conditional_criterion(fake, data["grayscale"].to(device))

        # Calculate G's total loss
        errG = errG_realism + mapping_weight * errG_mapping

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        # Output training stats
        if iters % 1000 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch + 1,
                    num_epochs,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        iters += 1

    # Save generated sample from each epoch
    # fake_image = netG((torch.randn(16, nz, 1, 1, device=device), dataset[0]['edges']))
    # img_list.append(np.transpose(vutils.make_grid(fake_image, padding = 5, normalize=True), (1, 2, 0)))

print("Finished training. Saving results...")

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(model_dir + "loss.png")

# Save model
if device.type == "cuda":
    netG = netG.module
torch.save(netG.state_dict(), model_dir + "model.pth")
