"""
Our current implementation for converting grayscale to RGB images. 
This script incorporates ResNet-6 (https://arxiv.org/abs/1512.03385)
as well as AMSGrad (https://arxiv.org/abs/1904.09237) to achieve 
optimal image-to-image translation.

Based off of:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Repurposed by Badr Belhiti
"""

# Imports
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
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Model name
model_name = "third_stage"
arch_name = "ResNet_6"

# Root directory for dataset
grayscale_root = "../data/CelebA/grayscale_rgb/large/grayscale/data/"
rgb_root = "../data/CelebA/grayscale_rgb/large/rgb/data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 1

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc_in = 1
nc_out = 3

# Mapping weight (cGAN lambda)
mapping_weight = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

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

model_dir += arch_name + "/"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Custom dataset that loads corresponding grayscale and RGB images together
class GrayscaleToRgbDataset(Dataset):
    def __init__(self, grayscale_dir, rgb_dir, transform=None):
        """
        Initialize layout for GrayscaleToRgbDataset. Keeps track of associated mapping
        grayscale_dir - Path to directory containing grayscale images
        rgb_dir - Path to directory containing RGB images
        transform (optional) - Any PyTorch transforms that should be applied to the images
        """
        assert len(os.listdir(grayscale_dir)) == len(os.listdir(rgb_dir))
        self.grayscale = grayscale_dir
        self.rgb = rgb_dir
        self.transform = transform
        self.length = len(os.listdir(grayscale_dir))

    def __len__(self):
        """
        Return number of grayscale files
        """
        return self.length

    def __getitem__(self, idx):
        """
        Get pair of grayscale and RGB images associated with idx
        idx: index of grayscale and RGB image files in respective directories
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        grayscale_image = Image.open(os.path.join(self.grayscale, str(idx) + ".jpg"))
        rgb_image = Image.open(os.path.join(self.rgb, str(idx) + ".jpg"))

        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            rgb_image = self.transform(rgb_image)

        grayscale_image = transforms.Grayscale()(grayscale_image)

        return {"grayscale": grayscale_image, "rgb": rgb_image}


transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts image into tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalizes pixel values to [0, 1]
    ]
)

# Use custom dataset to load in conditional input and desired output
dataset = GrayscaleToRgbDataset(grayscale_root, rgb_root, transform)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print("Training on:", device)


def weights_init(m):
    """Function to initialize weights using a normal distribution of some mean and variance"""

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ResNet-6 image-to-image implementation
class Generator(nn.Module):
    def __init__(self, ngpu):
        """
        Define architecture for grayscale to RGB generator
        ngpu - Number of gpu's to train model on
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Initial convolutional layer that preserves resolution and creates 'ngf' output channels
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc_in, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # Down sampling layers that take input from 128x128 to 32x32 while progressively adding channels
        n_downsampling = 2
        for i in range(n_downsampling):
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

        # Propagate signal through 6 ResNet blocks, preserving height and width
        mult = 2 ** n_downsampling
        for i in range(6):

            model += [ResnetBlock(ngf * mult)]

        # Upsample signal back to 128x128
        for i in range(n_downsampling):
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

        # Preserve height and width while decreasing channel depth to 1
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, nc_out, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        Conduct single forward pass
        input - 128x128x1 grayscale image
        """
        return self.model(input)


# ResNet block that uses reflection padding and instance normalization
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        """
        Defining the Resnet Block architecture
        dim - Input and output dimensions of block. This block doesn't change dimensionality.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    # An arbitrary convolutional block that maintains dimensionality
    def build_conv_block(self, dim):
        """
        Construct convolutional block for ResNet architecture
        dim - Input and output dimensions of block.
        """
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

    # ResNet architecture relies on "skip-connections" to achieve superior results
    def forward(self, x):
        """
        Conduct forward pass
        """
        return x + self.conv_block(x)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netG.apply(weights_init)

# Define 5 layer discriminator with batch normalization
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        """
        Defines discriminator architecture for grayscale to RGB stage
        ngpu - Number of gpu's to train model on
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        kw = 4
        padw = 1
        # Conditional discriminator takes in two images concatenated on channel dimension
        # Expand input to 'ndf' initial output channels
        sequence = [
            nn.Conv2d(nc_in + nc_out, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        # Convolutional downsampling with stride 2 and doubling of channels every layer
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # Convolutional downsampling with stride 1 and doubling of channels every layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Transform signal to have channel depth of 1
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        # Run signal through sigmoid to get range of (0, 1)
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Conducts forward pass
        input - Grayscale and RGB images concatenated on channel axis
        """
        grayscale, rgb = input
        return self.model(torch.cat([grayscale, rgb], dim=1))


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize loss functions
criterion = nn.BCELoss()
conditional_criterion = nn.L1Loss()

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, 0.999), amsgrad=True)
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, 0.999), amsgrad=True)


def save_checkpoint(checkpoint_name):
    """
    Function to save current GAN as a checkpoint
    checkpoint_name - Name of directory to save models in
    """

    save_dir = "%s%s/" % (model_dir, checkpoint_name)
    model_G = netG.module if device.type == "cuda" else netG
    model_D = netD.module if device.type == "cuda" else netD

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model_G.state_dict(), save_dir + "netG.pth")
    torch.save(model_D.state_dict(), save_dir + "netD.pth")


# Training Loop

# Lists to keep track of progress
last_n = 10000
running_G_loss = 0
running_D_loss = 0
patience = 50000

img_list = []
G_losses = []
G_adv_losses = []
D_losses = []
best_total_losses = []
best_total_loss = 200 * last_n
time_since_best = 0
iters = 0


def save_loss():
    """
    Function that saves the current loss graphs for the training
    """

    # Adversarial losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Adversarial Loss During Training")
    plt.plot(G_adv_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_dir + "adversarial_loss.png")

    # Generator losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator Loss During Training")
    plt.plot(G_losses, label="G", alpha=0.5)
    plt.plot(best_total_losses, label="Best")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_dir + "generator_loss.png")


def should_continue(G_losses, D_losses):
    """
    Function that returns true if losses are "acceptable", false otherwise
    G_losses - Array of past generator losses
    D_losses - Array of past discriminator losses
    """

    global running_G_loss
    global running_D_loss
    global best_total_loss
    global time_since_best

    # Not enough loss information to stop training
    if len(G_losses) < last_n:
        return True

    if len(G_losses) == last_n:
        running_G_loss = sum(G_losses)
        running_D_loss = sum(D_losses)
        best_total_loss = running_G_loss + running_D_loss
        return True

    # Include current loss
    running_G_loss -= G_losses[-last_n]
    running_G_loss += G_losses[-1]

    running_D_loss -= D_losses[-last_n]
    running_D_loss += D_losses[-1]

    running_total_loss = running_G_loss + running_D_loss

    best_total_losses.append(best_total_loss / last_n)

    # If average loss does not improve over 'patience' number of iterations, stop training
    if running_total_loss > best_total_loss and time_since_best > patience:
        return False

    time_since_best += 1

    # Update best loss if model achieves a better running average loss over the past 'last_n' iterations
    if running_total_loss < best_total_loss:
        best_total_loss = running_total_loss
        time_since_best = 0
        print("New best %f" % (best_total_loss / last_n))

    return True


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
        real_grayscale = data["grayscale"].to(device)
        real_rgb = data["rgb"].to(device)

        b_size = real_grayscale.size(0)
        label = torch.full(
            (b_size, 1, 14, 14), real_label, dtype=torch.float, device=device
        )

        # Forward pass real batch through D
        output = netD((real_grayscale, real_rgb))

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Edges
        conditional_input = real_grayscale

        # Generate fake image batch with G
        fake = netG(conditional_input)
        label = torch.full(
            (b_size, 1, 14, 14), fake_label, dtype=torch.float, device=device
        )

        # Classify all fake batch with D
        output = netD((real_grayscale, fake.detach()))

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
        label = torch.full(
            (b_size, 1, 14, 14), real_label, dtype=torch.float, device=device
        )  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD((real_grayscale, fake))

        # Calculate G's realism loss
        errG_realism = criterion(output, label)

        # Calculate G's mapping loss
        errG_mapping = conditional_criterion(fake, data["rgb"].to(device))

        # Calculate G's total loss
        errG = errG_realism + mapping_weight * errG_mapping

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        # Output training stats
        if iters % 1000 == 0 and iters != 0:
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: (%.4f, %.4f)\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch + 1,
                    num_epochs,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG_realism.item(),
                    mapping_weight * errG_mapping,
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

        G_losses.append(errG.item())
        G_adv_losses.append(errG_realism.item())
        D_losses.append(errD.item())

        # Safeguard to halt training if loss plateaus
        if not should_continue(G_losses, D_losses):
            print("Early stopping due to undesirable loss...")
            save_loss()
            save_checkpoint("early_stop-%d" % iters)
            exit(0)

        iters += 1

    # Save generator and discriminator after every epoch
    save_checkpoint("epoch_%d" % epoch)


print("Finished training. Saving results...")

# Save current loss graphs
save_loss()

# Save model
save_checkpoint("final")
