"""
Script that builds the grayscale->rgb dataset used
by our grayscale2rgb conditional GAN. It takes in raw
CelebA images and converts them into their respective
grayscale and rgb representations and saves the results
in corresponding subdirectories.

Written by Badr Belhiti
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters
from matplotlib import cm

# Specify working directories
input_directory = "../data/CelebA/mini_celeba/"
output_grayscale_directory = "../data/CelebA/grayscale_rgb/large/grayscale/data/"
output_rgb_directory = "../data/CelebA/grayscale_rgb/large/rgb/data/"

# Initialize directories
celeba = os.listdir(os.fsencode(input_directory))

if not os.path.exists(output_grayscale_directory):
    os.makedirs(output_grayscale_directory)

if not os.path.exists(output_rgb_directory):
    os.makedirs(output_rgb_directory)


def edges_grayscale_rgb(file_path):
    """
    Function that takes in path to CelebA image and outputs corresponding edges, grayscale, and rgb representation
    file_path - Path to raw CelebA image
    """

    # Read in image as RGB array
    rgb = np.array(Image.open(file_path))

    # Convert RGB array into grayscale array
    grayscale = rgb[:, :, 0]

    # Extract edges from grayscale array
    edges = filters.sobel(grayscale)

    return (edges, grayscale, rgb)


# Specify crop and downsampling parameters
celeba_crop = (0, 20, 178, 218 - 20)
resolution = (128, 128)


def build_samples(count=0):
    """
    Function that generates 'count' number of samples to generate. If count=0, build entire dataset
    """

    built = 0

    # Iterate through each sample
    for file in celeba:
        filename = os.fsdecode(file)

        # First get edges construction
        edges, grayscale, rgb = edges_grayscale_rgb(input_directory + filename)
        grayscale = Image.fromarray(grayscale)
        rgb = Image.fromarray(rgb)

        # Then scale down edges to desired dimensions

        # Crop image to square to maintain aspect ratio
        grayscale_cropped = grayscale.crop(celeba_crop)
        rgb_cropped = rgb.crop(celeba_crop)

        # Resample to desired resolution
        grayscale_resampled = grayscale_cropped.resize(resolution)
        rgb_resampled = rgb_cropped.resize(resolution)

        plt.imsave(
            output_grayscale_directory + str(built) + ".jpg",
            np.asarray(grayscale_resampled),
            cmap="gray",
        )
        plt.imsave(
            output_rgb_directory + str(built) + ".jpg",
            np.asarray(rgb_resampled),
            cmap="gray",
        )

        built += 1

        if built % 1000 == 0:
            print("Generated %s images" % built)

        # If 'count' is 0, then whole dataset will be generated.
        if not count == 0 and built == count:
            break


# Run code
build_samples()
