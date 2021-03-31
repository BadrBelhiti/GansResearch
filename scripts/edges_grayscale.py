import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters
from matplotlib import cm

input_directory = '../data/CelebA/celeba/'
output_edges_directory = '../data/CelebA/edges_grayscale/large/edges/data/'
output_grayscale_directory = '../data/CelebA/edges_grayscale/large/grayscale/data/'

celeba = os.listdir(os.fsencode(input_directory))

if not os.path.exists(output_edges_directory):
    os.makedirs(output_edges_directory)

if not os.path.exists(output_grayscale_directory):
    os.makedirs(output_grayscale_directory)

def edges_grayscale_rgb(file_path):
    # Read in image as RGB array
    rgb = np.array(Image.open(file_path))
    
    # Convert RGB array into grayscale array
    grayscale = rgb[:,:,0]
    
    # Extract edges from grayscale array
    edges = filters.sobel(grayscale)
    
    return (edges, grayscale, rgb)

celeba_crop = (0, 20, 178, 218 - 20)
resolution = (128, 128)

def build_samples(count=0):
    built = 0
    
    # Iterate through each sample
    for file in celeba:
        filename = os.fsdecode(file)
    
        # First get edges construction
        edges, grayscale, rgb = edges_grayscale_rgb(input_directory + filename)
        edges = Image.fromarray(edges)
        grayscale = Image.fromarray(grayscale)
    
        # Then scale down edges to desired dimensions
        
        # Crop image to square to maintain aspect ratio
        edges_cropped = edges.crop(celeba_crop)
        grayscale_cropped = grayscale.crop(celeba_crop)
        
        # Resample to desired resolution
        edges_resampled = edges_cropped.resize(resolution)
        grayscale_resampled = grayscale_cropped.resize(resolution)
        
        plt.imsave(output_edges_directory + str(built) + '.jpg', np.asarray(edges_resampled), cmap='gray')
        plt.imsave(output_grayscale_directory + str(built) + '.jpg', np.asarray(grayscale_resampled), cmap='gray')
        
        built += 1

        if built % 1000 == 0:
            print('Generated %s images' % built)

        # If 'count' is 0, then whole dataset will be generated.
        if not count == 0 and built == count:
            break;

# Run code
build_samples(1000)
