import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters
from matplotlib import cm

input_directory = '../data/CelebA/celeba/'
output_directory = '../data/CelebA/training_edges/data/'

celeba = os.listdir(os.fsencode(input_directory))

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

def edges_grayscale_rgb(file_path):
    # Read in image as RGB array
    rgb = np.array(Image.open(file_path))
    
    # Convert RGB array into grayscale array
    grayscale = rgb[:,:,0]
    
    # Extract edges from grayscale array
    edges = filters.sobel(grayscale)
    
    return (edges, grayscale, rgb)

celeba_crop = (0, 20, 178, 218 - 20)
resolution = (64, 64)

def build_edges(count=0):
    built = 0
    
    # Iterate through each sample
    for file in celeba:
        filename = os.fsdecode(file)
        output = output_directory + filename[:-4]
    
        # First get edges construction
        edges, grayscale, rgb = edges_grayscale_rgb(input_directory + filename)
        edges = Image.fromarray(edges)
    
        # Then scale down edges to desired dimensions
        
        # Crop image to square to maintain aspect ratio
        cropped = edges.crop(celeba_crop)
        
        # Resample to desired resolution
        resampled = cropped.resize(resolution)
        
        plt.imsave(output_directory + str(built) + '.jpg', np.asarray(resampled), cmap='gray')
        
        built += 1

        if built % 1000 == 0:
            print('Generated %s images' % built)

        # If 'count' is 0, then whole dataset will be generated.
        if not count == 0 and built == count:
            break;

# Run code
build_edges()
