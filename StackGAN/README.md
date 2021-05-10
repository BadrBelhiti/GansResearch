## Running the model
Use the StackRunner notebook to forward propagate through the entire StackGAN.

## Concerning CelebA
This zip includes a small subset of CelebA to test the
functionality of the dataset-building scripts. This
mini dataset of 64 samples is not enough to train the
models however.

## Building datasets (Requires CelebA)
Build edges dataset - python edges_builder.py
Build edges->grayscale dataset - python edges_grayscale.py
Build grayscale->rgb dataset - python grayscale_rgb.py

## Training the individual stages (Requires datasets)
Train baseline - python baseline_train.py
Train edges->grayscale - python edges2grayscale_train.py
Train grayscale->rgb - python grayscale2rgb_train.py
