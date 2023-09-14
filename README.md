Image Depixelizer
Project Overview
This project aims to develop a deep learning model capable of restoring pixelated regions within images to their original state. The goal is to accurately predict the original values of pixelated areas within images by training on a subset of images and evaluating performance on a separate test set. The trained model can find applications in various fields such as medical imaging, satellite imaging, CCTV footage enhancement, and more.

Dataset
The dataset consists of grayscale images that have been partially pixelated. The training set contains 34,635 images housed in 350 folders, and the test set contains 6,635 images in 67 folders. The pixelation process involves selecting random regions within images and pixelating them with varying block sizes.

Training Data
Location: image_files/images/
Details: Contains 34,635 images in 350 folders. Each image has associated pixelated regions and corresponding masks indicating the pixelated areas.
Test Data
Location: image_files/test_set.pkl
Details: The pickle file contains a dictionary with two entries:
pixelated_images: A tuple of 6,635 NumPy arrays representing the pixelated images.
known_arrays: A tuple of 6,635 NumPy arrays representing the boolean masks showing the pixelated area within the images.
Installation
Requirements
Python 3.7 or later
PyTorch 1.8 or later
torchvision
PIL
NumPy

![alt text]([https://github.com/AlanRoyMiller/image_depixelization_project/blob/main/readme%20images/known_array.png)
