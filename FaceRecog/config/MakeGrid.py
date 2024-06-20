import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_image_files(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg"))
    return image_files

def display_images_in_grid(folder):
    image_files = get_image_files(folder)
    
    if not image_files:
        print(f"No images found in the folder: {folder}")
        return None

    num_images = len(image_files)
    num_cols = 12
    num_rows = (num_images + num_cols - 1) // num_cols

    # Load and resize the first image to get dimensions
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_img.shape

    # Create an array to hold the resized image data
    grid_image = np.ones((height * num_rows, width * num_cols), dtype=np.uint8)

    # Iterate through images, resize, and concatenate
    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if idx < num_images:
                img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (width, height))  # Resize to match first image
                grid_image[row * height : (row + 1) * height, col * width : (col + 1) * width] = img
                idx += 1

    return grid_image

