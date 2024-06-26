import os
import shutil
import random
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_image_files(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg"))
    return image_files

def copy_image(image_file, destination_folder):
    shutil.copy(image_file, destination_folder)

def clear_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))

def apply_lighting_variation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    random_brightness = random.uniform(0.5, 1.5)
    v = np.clip(v * random_brightness, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = v
    lighting_variation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return lighting_variation

def apply_random_rotation(image):
    angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def get_next_image_index(image_files):
    return len(image_files)

def save_modified_image(image, output_folder, index):
    filename = f"{index}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, image)

def RandomRotationAndLightIntensities(folder, output_location):
    image_files = get_image_files(folder=folder)
    if not image_files:
        print(f"No images found in the folder: {folder}")
        
    else:
        print(f"Found {len(image_files)} images in the temporary folder.")
        # Initialize count variables
        rotation_count = 0
        lightintensity_count = 0
        ratio = 0
        ratios = []

        next_index = get_next_image_index(image_files)

        # Simulate the assignment process
        for idx, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            if random.random() < 0.8:
                modified_image = apply_random_rotation(image)
                rotation_count += 1
            else:
                modified_image = apply_lighting_variation(image)
                lightintensity_count += 1
            
            save_modified_image(modified_image, folder, next_index + idx +1)

            # Calculate and store the ratio
            if lightintensity_count > 0:
                ratio = rotation_count / lightintensity_count
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))  # Handle the case where valid_count is 0

            # Print the updated counts and ratio
            print(f"Rotated images: {rotation_count}, Light Intensity Changed images: {lightintensity_count}")
            print("Ratio:", ratios[-1])

        # Plotting the ratio values
        plt.figure(figsize=(12, 6))
        plt.plot(ratios, label='Rotated/Lightintensity Count Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.title('Rotated to Lightintensity Ratio Over Iterations')
        plt.legend()
        plt.grid(True)

        graph_filepath = os.path.join(output_location, "rotation_lighting_ratio.jpg")
        plt.savefig(graph_filepath)


