import os
import random
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_image_files(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg"))
    return image_files

def apply_random_occlusion(image):
    h, w = image.shape[:2]
    occlusion_size = random.randint(int(0.1 * h), int(0.3 * h))
    x1 = random.randint(0, w - occlusion_size)
    y1 = random.randint(0, h - occlusion_size)
    x2 = x1 + occlusion_size
    y2 = y1 + occlusion_size
    occluded_image = image.copy()
    cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return occluded_image

def apply_random_noise(image):
    noise_level = random.uniform(0.02, 0.1)
    noisy_image = image.copy()
    noise = np.random.randn(*image.shape) * 255 * noise_level
    noisy_image = cv2.add(noisy_image.astype(np.float64), noise, dtype=cv2.CV_64F)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def get_next_image_index(image_files):
    return len(image_files)

def save_modified_image(image, output_folder, index):
    filename = f"{index}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, image)


def RandomOcclusionsAndImageNoise(folder, output_location):
    image_files = get_image_files(folder=folder)

    if not image_files:
        print(f"No images found in the folder: {folder}")
    else:
        print(f"Found {len(image_files)} images in the temporary folder.")

        occlusion_count = 0
        noise_count = 0
        ratio = 0
        ratios = []

        next_index = get_next_image_index(image_files)

        for idx, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            if random.random() < 0.5:
                modified_image = apply_random_occlusion(image)
                occlusion_count += 1
            else:
                modified_image = apply_random_noise(image)
                noise_count += 1
            
            save_modified_image(modified_image, folder, next_index + idx + 1)

            if noise_count > 0:
                ratio = occlusion_count / noise_count
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))

            print(f"Occluded images: {occlusion_count}, Noisy images: {noise_count}")
            print("Ratio:", ratios[-1])

        plt.figure(figsize=(12, 6))
        plt.plot(ratios, label='Occlusion/Noise Count Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.title('Occlusion to Noise Ratio Over Iterations')
        plt.legend()
        plt.grid(True)

        if not os.path.exists(output_location):
            os.makedirs(output_location)

        graph_filepath = os.path.join(output_location, "occlusion_noise_ratio.jpg")
        plt.savefig(graph_filepath)
