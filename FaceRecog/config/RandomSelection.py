import os
import shutil
import random
import glob
import matplotlib.pyplot as plt

def get_image_files(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg"))
    return image_files

def copy_image(image_file, destination_folder):
    shutil.copy(image_file, destination_folder)

def RandomSelection(temp_output_folder, train_output_folder, valid_output_folder, graph_output_folder):
    image_files = get_image_files(temp_output_folder)
    print(f"Found {len(image_files)} images in the temporary folder.")

    # Initialize count variables
    train_count = 0
    valid_count = 0
    ratio = 0
    ratios = []

    # Simulate the assignment process
    for image_file in image_files:
        if random.random() < 0.8:
            copy_image(image_file, train_output_folder)
            train_count += 1
        else:
            copy_image(image_file, valid_output_folder)
            valid_count += 1
        
        # Calculate and store the ratio
        if valid_count > 0:
            ratio = train_count / valid_count
            ratios.append(ratio)
        else:
            ratios.append(float('inf'))  # Handle the case where valid_count is 0

        # Print the updated counts and ratio
        print(f"Train images: {train_count}, Valid images: {valid_count}")
        print("Ratio:", ratios[-1])

    # Plotting the ratio values
    plt.figure(figsize=(12, 6))
    plt.plot(ratios, label='Train/Valid Ratio')
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Train to Valid Image Ratio Over Iterations')
    plt.legend()
    plt.grid(True)

    graph_filepath = os.path.join(graph_output_folder, "train_valid_ratio.jpg")
    plt.savefig(graph_filepath)


