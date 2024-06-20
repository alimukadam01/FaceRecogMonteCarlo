import os
import shutil
import cv2

from django.conf import settings

from .FaceCapture import FaceCapture
from .MakeGrid import display_images_in_grid
from .RandomRotationAndLightIntensities import RandomRotationAndLightIntensities
from .RandomOcclusionsAndImageNoise import RandomOcclusionsAndImageNoise
from .RandomSelection import RandomSelection

def create_folders(name, dataset_folder):
    temp_output = os.path.join(dataset_folder, "temp", f"{name}")
    train_output = os.path.join(dataset_folder, "train", f"{name}")
    valid_output = os.path.join(dataset_folder, "valid", f"{name}")
    grid_output = os.path.join(dataset_folder, "grid", f"{name}")
    graph_output = os.path.join(dataset_folder, "graph", f"{name}")


    os.makedirs(train_output, exist_ok=True)
    os.makedirs(valid_output, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    os.makedirs(grid_output, exist_ok=True)
    os.makedirs(graph_output, exist_ok=True)

    return temp_output, train_output, valid_output, grid_output, graph_output


def clear_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))

def capture_face_and_build_data(roll_no):

    dataset_folder = settings.FACE_RECOG_DATASET_PATH
    temp_output_folder, train_output_folder, valid_output_folder, grid_output_folder, graph_output_folder = create_folders(roll_no, dataset_folder)

    FaceCapture(temp_output_folder=temp_output_folder)

    black_and_white_grid = display_images_in_grid(temp_output_folder)
    black_and_white_grid_output_file = os.path.join(grid_output_folder, "CapturedImages.jpg")
    cv2.imwrite(black_and_white_grid_output_file, black_and_white_grid)

    RandomRotationAndLightIntensities(temp_output_folder, graph_output_folder)

    rotation_and_lighting_grid = display_images_in_grid(temp_output_folder)

    rotation_and_lighting_grid_output_file = os.path.join(grid_output_folder, "RotationAndLighting.jpg")
    cv2.imwrite(rotation_and_lighting_grid_output_file, rotation_and_lighting_grid)

    RandomOcclusionsAndImageNoise(temp_output_folder, graph_output_folder)

    occlusion_and_noise_grid = display_images_in_grid(temp_output_folder)

    occlusion_and_noise_grid_output_file = os.path.join(grid_output_folder, "OcclusionAndNoise.jpg")
    cv2.imwrite(occlusion_and_noise_grid_output_file, occlusion_and_noise_grid)

    RandomSelection(temp_output_folder, train_output_folder, valid_output_folder, graph_output_folder)

    training_grid = display_images_in_grid(train_output_folder)

    training_grid_output_file = os.path.join(settings.FACE_RECOG_DATASET_PATH, "grid", f"{roll_no}", "TrainingSet.jpg")
    cv2.imwrite(training_grid_output_file, training_grid)

    testing_grid = display_images_in_grid(valid_output_folder)

    testing_grid_output_file = os.path.join(grid_output_folder, "TestingSet.jpg")
    cv2.imwrite(testing_grid_output_file, testing_grid)

    clear_folder(os.path.join(settings.FACE_RECOG_DATASET_PATH, "temp"))