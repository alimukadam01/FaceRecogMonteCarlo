import cv2
import numpy as np
import os
from sklearn import metrics

from django.conf import settings

def load_images_labels(dataset_path, label_mapping, target_size=(100, 100)):
    images = []
    labels = []

    # Assuming train folder is inside the dataset
    train_folder = os.path.join(dataset_path, "train")

    # Check if the train folder exists
    if not os.path.exists(train_folder):
        print(f"Error: Train folder not found at path: {train_folder}")
        return np.array([]), np.array([])

    print(f"Loading images from {train_folder}")

    for person_folder in os.listdir(train_folder):
        person_path = os.path.join(train_folder, person_folder)
        if os.path.isdir(person_path):
            label = label_mapping[person_folder]
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize the image to the target size
                image = cv2.resize(image, target_size)
                
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

def create_label_mapping(dataset_path):
    label_mapping = {}
    label_counter = 0

    # Assuming train folder is inside the dataset
    train_folder = os.path.join(dataset_path, "train")

    # Check if the train folder exists
    if not os.path.exists(train_folder):
        print(f"Error: Train folder not found at path: {train_folder}")
        return label_mapping

    print(f"Creating label mapping from {train_folder}")

    label_mapping_path = settings.FACE_RECOG_LABEL_MAPPING_PATH
    with open(label_mapping_path, "w") as file:
        for person_folder in os.listdir(train_folder):
            person_path = os.path.join(train_folder, person_folder)
            if os.path.isdir(person_path):
                label_mapping[person_folder] = label_counter
                file.write(f"{label_counter}\t{person_folder}\n")
                label_counter += 1

    return label_mapping

def load_dataset(dataset_path, label_mapping):
    images = []
    labels = []

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize the image to the target size
                image = cv2.resize(image, (100, 100))  # Use the same size as during training
                
                images.append(image)
                labels.append(label_mapping[person_folder])

    return np.array(images), np.array(labels)

def train_model(X_train, y_train):
    # Train the Eigenfaces model
    if len(X_train) == 0:
        print("Error: No training data found. Ensure the dataset is correctly loaded.")
        return None

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X_train, y_train)

    return model

def evaluate_model(model, X_valid, y_valid):
    # Evaluate the model on the validation set
    predictions = []
    for face in X_valid:
        prediction, _ = model.predict(face)
        predictions.append(prediction)

    # Calculate performance metrics
    accuracy = metrics.accuracy_score(y_valid, predictions)
    precision = metrics.precision_score(y_valid, predictions, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")


def train_and_evaluate_model():
    # Load and prepare the training dataset
    dataset_path = settings.FACE_RECOG_DATASET_PATH
    label_mapping = create_label_mapping(dataset_path)
    X_train, y_train = load_images_labels(dataset_path, label_mapping)

    # Check if training data is loaded correctly
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: No training data loaded. Exiting script.")
        exit()

    # Train the model
    model = train_model(X_train, y_train)

    if model is not None:
        # Save the trained model
        model.save(f"{settings.FACE_RECOG_MODELS_DIR}/model.xml")

        # Load and prepare the validation dataset
        valid_dataset_path = f"{settings.FACE_RECOG_DATASET_PATH}/valid"
        X_valid, y_valid = load_dataset(valid_dataset_path, label_mapping)

        # Evaluate the model on the validation set
        evaluate_model(model, X_valid, y_valid)

    else:
        print("Model training failed due to lack of training data.")
