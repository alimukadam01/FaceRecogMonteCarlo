import cv2
import numpy as np

from django.conf import settings

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def load_trained_model(model_path):
    model = cv2.face.EigenFaceRecognizer_create()
    model.read(model_path)
    return model

def load_label_mapping(label_mapping_path):
    label_mapping = {}
    with open(label_mapping_path, "r") as file:
        for line in file:
            label, name = line.strip().split("\t")
            label_mapping[int(label)] = name
    return label_mapping

def detect_faces(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return faces

def predict_faces(face_roi, model, target_size=(200, 200)):
    # Resize the input face region to the target size used during training
    face_roi_resized = cv2.resize(face_roi, target_size)
    gray_face = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
    
    # IMPORTANT: Resize the flattened face to the size expected by the model (10000 elements for 200x200)
    flattened_face = cv2.resize(gray_face, (100, 100)).flatten()

    # Make predictions using the trained model
    prediction, _ = model.predict(np.array([flattened_face]))
    return prediction

def draw_prediction(frame, x, y, w, h, label, label_mapping):
    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the predicted label and confidence
    confidence = 1.0  # set confidence to 1.0 for simplicity
    text = f"{label_mapping[label]} ({confidence:.2f})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def mode_result(labels):
    count_dict = {}

    for element in labels:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1

    label = max(count_dict, key=count_dict.get)

    return label


def main():
    labels= []

    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the trained model
    model_path = f"{settings.FACE_RECOG_MODELS_DIR}/model.xml"
    model = load_trained_model(model_path)

    # Load label mapping
    label_mapping_path = settings.FACE_RECOG_LABEL_MAPPING_PATH
    label_mapping = load_label_mapping(label_mapping_path)

    # Start capturing video from the camera (index 0)
    cap = initialize_camera()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]

            # Make predictions using the trained model
            label = predict_faces(face_roi, model)
            labels.append(label)

            # Draw prediction on the frame
            draw_prediction(frame, x, y, w, h, label, label_mapping)

        # Display the frame with predictions
        cv2.imshow("Face Recognition", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()

    # Close the OpenCV window
    cv2.destroyAllWindows()

    return label_mapping[mode_result(labels)]


