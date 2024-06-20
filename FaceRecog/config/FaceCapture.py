import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
image_count = 1

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def configure_window():
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_TOPMOST, 1)

def is_blurry(image):
    # Calculate image variance as a measure of blur
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < 100

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    return len(eyes) >= 2

def save_raw_face(frame, output_folder, elapsed_time):
    global image_count  # Use the global image_count variable

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Check for blurry image and detect eyes
        if not is_blurry(face_roi) and detect_eyes(frame):
            # Construct the file name with the image count
            frame_filename = os.path.join(output_folder, f"{image_count}.jpg")
            cv2.imwrite(frame_filename, face_roi)

            # Increment the image count for the next image
            image_count += 1

def capture_frames(cap, temp_output_folder, duration=30):
    start_time = cv2.getTickCount()
    image_count = 1


    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            cv2.imshow("Camera Feed", frame)

            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            save_raw_face(frame, temp_output_folder, elapsed_time)


            # # Determine whether to save in train or valid folder based on elapsed time
            # if elapsed_time < 20:
            #     save_raw_face(frame, train_output_folder, elapsed_time)
            # else:
            #     save_raw_face(frame, valid_output_folder, elapsed_time - 20)

            # Break the loop if the total duration is reached
            if elapsed_time >= duration:
                break

            if cv2.waitKey(1) == 13:
                break
    except Exception as error:
        print(f"An error occurred: {error}")

def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()

def FaceCapture(temp_output_folder):
    cap = initialize_camera()

    if cap is not None:
        configure_window()
        capture_frames(cap, temp_output_folder)
        cleanup(cap)