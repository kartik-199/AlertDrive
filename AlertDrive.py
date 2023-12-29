from tkinter.simpledialog import askstring
import cv2
import json
import numpy as np
from scipy.spatial import distance
import tkinter as tk
import time


# Function to calculate the gradient data of the image
def calculate_hog_descriptor(image):
    # Resizes the image in the parameter to optimize storage
    resized_image = cv2.resize(image, (64, 128))
    # Converts the image to grayscale and creates an object to obtain the HOG descriptor of the image
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog_descriptor = hog.compute(gray)
    # Converts the HOG descriptor to list format and returns it
    hog_descriptor = list(hog_descriptor)
    return hog_descriptor


# Function to compute HOG similarity of two HOG descriptors - used for checking if person detected exists in data
def calculate_cosine_similarity(descriptor1, descriptor2):
    # Ensures both descriptors are of the same length
    min_length = min(len(descriptor1), len(descriptor2))
    descriptor1 = descriptor1[:min_length]
    descriptor2 = descriptor2[:min_length]
    # Computes similarity in the range -1 to 1
    similarity = 1 - distance.cosine(descriptor1, descriptor2)
    return similarity


# Function to attempt to determine the closest match of a detected face and face data in the JSON file
def recognize_face(descriptor, face_data):
    max_similarity = 0
    recognized_name = None
    # Loops through the sequence of face data sets
    for person in face_data:
        # Accesses the HOG descriptor of an element
        known_descriptor = np.array(person['descriptor'])
        # Calculates the cosine similarity
        similarity = calculate_cosine_similarity(descriptor, known_descriptor)
        # If this similarity is better than previous detections, store this score
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = person['name']
    # Return the name and similarity score of the best match - note this does not mean that there has been a match
    return recognized_name, max_similarity


# Function to take name of person if face is detected
def take_input():
    root = tk.Tk()
    root.title("Face Detected!")
    root.geometry("100x100")
    root.withdraw()
    name = askstring("Name Input", "Enter the person's name: (ENTER C TO CANCEL)")
    return name


# Main driver function
def main():
    # Timer variables and necessary alert variables
    face_timer_start = time.time()
    eyes_timer_start = time.time()
    program_start = False
    program_started = False  # Flag to check if the program has started
    # Initializes an object for detecting faces
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Creates a camera window with a running video input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    modified = False
    # Attempt to open the JSON file with potentially existing face data
    try:
        with open('face_data.json', 'r') as file:
            face_data = json.load(file)
    except FileNotFoundError:
        print("JSON file not found!")
        return 1
    # Loop to read in frames from running video input
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 600))
        if not ret:
            break
        # Converts the current frame to grayscale for optimized face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=25)
        if len(faces) > 0 and not program_started:
            # If face is detected, start the program
            program_start = True
            program_started = True
            eyes_timer_start = time.time()
            face_timer_start = time.time()
        for (x, y, w, h) in faces:
            # If face is detected, reset the timer
            face_timer_start = time.time()
            face_roi = frame[y:y + h, x:x + w]
            # Calculate HOG descriptor if face is detected
            current_descriptor = calculate_hog_descriptor(face_roi)
            # Tries to find the closest match in the JSON file
            recognized_name, similarity = recognize_face(current_descriptor, face_data)
            # Code to detect eyes
            eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w], scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))
            # Start timer if eyes are not detected
            if len(eyes) != 0:
                eyes_timer_start = time.time()
            # Place box around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            # Places box around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Bool to store whether eyes were detected and display respective text
            eyes_open = 2 >= len(eyes) > 0
            if eyes_open:
                cv2.putText(frame, "Eyes Open", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eye(s) Closed", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Displays details if face name and data are found in JSON
            cv2.putText(frame, f"Name: {recognized_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Similarity: {similarity:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            # If no relevant match could be found, possibly add new data ...
            if similarity < 0.80:
                person_name = take_input()
                if person_name.lower() != "c":
                    face_data.append({'name': person_name, 'descriptor': list(current_descriptor)})
                    print(f"Face data for {person_name} added.")
                    modified = True
        if time.time() - face_timer_start >= 5 and program_start:
            cv2.putText(frame, "ALERT: DROWSY DRIVING (Face)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if time.time() - eyes_timer_start >= 5 and program_start:
            cv2.putText(frame, "ALERT: DROWSY DRIVING (Eyes)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Display frame
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Face Recognition', frame)
        # Exit program via clicking 'x'
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        # Start the program
        if cv2.waitKey(1) & 0xFF == ord('s'):
            program_start = True
            face_timer_start = time.time()
            eyes_timer_start = time.time()
    # Dump any new data into the JSON file
    with open('face_data.json', 'w') as file:
        json.dump(face_data, file, default=lambda y: y.tolist() if isinstance(y, np.ndarray) else float(y))
    if modified:
        print("File modified - please check for any unintended modifications...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
