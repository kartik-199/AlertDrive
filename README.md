Face Recognition and Drowsy Driving Alert System
Overview
This Python program utilizes OpenCV, NumPy, and Tkinter to implement a face recognition system with an added drowsy driving alert feature. The system captures video frames from a webcam, detects faces, and recognizes individuals by comparing their facial features using Histogram of Oriented Gradients (HOG) descriptors. If a face is detected, the program allows the user to input the person's name for recognition.

Additionally, the program includes an alert system for drowsy driving. If a face is detected but eyes are closed for more than 5 seconds, an alert is displayed, indicating potential drowsiness.

Features
Face detection and recognition using HOG descriptors
Dynamic addition of new faces to the recognition database
Eye detection for drowsy driving alert
User interface for entering names of recognized individuals
Dependencies
Python 3.x
OpenCV
NumPy
Tkinter
Installation
Install the required dependencies:

bash
Copy code
pip install opencv-python numpy
Run the program:

bash
Copy code
python face_recognition_alert.py
Instructions
Press 's' to start the program and initiate face recognition.
If a face is detected, enter the person's name when prompted.
Press 'x' to exit the program.
Configuration
Face data is stored in the face_data.json file.
Modify the parameters such as cascade classifiers, timers, and alert messages as needed.
Notes
The program allows for continuous face recognition until manually terminated.
Be cautious while using the drowsy driving alert feature, and ensure the program is running in a suitable environment.
License
This program is open-source and available under the MIT License.

Feel free to customize the README based on additional features, configurations, or specific instructions for your program.
