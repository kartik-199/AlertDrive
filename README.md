# AlertDrive

## Overview
This prototype algorithm intends to aid in drowsy driving detection via a camera to capture live-time feed and relies on the usage of a main script and secondary data file. 

## Operation
The program operates by capturing video frames through a webcam and implementing facial recognition to either identify an existing user or create a new user profile. The facial recognition part of this program is completed using the concept Histogram of Oriented Gradients (HOG) descriptors. Once the program has started, constant eye detection and face monitoring is used to ensure the user is in the proper state. An alert system will be triggered if the user does not seem attentive for even a short period.  

## Features 
- Live camera feed and real-time analysis 
- Face recognition based on HOG descriptor profiles
- UI for new user profiles
- Constant eye detection monitoring
- Constant face monitoring
- Alert system triggered in real-time

## Dependencies
- Python 3.9
- OpenCV
- NumPy
- Tkinter

## Installation
Ensure that you have obtained the main script, requirements.txt, and the data file locally on your machine. Make sure to install all the necessary requirmements as listed in the requirements.txt file via "pip install -r requirements.txt"

## Instructions
The program can be controlled through a keyboard interface. Press 's' to start the alert system once the facial recognition and setup has been completed. Press 'x' to exit the program. 

## Configuration
The face data will be stored in "face_data.json" for use throughout the main driver script, "AlertDrive.py"

## Notes
- The program allows for continuous face recognition until manually terminated.
- Be cautious while using the drowsy driving alert feature, and ensure the program is running in a suitable environment.
