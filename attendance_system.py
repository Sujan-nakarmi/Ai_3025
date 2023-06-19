# Importing the required libraries
import cv2  # To utilize OpenCV for computer vision tasks, include the library in your project by importing it
import tensorflow as tf  # To incorporate TensorFlow for deep learning tasks, import the library into your project
from tensorflow import keras  # To facilitate the creation and training of neural networks, import the Keras module into project
import numpy as np  # To perform numerical operations, import the NumPy library into your project
import datetime  # To add timestamps to your code, import the datetime module into your project

attendance_log = "attendance.txt"  #Specify the file path for the attendance log

def get_class_name(class_no):
    """
    Function to map class numbers to class names.
    :param class_no: Integer representing the class number
    :return: Corresponding class name as a string
    """
    class_names = ["alice", "pawan", "rupendra", "saugat", "sujan"]
    if class_no >= 0 and class_no < len(class_names):
        return class_names[class_no]
    else:
        return "Unknown"

model = keras.models.load_model('facedetection.h5')  # Load the face detection model that has been trained

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load the face cascade classifier into the program

cap = cv2.VideoCapture(0)  # Initialize and open the video capture device (webcam) for further usage.
cap.set(4, 480)  # Adjust the height of the captured video to the desired value.
font = cv2.FONT_HERSHEY_COMPLEX  #Specify the font to be used for displaying text on the image.

while True:
    success, img_original = cap.read()  # Retrieve a frame from the video capture device.
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  #Convert the frame to grayscale format.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) \

    for (x, y, w, h) in faces:
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)  #it Draws a square around the face when it is detected
        crop_img = img_original[y:y + h, x:x + w]  
        img = cv2.resize(crop_img, (224, 224)) 
        img = img / 255.0 
        img = np.expand_dims(img, axis=0)  
        prediction = model.predict(img) 
        class_index = np.argmax(prediction)  
        class_name = get_class_name(class_index)  

        cv2.putText(img_original, class_name, (x, y + h + 20), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)  # Display the class name above the detected face

        with open(attendance_log, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the timestamp in current event
            attendance_entry = f"{timestamp} - {class_name}\n"  # this will create a attendance entery
            file.write(attendance_entry)  

        cv2.putText(img_original, f"{class_name} - Attendance Registered", (x, y - 10), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)  

    cv2.imshow("Result", img_original) #Display the processed frame, showcasing the results of face detection and recognition.
    cv2.waitKey(1)  

    if len(faces) > 0:
        cv2.waitKey(3000) # If at least one face is detected, wait for 3 seconds before terminating the program.
        break

cap.release()  
cv2.destroyAllWindows()  # Closes windows
