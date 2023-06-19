import cv2
import os

# Prompt the user to select an option: capture an image from the webcam or load an image from a file
option = input("Select one option:\n1. open webcam to capture\n2. Browse the image on your desktop\n")

if option == '1':
    # If option 1 is selected, capture an image from the webcam
    
   
    name = input("Enter a name: ")
    
    # Create a directory to store the generated images, based on the user's name
    directory = "image_generated/{name}"
    os.makedirs(directory, exist_ok=True)
    
    # Access the webcam
    cap = cv2.VideoCapture(0)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Counter for the captured images
    count = 1
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw rectangles around the detected faces in the original frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the frame with the rectangles
        cv2.imshow("Capture", frame)
        
        # If 's' key is pressed and at least one face is detected, save the image
        if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
            image_path = f"{directory}/{count:02d}.jpg"
            cv2.imwrite(image_path, frame)
            print("Saved image:", image_path)
            count += 1
        
        # If 'q' key is pressed or the maximum number of images (100) is reached, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 100:
            break
    
    # Release the webcam and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

elif option == '2':
    # If option 2 is selected, load an image from a file
    
    # Prompt the user to enter the image file path
    file_path = input("Enter the path for the image: ")
    
    if os.path.isfile(file_path):
        # If the file path is valid, proceed with generating images
        
        # Prompt the user to enter the folder name
        name = input("Enter one folder name: ")
        
        # Create a directory to store the generated images, based on the folder name
        directory = f"image_generated/{name}"
        os.makedirs(directory, exist_ok=True)
        
        # Read the image from the file
        image = cv2.imread(file_path)
        
        # Counter for the generated images
        count = 1
        while count <= 100:
            # Generate image paths and save the image
            image_path = f"{directory}/{count:02d}.jpg"
            cv2.imwrite(image_path, image)
            print("Image saved:", image_path)
            count += 1
    
    else:
        # If the file path is invalid, display an error message
        print("File cannot be found!")

else:
    # If an invalid option is selected, display an error message
    print("Invalid option chosen!")

# Create a label file to store the labels of the generated images
label_file = open("label.txt", "w")

# Get a list of folders in the "image_generated" directory
folders = os.listdir("image_generated")

# Write the labels (folder names) to the label file
for i, folder in enumerate(folders):
    label = f"{i} {folder}\n"
    label_file.write(label)

# Close the label file
label_file.close()

# Display a success message
print("Successfully generated label.txt!"

)