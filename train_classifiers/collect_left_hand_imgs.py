# Collect images for left hand classes

import pickle
import os
import cv2

# Create a directory to store the captured data
DATA_DIR = './train_classifiers/data/left_hand'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Load classes and define the desired dataset size for each class
with open('map/left_hand_classes.ls', 'rb') as f:
    left_hand_classes = pickle.load(f)
dataset_size = 200


# Set desired width and height for video capture
width, height = 640, 480

# Open the video capture with the desired resolution
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Function for capturing images for the passing class
def capture_class_images(class_label):
    # Create a directory for class to store images
    if not os.path.exists(os.path.join(DATA_DIR, str(class_label))):
        os.makedirs(os.path.join(DATA_DIR, str(class_label)))

        print('Collecting data for class {}'.format(class_label))
        
    # Show a message to the user and wait for 's' key press to start collecting data
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Horizontal flip
        cv2.putText(frame, 'Press "S" to start collecting!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collecting data', frame)
        if cv2.waitKey(25) == ord('s'):
            break

    # Capture dataset_size number of frames and save them to the respective class directory
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Horizontal flip
        cv2.imshow('Collecting data', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, class_label, '{}.jpg'.format(counter)), frame)
        counter += 1

    cv2.destroyWindow('Collecting data')



collecting_mode = str(input('Colleting mode:\n1. Collect all classes\n2. Select one class to collect\n'))

# Collect data for all classes
if collecting_mode == '1':
    # Loop through each class
    for left_hand_class in left_hand_classes:
        capture_class_images(left_hand_class)

# Collect data for one class            
elif collecting_mode == '2':
    print(left_hand_classes)
    chosen_class = str(input('Choose the class you want to collect data for: '))
    if chosen_class in left_hand_classes:
        capture_class_images(chosen_class)



# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()