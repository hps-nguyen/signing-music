# Create dataset for both hands

import os
import pickle

import cv2
import mediapipe as mp




# Setup MediaPipe Hands module and other required modules
mp_hands = mp.solutions.hands

# Create a Hands object for hand detection using MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

# Specify the directory containing the data (images of hands)
RIGHT_HAND_DATA_DIR = './train_classifiers/data/right_hand'
LEFT_HAND_DATA_DIR = './train_classifiers/data/left_hand'



  
# Create dataset for right hand images

# Lists to store the extracted hand landmarks and corresponding labels
rightHand_data = []
rightHand_labels = []

# Loop through each subdirectory in the RIGHT_HAND_DATA_DIR (representing each class)
for dir_ in os.listdir(RIGHT_HAND_DATA_DIR):
    # Loop through each image file in the subdirectory
    for img_path in os.listdir(os.path.join(RIGHT_HAND_DATA_DIR, dir_)):
        # List to store the normalized hand landmark data for this image
        rightHand_data_aux = []

        # Lists to store x and y coordinates of detected landmarks for normalization
        x_ = []
        y_ = []

        # Read the image and convert it to RGB format
        img = cv2.imread(os.path.join(RIGHT_HAND_DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each detected hand landmark to extract x and y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Loop through each detected hand landmark again for normalization
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    rightHand_data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
                    rightHand_data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

                # If the number of detected landmarks is less than 21 (total landmarks in hand), append zero for the missing ones
                if len(rightHand_data_aux) < 42:
                    rightHand_data_aux.extend([0]*(42 - len(rightHand_data_aux)))

            # Add the normalized hand landmark data for this image to the main data list ONLY if it has correct number of landmarks
            if len(rightHand_data_aux) == 42:
                rightHand_data.append(rightHand_data_aux)
                # Add the corresponding label (class name) for this image to the labels list
                rightHand_labels.append(dir_)

# Save the extracted hand landmark data and labels as a pickle file (binary)
f = open('train_classifiers/right_hand_data.pickle', 'wb')
pickle.dump({'data': rightHand_data, 'labels': rightHand_labels}, f)
print('train_classifiers/right_hand_data.pickle was created')
f.close()




# Create dataset for left hand images

# Lists to store the extracted hand landmarks and corresponding labels
leftHand_data = []
leftHand_labels = []

# Loop through each subdirectory in the LEFT_HAND_DATA_DIR (representing each class)
for dir_ in os.listdir(LEFT_HAND_DATA_DIR):
    # Loop through each image file in the subdirectory
    for img_path in os.listdir(os.path.join(LEFT_HAND_DATA_DIR, dir_)):
        # List to store the normalized hand landmark data for this image
        leftHand_data_aux = []

        # Lists to store x and y coordinates of detected landmarks for normalization
        x_ = []
        y_ = []

        # Read the image and convert it to RGB format
        img = cv2.imread(os.path.join(LEFT_HAND_DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each detected hand landmark to extract x and y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Loop through each detected hand landmark again for normalization
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    leftHand_data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
                    leftHand_data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

                # If the number of detected landmarks is less than 21 (total landmarks in hand), append zero for the missing ones
                if len(leftHand_data_aux) < 42:
                    leftHand_data_aux.extend([0]*(42 - len(leftHand_data_aux)))

            # Add the normalized hand landmark data for this image to the main data list ONLY if it has correct number of landmarks
            if len(leftHand_data_aux) == 42:
                leftHand_data.append(leftHand_data_aux)
                # Add the corresponding label (class name) for this image to the labels list
                leftHand_labels.append(dir_)
        
# Save the extracted hand landmark data and labels as a pickle file (binary)
f = open('train_classifiers/left_hand_data.pickle', 'wb')
pickle.dump({'data': leftHand_data, 'labels': leftHand_labels}, f)
print('train_classifiers/left_hand_data.pickle was created')
f.close()
