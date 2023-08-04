import pickle

import cv2
import numpy as np
import mediapipe as mp

import calibrate


# Loading trained models
with open('train_classifiers/right_hand_rf_model.p', 'rb') as f:
    right_rf_model_dict = pickle.load(f)
    right_rf_model = right_rf_model_dict['model']

with open('train_classifiers/right_hand_iso_model.p', 'rb') as f:
    right_iso_model_dict = pickle.load(f)
    right_iso_model = right_iso_model_dict['model']

with open('train_classifiers/left_hand_rf_model.p', 'rb') as f:
    left_rf_model_dict = pickle.load(f)
    left_rf_model = left_rf_model_dict['model']

with open('train_classifiers/left_hand_iso_model.p', 'rb') as f:
    left_iso_model_dict = pickle.load(f)
    left_iso_model = left_iso_model_dict['model']



# MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)

# Initialize the video capture object and set frame size
cap = cv2.VideoCapture(1)
width, height = 320, 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


is_calibrated = False   # Always start with calibration

while True:
    # Frame setting
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Horizontal flip
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    H, W, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, 'Press "E" to exit', (20, 20), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)
    if not ret:
        print('Video stream has stopped!')
        break

    # Process the image to find hand landmarks
    results = hands.process(image)

    # Do calibration to position hands decently and get the reference distance between two hands (for later musical dynamics calculation)
    if not is_calibrated:
        is_calibrated, calib_distance = calibrate.calibrate(cap)

    # Detect hand(s)
    # One hand is detected
    if is_calibrated and (not results.multi_hand_landmarks or results.multi_hand_landmarks and len(results.multi_hand_landmarks)) != 2:
        text = 'Show two hands!'
        text_size, _ = cv2.getTextSize(text, font, font_scale*1.5, font_thickness)
        center_position = ((W - text_size[0]) // 2, (H + text_size[1]) // 2)
        cv2.putText(frame, text, center_position, font, font_scale*1.5, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Two hands are detected
    elif is_calibrated and results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        # Define empty lists for landmarks, outlier_prediction, prediction, and center coordinates
        landmarks = [None, None]
        outlier_prediction = [None, None]
        prediction = [None, None]
        x_centers = [None, None]
        y_centers = [None, None]

        # Loop through each hand
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect x and y coordinates
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize the landmarks
            landmarks[i] = []
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y
                landmarks[i].append((x - min(x_)) / (max(x_) - min(x_)))
                landmarks[i].append((y - min(y_)) / (max(y_) - min(y_)))

            # If less than 21 landmarks detected, fill the rest with 0
            if len(landmarks[i]) < 42:
                landmarks[i].extend([0] * (42 - len(landmarks[i])))

            landmarks[i] = np.array([landmarks[i]])


            # Make a prediction using trained classifiers
            confidence_threshold = 0.8

            # Left hand prediction
            if i == 0:
                # Outlier detection using trained Isolation Forest
                outlier_prediction[0] = left_iso_model.predict(landmarks[0])
                
                # Make a prediction using trained Random Forest regardless of the outlier prediction
                pred_proba = left_rf_model.predict_proba(landmarks[0])
                confidence = np.max(pred_proba)

                # If the point is not an outlier or the classifier is confident in its prediction
                if outlier_prediction[0] == 1 or confidence > confidence_threshold:
                    prediction[0] = left_rf_model.predict(landmarks[0])
                else:
                    prediction[0] = 'None'

            # Right hand prediction
            else:
                # Outlier detection using trained Isolation Forest
                outlier_prediction[1] = right_iso_model.predict(landmarks[1])
                
                # Make a prediction using trained Random Forest regardless of the outlier prediction
                pred_proba = right_rf_model.predict_proba(landmarks[1])
                confidence = np.max(pred_proba)

                # If the point is not an outlier or the classifier is confident in its prediction
                if outlier_prediction[1] == 1 or confidence > confidence_threshold:
                    prediction[1] = right_rf_model.predict(landmarks[1])
                else:
                    prediction[1] = None

            # Draw bounding box and label on the frame
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            if i == 0:
                color = (200, 100, 0)
            else:
                color = (0, 100, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, str(prediction[i]), (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

            # Get bounding box's coordinate at its center
            x_centers[i] = (x1 + x2) // 2
            y_centers[i] = (y1 + y2) // 2

        x_dist = abs(x_centers[1] - x_centers[0])
        y_dist = y_centers[1] - y_centers[0]

        # Calculate normalized horizontal distance between two hands (for musical dynamics modulation)
        x_norm_dist = x_dist / calib_distance

        # Calculate the angular degree between the center line and the horizontal line (for pitch bend modulation)
        ver_angle = np.degrees(np.arctan2(y_dist, x_dist))

        # Draw the center line
        cv2.line(frame, (x_centers[0], y_centers[0]), (x_centers[1], y_centers[1]), (200, 0, 50), 2)

        # Print results
        left_hand_label  = prediction[0]
        right_hand_label = prediction[1]
        dynamics_factor   = round(x_norm_dist, 2)
        pitch_bend_factor = round(ver_angle, 2)
        print('{}, {}, {}, {}'.format(left_hand_label, right_hand_label, dynamics_factor, pitch_bend_factor))

        # Play sounds
        sound_parameters = [left_hand_label, right_hand_label, dynamics_factor, pitch_bend_factor]
        

    # Show the frame with hand landmarks and recognized gestures
    cv2.imshow('Signing and computer music', frame)

    # Exit condition
    if cv2.waitKey(1) == ord('e'):
        break
                        
cap.release()
cv2.destroyAllWindows()
