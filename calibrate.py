# Hands position calibration

import cv2
import time
import mediapipe as mp
import numpy as np

def calibrate(cap):

    is_calibrated = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    hand_centers = []
    hand_distances = []
    hold_time = 0
    start_time = time.time()


    while hold_time <= 4:
        # Frame setting
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Horizontal flip
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        H, W, _ = frame.shape
        cv2.putText(frame, 'Press "E" to exit', (20, 20), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)
        if not ret:
            print('Video stream has stopped!')
            break

        # Process the image to find hand landmarks
        results = hands.process(image)

        if not results.multi_hand_landmarks or results.multi_hand_landmarks and len(results.multi_hand_landmarks) != 2:
            text = 'Show your hands at performing positions'
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            center_position = ((W - text_size[0]) // 2, (H + text_size[1]) // 2)
            cv2.putText(frame, text, center_position, font, font_scale, (50, 255, 0), font_thickness, cv2.LINE_AA)
            hold_time = 0
            start_time = time.time()
            hand_centers = []

        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # Loop through each hand
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # Collect x and y coordinates
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                # Draw bounding box on the frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                if x1 < W // 2:
                    hand_label = 'Left'
                else:
                    hand_label = 'Right'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                cv2.putText(frame, hand_label, (x1, y1 - 10), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

                # Store the center of each hand
                hand_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

            # Calculate the distance between two hands and store it
            x1, y1 = hand_centers[0]
            x2, y2 = hand_centers[1]
            hand_distances.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))  # Euclidean distance calculation
            hand_centers = []  # Reset the list of hand centers for the next frame

            hold_time = time.time() - start_time
            
            text = 'Hold still and wait!'
            text_size, _ = cv2.getTextSize(text, font, font_scale*1.5, font_thickness)
            center_position = ((W - text_size[0]) // 2, (H + text_size[1]) // 10)
            cv2.putText(frame, text, center_position, font, font_scale*1.5, (50, 50, 255), font_thickness, cv2.LINE_AA)

        cv2.imshow('Signing and computer music', frame)   # Show the frame during calibration

        # Exit condition
        if cv2.waitKey(1) == ord('e'):
            is_calibrated = False
            calib_distance = None
            print("Calibration was manually stopped before completion.")
            return (is_calibrated, calib_distance)


    if not hand_distances:
        print("No hand distances recorded.")
        return (False, None)

    # Remove outliers from hand distances using IQR
    hand_distances.sort()
    Q1 = hand_distances[len(hand_distances) // 4]
    Q3 = hand_distances[len(hand_distances) * 3 // 4]
    IQR = Q3 - Q1
    hand_distances = [x for x in hand_distances if Q1 - 1.5 * IQR <= x <= Q3 + 1.5 * IQR]

    if not hand_distances:
        print("No hand distances left after IQR filtering.")
        return (False, None)

    # Calculate the average distance
    calib_distance = np.mean(hand_distances)

    return (is_calibrated, calib_distance)
