# Main program

import pickle
import traceback

import cv2
import numpy as np
import mediapipe as mp
import calibrate

import time
import queue
import threading
import fluidsynth
import sound_generation




# Sound and musical time setup

# Initialize FluidSynth and load SoundFont
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload('./FluidR3_GM.sf2')
fs.program_select(0, sfid, 0, 0)    # Piano
fs.program_select(1, sfid, 0, 15)   # Dulcimer
fs.program_select(2, sfid, 0, 77)   # Shakuhachi
fs.program_select(3, sfid, 0, 9)    # Glockenspiel
fs.program_select(4, sfid, 0, 48)   # Strings Ensemble

# Create sequencer
sequencer = fluidsynth.Sequencer(use_system_timer=False)

# Create the sound port
synth_port = sequencer.register_fluidsynth(fs)

# Create a flag to control the thread
thread_running = threading.Event()
thread_running.set()    # Start as True

# Define thread-safe queues for sound parameters and musical timing
sound_parameters_queue = queue.Queue()
beat_queue = queue.Queue()

# Define musical time
TEMPO = 80  # bpm
METER = 4   # quadruple meter
beat_duration = 60 / TEMPO


# Metronome function (threading musical timing)
# This function is not used yet
def metronome(beat_queue):
    try:
        while thread_running.is_set():
            
            time.sleep(beat_duration)
            beat_queue.put(1)
            
    except Exception as e:
        traceback.print_exc()
        print(f"Exception in metronome: {e}")


# Sound generation function
def generate_sound(sound_parameters_queue, beat_queue):
    try:
        while thread_running.is_set():
            sound_parameters = sound_parameters_queue.get()
            if sound_parameters == 'END':
                break

            # Generate sound based on the parameters
            sound_generation.generate(fs, sfid, sequencer, synth_port, sound_parameters_queue, beat_queue)

    except Exception as e:
        traceback.print_exc()
        print(f"Exception in generate_sound: {e}")


# Create and start the sound processing thread
sound_thread = threading.Thread(target=generate_sound, args=(sound_parameters_queue, beat_queue,))
sound_thread.start()

# Create and start the metronome thread
metronome_thread = threading.Thread(target=metronome, args=(beat_queue,))
metronome_thread.start()





# Classifier setup

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




# Video capture and sound playing

is_calibrated = False   # Always start with calibration
exit_flag = False       # Flag to control main loop
is_started = False      # Flag to control when performance starts
previous_time = time.perf_counter()

while not exit_flag:
    # Calculate the time delta since the last frame
    current_time = time.perf_counter()
    time_delta = int((current_time - previous_time) * 1000)  # Convert to milliseconds
    previous_time = current_time

    # Manually advance the sequencer
    sequencer.process(time_delta)

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
        if not is_calibrated:
            exit_flag = True

    # Detect hand(s)
    # One hand is detected
    if is_calibrated and (not results.multi_hand_landmarks or results.multi_hand_landmarks and len(results.multi_hand_landmarks)) != 2:
        text = 'Show two hands!'
        text_size, _ = cv2.getTextSize(text, font, font_scale*1.5, font_thickness)
        center_position = ((W - text_size[0]) // 2, (H + text_size[1]) // 2)
        cv2.putText(frame, text, center_position, font, font_scale*1.5, (50, 255, 0), font_thickness, cv2.LINE_AA)

    # Two hands are detected
    elif is_calibrated and results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        # Define empty lists for landmarks, outlier_prediction, prediction, and center coordinates
        landmarks = [[], []]
        outlier_prediction = [None, None]
        prediction = [None, None]
        x_centers = [None, None]
        y_centers = [None, None]
        x_ = [[], []]
        y_ = [[], []]

        # Loop through each hand to collect landmarks
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect x and y coordinates
            x_[i] = [landmark.x for landmark in hand_landmarks.landmark]
            y_[i] = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize the landmarks
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y
                landmarks[i].append((x - min(x_[i])) / (max(x_[i]) - min(x_[i])))
                landmarks[i].append((y - min(y_[i])) / (max(y_[i]) - min(y_[i])))

            # If less than 21 landmarks detected, fill the rest with 0
            if len(landmarks[i]) < 42:
                landmarks[i].extend([0] * (42 - len(landmarks[i])))

            landmarks[i] = np.array([landmarks[i]])

        # Determine which hand is left and which is right
        if np.mean(x_[0]) < np.mean(x_[1]):
            left_hand_index, right_hand_index = 0, 1
        else:
            left_hand_index, right_hand_index = 1, 0

        for i in [left_hand_index, right_hand_index]:
            # Make a prediction using trained classifiers
            confidence_threshold = 0.8
            if i == left_hand_index:
                # Outlier detection using trained Isolation Forest
                outlier_prediction[i] = left_iso_model.predict(landmarks[i])
                # Make a prediction using trained Random Forest regardless of the outlier prediction
                pred_proba = left_rf_model.predict_proba(landmarks[i])
                confidence = np.max(pred_proba)
                # If the point is not an outlier or the classifier is confident in its prediction
                if outlier_prediction[i] == 1 or confidence > confidence_threshold:
                    prediction[i] = left_rf_model.predict(landmarks[i])
                else:
                    prediction[i] = 'None'
            else:
                # Outlier detection using trained Isolation Forest
                outlier_prediction[i] = right_iso_model.predict(landmarks[i])
                # Make a prediction using trained Random Forest regardless of the outlier prediction
                pred_proba = right_rf_model.predict_proba(landmarks[i])
                confidence = np.max(pred_proba)
                # If the point is not an outlier or the classifier is confident in its prediction
                if outlier_prediction[i] == 1 or confidence > confidence_threshold:
                    prediction[i] = right_rf_model.predict(landmarks[i])
                else:
                    prediction[i] = None

            # Draw bounding box and label on the frame
            x1 = int(min(x_[i]) * W) - 10
            y1 = int(min(y_[i]) * H) - 10
            x2 = int(max(x_[i]) * W) + 10
            y2 = int(max(y_[i]) * H) + 10
            color = (200, 100, 0) if i == left_hand_index else (0, 100, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, str(prediction[i]), (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

            # Get bounding box's coordinate at its center
            x_centers[i] = (x1 + x2) // 2
            y_centers[i] = (y1 + y2) // 2

        x_dist = abs(x_centers[1] - x_centers[0])
        y_dist = y_centers[1] - y_centers[0]

        # Calculate normalized horizontal distance between two hands (for musical dynamics modulation)
        x_norm_dist = x_dist / (calib_distance * 1.2)

        # Calculate the angular degree between the center line and the horizontal line (for pitch bend modulation)
        ver_angle = np.degrees(np.arctan2(y_dist, x_dist))

        # Draw the center line
        cv2.line(frame, (x_centers[0], y_centers[0]), (x_centers[1], y_centers[1]), (200, 0, 50), 2)

        # Print results
        left_hand_label  = prediction[left_hand_index]
        right_hand_label = prediction[right_hand_index]
        if isinstance(left_hand_label, np.ndarray):
            left_hand_label  = str(left_hand_label[0])     # Convert to string
        if isinstance(right_hand_label, np.ndarray):
            right_hand_label = str(right_hand_label[0])    # Convert to string
        dynamics_factor   = round(x_norm_dist, 2)
        pitch_bend_factor = round(ver_angle, 2)
        #print('{}, {}, {}, {}'.format(left_hand_label, right_hand_label, dynamics_factor, pitch_bend_factor))

        # Play sounds
        if not is_started:
            beat_queue.put(1)
            is_started = sound_generation.start(fs, sequencer, synth_port)

        sound_parameters = [left_hand_label, right_hand_label, dynamics_factor, pitch_bend_factor]
        sound_parameters_queue.put(sound_parameters)



    # Show the frame with hand landmarks and recognized gestures
    cv2.imshow('Signing and computer music', frame)

    # Exit condition. Check for the 'E' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        exit_flag = True



# Stop threading
sound_parameters_queue.put('END')
thread_running.clear()
sound_thread.join()
metronome_thread.join()

# Release resources and close the video capture
cap.release()
cv2.destroyAllWindows()

