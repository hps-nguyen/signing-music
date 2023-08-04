# MIDI processing and sound generation

import fluidsynth
from collections import deque


label_to_midi = {
    'Rest': 0,
    'C': 48,  # C3
    'D': 50,  # D3
    'E': 52,  # E3
    'F': 53,  # F3
    'G': 55,  # G3
    'A': 57,  # A3
    'B': 59,  # B3
}



track_octave_multiplier = {
    0: 1,  # Track 0: C3 to B3
    1: 2,  # Track 1: C4 to B4
    2: 3,  # Track 2: C5 to B5
    3: 4,  # Track 3: C6 to B6
    4: 2,  # Track 4: C4 to B4
}



triads = {
    # 'label': [(root, volume), (third, volume), (fifth, volume)]
    'C': [(48, 1), (52, 0.7), (55, 0.7)],  # C3, E3, G3
    'D': [(50, 1), (53, 0.7), (57, 0.7)],  # D3, F3, A3
    'E': [(52, 1), (55, 0.7), (59, 0.7)],  # E3, G3, B3
    'F': [(53, 1), (57, 0.7), (60, 0.7)],  # F3, A3, C4
    'G': [(55, 1), (59, 0.7), (62, 0.7)],  # G3, B3, D4
    'A': [(57, 1), (60, 0.7), (64, 0.7)],  # A3, C4, E4
    'B': [(59, 1), (62, 0.7), (65, 0.7)],  # B3, D4, F4
}



last_chord = {0: None, 1: None, 2: None, 3: None, 4: None}



def create_control_change_event(fs, track_num, control, control_value):
    fs.cc(track_num, control, int(control_value))



def play_note(fs, seq, sound_port, track_num, note_label, volume_factor, bending_factor):
    current_time = seq.get_tick()

    if track_num != 4:  # Not the chord track
        note_number = label_to_midi[note_label] + (track_octave_multiplier[track_num] - 1) * 12
        velocity = int(80 * volume_factor)
        seq.note_on(current_time, channel=track_num, key=note_number, velocity=velocity, dest=sound_port)
        create_control_change_event(fs, track_num, 1, int(abs(bending_factor)))  # Apply vibrato
        create_control_change_event(fs, track_num, 7, int(volume_factor*127))  # Make dynamics more sensitive
        print(f'{note_label} is played on track {track_num}')
    else:  # Chord track
        for note_number, velocity_factor in triads[note_label]:
            note_number += (track_octave_multiplier[track_num] - 1) * 12  # Shift the notes of the chord by the desired number of octaves
            velocity = int(80 * volume_factor * velocity_factor)
            seq.note_on(current_time, channel=track_num, key=note_number, velocity=velocity, dest=sound_port)
        print(f'{note_label} major chord is played on track {track_num}')
        last_chord[track_num] = note_label  # Save the last played chord



def stop_note(fs, seq, sound_port, track_num, note_label):
    current_time = seq.get_tick()

    if track_num != 4:  # Not the chord track
        note_number = label_to_midi[note_label] + (track_octave_multiplier[track_num] - 1) * 12
        seq.note_off(current_time, channel=track_num, key=note_number, dest=sound_port)
        create_control_change_event(fs, track_num, 64, 0)  # Release the sustain pedal
        print(f'{note_label} is stopped on track {track_num}')
    else:  # Chord track
        if note_label == 'Rest' and last_chord[track_num] is not None:  # If it's a rest, stop the last chord
            for note_number, _ in triads[last_chord[track_num]]:
                note_number += (track_octave_multiplier[track_num] - 1) * 12  # Shift the notes of the chord by the desired number of octaves
                seq.note_off(current_time, channel=track_num, key=note_number, dest=sound_port)
            print(f'{last_chord[track_num]} major chord is stopped on track {track_num}')
            last_chord[track_num] = None  # Reset the last chord
        elif note_label in triads:  # If it's not a rest and there's a chord to be stopped
            for note_number, _ in triads[note_label]:
                note_number += (track_octave_multiplier[track_num] - 1) * 12  # Shift the notes of the chord by the desired number of octaves
                seq.note_off(current_time, channel=track_num, key=note_number, dest=sound_port)
            print(f'{note_label} major chord is stopped on track {track_num}')
            last_chord[track_num] = note_label  # Save the last played chord



def generate(fs, sfid, seq, sound_port, param_queue, beat_queue):
    buffer_size = 10  # To keep last 10 sound_parameters
    buffer = deque(maxlen=buffer_size)
    threshold = 0.25  # Applying label sensitivity

    last_right_label = 'None'
    track_num = 0  # Default track: Piano
    changing_track = False

    while True:
        # Process received data from video capturing
        sound_parameters = param_queue.get()
        buffer.append(sound_parameters)

        if sound_parameters is None or sound_parameters == 'END':
            break
        else:
            # Unpack the sound parameters
            if len(sound_parameters) == 4:
                left_label, right_label, volume_factor, bending_factor = sound_parameters
            else:
                continue

        # Check for global stop command
        if left_label == 'Stop' and right_label == 'Stop':
            for track_num in range(5):  # Stop all tracks
                if last_chord[track_num] is not None:  # If a chord was playing on this track, stop it
                    create_control_change_event(fs, track_num, 7, 0)  # Set volume to zero
                    stop_note(fs, seq, sound_port, track_num, last_chord[track_num])
                    print('Performance has stopped')
            continue

        # Change track
        if changing_track:
            buffer_labels = [param[0] for param in buffer]  # Extract the left_label from the buffer
            most_common_label = max(set(buffer_labels), key=buffer_labels.count)
            print('Changing track. Most common label: ', most_common_label)

            if most_common_label in ['1', '2', '3', '4', '5']:
                track_num = int(most_common_label) - 1  # Tracks 0 to 4
                changing_track = False

        # Check for track changing command
        if left_label == 'Track' and right_label == 'Rest':
            changing_track = True
            print('Changing track is requested')
            continue

        # Change the note only if the new note has reached the threshold in the buffer
        buffer_labels = [param[1] for param in buffer]  # Extract the right_label from the buffer
        most_common_label = max(set(buffer_labels), key=buffer_labels.count)

        if most_common_label != last_right_label and buffer_labels.count(most_common_label) / buffer_size >= threshold:
            # Stop playing the previous note when the label changes
            if last_right_label in label_to_midi:
                stop_note(fs, seq, sound_port, track_num, last_right_label)

            # Start playing the new note when the label changes
            if most_common_label in label_to_midi and most_common_label != 'Rest':
                create_control_change_event(fs, track_num, 7, int(volume_factor*127))  # Make dynamics more sensitive
                create_control_change_event(fs, track_num, 64, 127)  # Press the sustain pedal
                play_note(fs, seq, sound_port, track_num, most_common_label, volume_factor, bending_factor)

            last_right_label = most_common_label




# In progress
            
def start(fs, seq, synth_port):
    count_in_beats = 4 * 2

    # TODO for metronome, timing synchronization

    return True


