import pickle

classes = ['Dynamics', 'Vibrato', 'Track', '1', '2', '3', '4', '5', 'Stop', 'Rest']

with open('map/left_hand_classes.ls', 'wb') as f:
    pickle.dump(classes, f)
    print('map/left_hand_classes.ls was created')