import pickle

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Stop', 'Rest']

with open('map/right_hand_classes.ls', 'wb') as f:
    pickle.dump(classes, f)
    print('map/right_hand_classes.ls was created')