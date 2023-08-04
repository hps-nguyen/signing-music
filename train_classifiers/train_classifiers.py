# Train classifiers

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np




# Train right hand classifier

# Load dataset
rightHand_data_dict = pickle.load(open('./train_classifiers/right_hand_data.pickle', 'rb'))

data = np.asarray(rightHand_data_dict['data'])
labels = np.asarray(rightHand_data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier()
# Train the model
rf_model.fit(x_train, y_train)

# Make predictions on the test data using the trained model
y_predict = rf_model.predict(x_test)

# Calculate the accuracy of the model by comparing predicted labels with true labels
score = accuracy_score(y_predict, y_test)
print('{}% of right-hand samples were classified correctly!'.format(score * 100))


# Create an Isolation Forest model to identify outliers
iso_model = IsolationForest(contamination=0.1)
iso_model.fit(data)


# Save the trained models to a pickle files
f = open('train_classifiers/right_hand_rf_model.p', 'wb')
pickle.dump({'model': rf_model}, f)
print('train_classifiers/right_hand_rf_model.p was created')
f.close()

f = open('train_classifiers/right_hand_iso_model.p', 'wb')
pickle.dump({'model': iso_model}, f)
print('train_classifiers/right_hand_iso_model.p was created')
f.close()




# Train left hand classifier

# Load dataset
leftHand_data_dict = pickle.load(open('./train_classifiers/left_hand_data.pickle', 'rb'))

data = np.asarray(leftHand_data_dict['data'])
labels = np.asarray(leftHand_data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier()
# Train the model
rf_model.fit(x_train, y_train)

# Make predictions on the test data using the trained model
y_predict = rf_model.predict(x_test)

# Calculate the accuracy of the model by comparing predicted labels with true labels
score = accuracy_score(y_predict, y_test)
print('{}% of left-hand samples were classified correctly!'.format(score * 100))


# Create an Isolation Forest model to identify outliers
iso_model = IsolationForest(contamination=0.1)
iso_model.fit(data)


# Save the trained models to a pickle files
f = open('train_classifiers/left_hand_rf_model.p', 'wb')
pickle.dump({'model': rf_model}, f)
print('train_classifiers/left_hand_rf_model.p was created')
f.close()

f = open('train_classifiers/left_hand_iso_model.p', 'wb')
pickle.dump({'model': iso_model}, f)
print('train_classifiers/left_hand_iso_model.p was created')
f.close()