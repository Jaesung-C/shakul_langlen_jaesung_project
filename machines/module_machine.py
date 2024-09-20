import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Function to train and evaluate Random Forest
def evaluate_random_forest(train_data, train_labels, test_data, test_labels):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    cm = confusion_matrix(test_labels, predictions, labels=np.unique(train_labels))
    return cm, clf, predictions