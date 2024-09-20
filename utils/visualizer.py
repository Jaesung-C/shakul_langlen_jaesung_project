import numpy as np
from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Function to visualize misclassified samples
def visualize_misclassified(test_data, test_labels, predictions, n_samples=9):
    misclassified_indices = np.where(test_labels != predictions)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples to visualize.")
        return
    
    misclassified_dict = defaultdict(list)
    for idx in misclassified_indices:
        key = (test_labels[idx], predictions[idx])
        misclassified_dict[key].append(idx)
    
    selected_indices = []
    for key in misclassified_dict:
        selected_indices.extend(misclassified_dict[key][:n_samples // len(misclassified_dict)])
    
    selected_indices = selected_indices[:n_samples]
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indices):
        plt.subplot(3, 3, i + 1)
        plt.plot(test_data[idx])
        plt.title(f"True: {test_labels[idx]}, Pred: {predictions[idx]}")
    
    plt.tight_layout()
    plt.show()

# Function to visualize misclassified samples
def visualize_classified(test_data, test_labels, predictions, n_samples=9):
    classified_indices = np.where(test_labels == predictions)[0]
    
    if len(classified_indices) == 0:
        print("No classified samples to visualize.")
        return
    
    classified_dict = defaultdict(list)
    for idx in classified_indices:
        key = (test_labels[idx], predictions[idx])
        classified_dict[key].append(idx)
    
    selected_indices = []
    for key in classified_dict:
        selected_indices.extend(classified_dict[key][:n_samples // len(classified_dict)])
    
    selected_indices = selected_indices[:n_samples]
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indices):
        plt.subplot(3, 3, i + 1)
        plt.plot(test_data[idx])
        plt.title(f"True: {test_labels[idx]}, Pred: {predictions[idx]}")
    
    plt.tight_layout()
    plt.show()



def visualize_beta_epsilon_points(train_labels, test_labels, train_pairs, test_pairs, predictions):
    plt.figure(figsize=(12, 10))
    
    color_map = {'B': 'blue', 'OSC': 'green', 'SSS': 'yellow'}
    
    for system_type in ['B', 'OSC', 'SSS']:
        mask = train_labels == system_type
        plt.scatter(
            np.array(train_pairs)[mask, 0],
            np.array(train_pairs)[mask, 1], 
            c=color_map[system_type],
            label=f'{system_type} (Train)',
            alpha=0.6
        )
    
    for system_type in ['B', 'OSC', 'SSS']:
        mask = test_labels == system_type
        correct_mask = (test_labels == predictions) & mask
        incorrect_mask = (test_labels != predictions) & mask
        
        plt.scatter(
            np.array(test_pairs)[correct_mask, 0],  
            np.array(test_pairs)[correct_mask, 1], 
            c=color_map[system_type],
            marker='s',
            label=f'{system_type} (Test)',
            alpha=0.6
        )
        plt.scatter(
            np.array(test_pairs)[incorrect_mask, 0], 
            np.array(test_pairs)[incorrect_mask, 1],
            c='red',
            marker='s',
            label=f'{system_type} (Misclassified)' ,
            alpha=0.6
        )
    
    plt.xlabel('Beta')
    plt.ylabel('Epsilon')
    plt.title('Beta-Epsilon Points for Train and Test Data')
    plt.legend()
    plt.show()