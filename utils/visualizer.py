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
# def visualize_misclassified(test_data, test_labels, predictions, n_samples=9):
#     misclassified_indices = np.where(test_labels != predictions)[0]
    
#     if len(misclassified_indices) == 0:
#         print("No misclassified samples to visualize.")
#         return
    
#     misclassified_dict = defaultdict(list)
#     for idx in misclassified_indices:
#         key = (test_labels[idx], predictions[idx])
#         misclassified_dict[key].append(idx)
    
#     selected_indices = []
#     for key in misclassified_dict:
#         selected_indices.extend(misclassified_dict[key][:n_samples // len(misclassified_dict)])
    
#     selected_indices = selected_indices[:n_samples]
    
#     plt.figure(figsize=(10, 10))
#     for i, idx in enumerate(selected_indices):
#         plt.subplot(3, 3, i + 1)
#         plt.plot(test_data[idx])
#         plt.title(f"True: {test_labels[idx]}, Pred: {predictions[idx]}")
    
#     plt.tight_layout()
#     plt.show()

def visualize_misclassified(test_data, test_labels, predictions, samples_per_case=4):
    """
    Visualizes misclassified samples with separate figures for each type of misclassification.
    
    Args:
        test_data: array-like, test data samples
        test_labels: array-like, true labels
        predictions: array-like, predicted labels
        samples_per_case: int, number of samples to show for each misclassification type
    """
    
    # Find misclassified samples
    misclassified_indices = np.where(test_labels != predictions)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found.")
        return
    
    # Group misclassified samples by their (true_label, predicted_label) pairs
    misclassified_dict = defaultdict(list)
    for idx in misclassified_indices:
        key = (test_labels[idx], predictions[idx])
        misclassified_dict[key].append(idx)
    
    # Calculate grid size based on samples_per_case
    n_cols = min(samples_per_case, 4)  # Max 4 columns
    n_rows = (samples_per_case - 1) // n_cols + 1
    
    # Create a figure for each type of misclassification
    for (true_label, pred_label), indices in misclassified_dict.items():
        # Select up to samples_per_case samples for this case
        case_indices = indices[:samples_per_case]
        
        # Create figure and title
        plt.figure(figsize=(15, 3))
        plt.suptitle(f'Misclassified Samples: True {true_label} → Predicted {pred_label}\n'
                    f'Total {len(indices)} cases', fontsize=16)
        
        # Plot each sample
        for i, idx in enumerate(case_indices):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Plot the time series
            plt.plot(test_data[idx], 'b-', label='Signal')
            
            # Add grid and legend
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Sample {idx}')
            
            # Add confidence scores if classifier provides them
            if hasattr(predictions, 'predict_proba'):
                proba = predictions.predict_proba([test_data[idx]])[0]
                confidence = proba[np.where(predictions.classes_ == pred_label)[0]][0]
                plt.title(f'Sample {idx}\nConfidence: {confidence:.2f}')
        
        plt.tight_layout()
        
        # Add summary statistics
        plt.figtext(0.02, 0.02, 
                   f'Number of samples shown: {len(case_indices)}/{len(indices)}\n'
                   f'Percentage of total misclassifications: {len(indices)/len(misclassified_indices)*100:.1f}%',
                   fontsize=10)
    
    # Create summary figure with confusion matrix-like visualization
    plt.figure(figsize=(10, 6))
    plt.title('Misclassification Distribution', fontsize=14)
    
    # Create bar plot of misclassification counts
    cases = list(misclassified_dict.keys())
    counts = [len(indices) for indices in misclassified_dict.values()]
    
    y_pos = np.arange(len(cases))
    plt.barh(y_pos, counts)
    plt.yticks(y_pos, [f'{true}->{pred}' for true, pred in cases])
    
    plt.xlabel('Number of Misclassifications')
    plt.ylabel('True -> Predicted Label')
    
    # Add percentage labels on bars
    total_misclassified = len(misclassified_indices)
    for i, count in enumerate(counts):
        plt.text(count, i, f' {count/total_misclassified*100:.1f}%', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nMisclassification Summary:")
    print(f"Total test samples: {len(test_labels)}")
    print(f"Total misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/len(test_labels)*100:.1f}%)")
    print("\nBreakdown by type:")
    for (true_label, pred_label), indices in misclassified_dict.items():
        print(f"{true_label} → {pred_label}: {len(indices)} cases ({len(indices)/len(misclassified_indices)*100:.1f}%)")


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