from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Data loading function
# def load_data(system_types, V, max_samples=None):
#     data = []
#     labels = []
#     beta_epsilon_pairs = []
#     for system_type in system_types:
#         if np.isinf(V):
#             base_dir = f'./data_set/{system_type}/euler/Vinf/'
#         else:
#             base_dir = f'./data_set/{system_type}/V{V}/'
#         for parm_dir in os.listdir(base_dir):
#             parm_path = os.path.join(base_dir, parm_dir)
#             sample_files = os.listdir(parm_path)
#             if max_samples is not None:
#                 sample_files = sample_files[:max_samples]
            
#             for sample_file in sample_files:
#                 sample_data = np.load(os.path.join(parm_path, sample_file), allow_pickle=True).item()
#                 x_data = sample_data['samples'][:, 0]
#                 beta = sample_data['beta']
#                 epsilon = sample_data['epsilon']
#                 data.append(x_data.flatten())
#                 labels.append(system_type)
#                 beta_epsilon_pairs.append((beta, epsilon))

#     return np.array(data), np.array(labels), np.array(beta_epsilon_pairs)
# Data loading function
def load_data(system_types, V, max_samples=None):
    data = []
    labels = []
    beta_epsilon_pairs = []
    for system_type in system_types:
        if np.isinf(V):
            base_dir = f'./data_set/{system_type}/euler/Vinf/'
        else:
            base_dir = f'./data_set/{system_type}/euler/V{V}/'
        for parm_dir in os.listdir(base_dir):
            parm_path = os.path.join(base_dir, parm_dir)
            sample_files = os.listdir(parm_path)
            if max_samples is not None:
                sample_files = sample_files[:max_samples]
            
            for sample_file in sample_files:
                sample_data = np.load(os.path.join(parm_path, sample_file), allow_pickle=True).item()
                x_data = sample_data['samples'][:, 0]
                beta = sample_data['beta']
                epsilon = sample_data['epsilon']
                data.append(x_data.flatten())

                # 'Bosc'를 'B'로 처리
                if system_type == 'Bosc':
                    labels.append('B')
                else:
                    labels.append(system_type)

                beta_epsilon_pairs.append((beta, epsilon))

    return np.array(data), np.array(labels), np.array(beta_epsilon_pairs)


def balance_data(data, labels, beta_epsilon_pairs):
    unique_labels = np.unique(labels)
    label_pair_counts = {}
    label_pair_indices = {}

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        
        label_beta_epsilon_pairs = beta_epsilon_pairs[label_indices]
        
        unique_pairs, pair_indices = np.unique(label_beta_epsilon_pairs, axis=0, return_inverse=True)
        
        label_pair_counts[label] = len(unique_pairs)
        label_pair_indices[label] = (label_indices, unique_pairs, pair_indices)
    
    min_pair_count = min(label_pair_counts.values())

    balanced_data, balanced_labels, balanced_pairs = [], [], []
    for label in unique_labels:
        label_indices, unique_pairs, pair_indices = label_pair_indices[label]
        
        selected_pairs = resample(unique_pairs, n_samples=min_pair_count, random_state=42, replace=False)
        
        for pair in selected_pairs:
            pair_indices_in_label = np.where(np.all(beta_epsilon_pairs[label_indices] == pair, axis=1))[0]
            balanced_data.extend(data[label_indices[pair_indices_in_label]])
            balanced_labels.extend(labels[label_indices[pair_indices_in_label]])
            balanced_pairs.extend(beta_epsilon_pairs[label_indices[pair_indices_in_label]])

    balanced_data = np.array(balanced_data)
    balanced_labels = np.array(balanced_labels)
    balanced_pairs = np.array(balanced_pairs)
    
    return balanced_data, balanced_labels, balanced_pairs




def split_data(data, labels, beta_epsilon_pairs, option="option1", test_size=0.5, balanced=False):
    if balanced:
        data, labels, beta_epsilon_pairs = balance_data(data, labels, beta_epsilon_pairs)
    
    # Option1: Random split
    if option == "option1":
        train_data, test_data, train_labels, test_labels, train_pairs, test_pairs = train_test_split(
            data, labels, beta_epsilon_pairs, test_size=test_size, random_state=42
        )
        return train_data, test_data, train_labels, test_labels, train_pairs, test_pairs
    
    # Option2: Split by beta_epsilon_pairs
    elif option == "option2":
        unique_pairs = np.unique(beta_epsilon_pairs, axis=0)
        num_train_pairs = int(len(unique_pairs) * (1 - test_size))
        train_pairs_indices = np.random.choice(len(unique_pairs), num_train_pairs, replace=False)
        train_pairs = unique_pairs[train_pairs_indices]

        train_data, test_data = [], []
        train_labels, test_labels = [], []
        train_pairs_list, test_pairs_list = [], []
        
        for i, pair in enumerate(beta_epsilon_pairs):
            if tuple(pair) in map(tuple, train_pairs):
                train_data.append(data[i])
                train_labels.append(labels[i])
                train_pairs_list.append(pair)
            else:
                test_data.append(data[i])
                test_labels.append(labels[i])
                test_pairs_list.append(pair)

        return (np.array(train_data), np.array(test_data), 
                np.array(train_labels), np.array(test_labels), 
                np.array(train_pairs_list), np.array(test_pairs_list))
    
    else:
        raise ValueError("Invalid option. Choose either 'option1' or 'option2'.")
