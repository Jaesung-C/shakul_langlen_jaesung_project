import numpy as np
from sklearn.preprocessing import StandardScaler

def fourier_sel(data, top_k=10):
    """Selects top_k Fourier components based on magnitude after Fourier Transform"""
    transformed_data = np.fft.fft(data, axis=1)
    magnitude = np.abs(transformed_data)
    top_k_data = []
    
    for i in range(data.shape[0]):
        top_indices = np.argsort(magnitude[i])[-top_k:]
        top_components = np.concatenate([np.real(transformed_data[i, top_indices]), 
                                         np.imag(transformed_data[i, top_indices])])
        top_k_data.append(top_components)
        
    return np.array(top_k_data)

def standardization(data):
    """Applies standardization to make data have mean 0 and standard deviation 1"""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def average_pooling(data, window_size=5):
    """Extracts local features by applying average pooling with a specified window size"""
    pooled_data = []
    for sample in data:
        pooled_sample = [np.mean(sample[i:i + window_size]) for i in range(0, len(sample), window_size)]
        pooled_data.append(pooled_sample)
    return np.array(pooled_data)

def label_merging_process(labels):
    """
    Merges stability labels by removing numeric prefixes.
    For example: ['1OSC', '2OSC', '3OSC'] -> ['OSC', 'OSC', 'OSC']
    
    Args:
        labels: array-like, original labels (e.g., '1OSC', '2SSS', etc.)
    
    Returns:
        merged_labels: array-like, labels with numeric prefixes removed
    """    
    # Remove numeric prefix from each label using string filter
    merged_labels = np.array([''.join(filter(str.isalpha, str(label))) for label in labels])
    
    return merged_labels