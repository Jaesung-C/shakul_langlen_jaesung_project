# main.py

from functions.stochastic_solver import generate_data
import numpy as np
from tqdm import tqdm


time_step = 1e-6

# Main execution
if __name__ == "__main__":
    # Parameters
    num_samples_per_pair = 1
    sampling_num = 1000  # Sampling interval for data generation
    
    # Generate data for each system type ['B', 'OSC', 'SSS']
    system_types = ['B', 'OSC', 'SSS']
    V_set = [1000, 10000]
    
    for system_type in system_types:
        for V in V_set:
            print(f'Generating data for system type: {system_type}, V = {V}')
            generate_data(system_type, num_samples_per_pair, sampling_num, V, time_step)
    
    print('Data generation complete for all system types.')
