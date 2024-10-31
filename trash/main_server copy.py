# main_server.py

from functions.data_gen import generate_data
import numpy as np
from multiprocessing import Pool
import os

time_step = 1e-6

def generate_data_for_config(args):
    system_type, V, num_samples_per_pair, sampling_num, time_step = args
    print(f'Generating data for system type: {system_type}, V = {V}')
    generate_data(system_type, num_samples_per_pair, sampling_num, V, time_step)

if __name__ == "__main__":
    # Parameters
    num_samples_per_pair = 20 # 2000
    sampling_num = 1000
    
    # Generate data for each system type
    system_types = ['B'] # ['B', 'OSC', 'SSS']
    V_set = [10000]
    
    # Create a list of all configurations
    config_list = [(system_type, V, num_samples_per_pair, sampling_num, time_step)
                   for system_type in system_types for V in V_set]
    
    # Use multiprocessing to parallelize data generation
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    with Pool(num_cpus) as pool:
        pool.map(generate_data_for_config, config_list)
    
    print('Data generation complete for all system types.')
