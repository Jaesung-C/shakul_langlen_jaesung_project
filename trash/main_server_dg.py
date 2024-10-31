# main_server.py

import multiprocessing as mp
from trash.dg import generate_data
import numpy as np
from tqdm import tqdm
import os

def load_epsilon_beta(system_type):
    if system_type == 'B':
        data = np.loadtxt('./data_set/EpsilonBeta/BurstingPoints.dat')
    elif system_type == 'OSC':
        data = np.loadtxt('./data_set/EpsilonBeta/OSCPoints.dat')
    elif system_type == 'SSS':
        data = np.loadtxt('./data_set/EpsilonBeta/SSSPoints.dat')
    else:
        raise ValueError('Invalid system type. Choose B, OSC, or SSS.')
    
    return data

def generate_sample_data(args):
    system_type, V, sampling_num, time_step, epsilon, beta, parm_idx, sample_idx = args
    sample_data = generate_data(system_type, 1, sampling_num, V, time_step, epsilon, beta)
    base_dir = f'./data_set/{system_type}/V{V}/parm{parm_idx}/'

    os.makedirs(base_dir, exist_ok=True)
    
    sample_file_path = os.path.join(base_dir, f'sample{sample_idx}.npy')    
    np.save(sample_file_path, {'samples': sample_data, 'epsilon': epsilon, 'beta': beta})
    
    return f'Sample {sample_idx} (epsilon={epsilon}, beta={beta}) saved for {system_type}, V = {V}'

if __name__ == "__main__":
    time_step = 1e-6
    total_samples = 1
    sampling_num = 1000
    
    system_types = ['B']  # ['B', 'OSC', 'SSS']
    V_set = [100]  # [1000, 10000]
    
    config_list = []
    
    for system_type in system_types:
        epsilon_beta_data = load_epsilon_beta(system_type)
        for V in V_set:
            for parm_idx, (epsilon, beta) in enumerate(epsilon_beta_data, start=1):
                for sample_idx in range(1, total_samples + 1):
                    config_list.append((system_type, V, sampling_num, time_step, epsilon, beta, parm_idx, sample_idx))
    
    results = []
    for config in tqdm(config_list, desc="Overall Progress"):
        result = generate_sample_data(config)
        results.append(result)
    
    for result in results:
        print(result)
    
    print('Data generation complete for all system types.')
