# main_server.py

import multiprocessing as mp
from functions.module_gen import generate_sample_data_old, generate_data_old, load_beta_epsilon
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    time_step = 1e-8
    total_samples = 100
    sampling_num = 1000
    
    system_types = ['B']  
    # ['B', 'OSC', 'SSS']
    V_set = [10**8]  
    # [1000 - strong fluctuation, 100000 - moderate fluctuations, 100000000 - almost deterministic limit]
    # 10^5~10^6 is interesting from a biophysics perspective.
    
    config_list = []
    
    for system_type in system_types:
        epsilon_beta_data = load_beta_epsilon(system_type)
        for V in V_set:
            for parm_idx, (beta, epsilon) in enumerate(epsilon_beta_data, start=1):
                for sample_idx in range(1, total_samples + 1):
                    config_list.append((system_type, V, sampling_num, time_step, epsilon, beta, parm_idx, sample_idx))
    
    num_cpus = mp.cpu_count() // 2
    
    with mp.Pool(num_cpus) as pool:
        results = list(tqdm(pool.imap(generate_sample_data_old, config_list),
                            total=len(config_list),
                            desc="Overall Progress"))
    
    for result in results:
        print(result)
    
    print('Data generation complete for all system types.')
