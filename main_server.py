import multiprocessing as mp
from functions.module_gen import load_beta_epsilon, generate_sample_data
from tqdm import tqdm

if __name__ == "__main__":
    time_step = 1e-6 #dont't change
    total_samples = 200
    sampling_step = 1000
    
    solver_type = 'euler'  # Declare solver_type here
    system_types = ['OSC'] 
    # ['B', 'OSC', 'SSS', 'Bosc']
    V_set = [float('inf')]  # Use float('inf') to represent infinite V
    
    config_list = []
    
    for system_type in system_types:
        epsilon_beta_data = load_beta_epsilon(system_type)
        for V in V_set:
            for parm_idx, (beta, epsilon) in enumerate(epsilon_beta_data, start=1):
                for sample_idx in range(1, total_samples + 1):
                    config_list.append((system_type, V, sampling_step, time_step, beta, epsilon, parm_idx, sample_idx, solver_type))
    
    num_cpus = mp.cpu_count() // 2
    
    with mp.Pool(num_cpus) as pool:
        results = list(tqdm(pool.imap(generate_sample_data, config_list),
                            total=len(config_list),
                            desc="Overall Progress"))
    
    for result in results:
        print(result)
    
    print('Data generation complete for all system types.')
