from .stochastic_solver import *
from .deterministic_solver import *
import time
import os

def generate_sample_data(args):
    system_type, V, sampling_step, time_step, beta, epsilon, parm_idx, sample_idx, solver_type = args

    # Handle 'inf' value for V when creating directory names
    V_dir_name = 'inf' if V == float('inf') else f'{V}'
    
    # Generate sample data
    sample_data = generate_data(solver_type, sampling_step, V, time_step, beta, epsilon)
    
    # Create the directory path including solver_type
    base_dir = f'./data_set/{system_type}/{solver_type}/V{V_dir_name}/parm{parm_idx}/'
    os.makedirs(base_dir, exist_ok=True)
    
    # Save the sample data
    sample_file_path = os.path.join(base_dir, f'sample{sample_idx}.npy')    
    np.save(sample_file_path, {'samples': sample_data, 'beta': beta, 'epsilon': epsilon})
    
    return f'Sample {sample_idx} (beta={beta}, epsilon={epsilon}) saved for {system_type}/{solver_type}, V = {V}'


def generate_data(solver_type, sampling_step, V, time_step, beta, epsilon):
    sampling_interval = int(round((10 / time_step) / sampling_step))
    seed = int(time.time() * 1000) % (2**32 - 1)
    
    if V == float('inf'):
        # Deterministic solvers based on solver_type
        if solver_type == 'euler':
            x, y, z = solve_system_euler(beta, epsilon, time_step, 200, 
                                         V0=2, V1=2, VM2=6, k2=0.1, VM3=20, kx=0.3, ky=0.2, kz=0.1, 
                                         VM5=30, k5=1, kd=0.6, V4=2.5, k=10, kf=1, m=4, n=2, p=1)
        elif solver_type == 'rk4':
            # Implement RK4 method solver (assume it's not stochastic)
            x, y, z = solve_system_rk4(beta, epsilon, time_step, 200, 
                                       V0=2, V1=2, VM2=6, k2=0.1, VM3=20, kx=0.3, ky=0.2, kz=0.1, 
                                       VM5=30, k5=1, kd=0.6, V4=2.5, k=10, kf=1, m=4, n=2, p=1)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}")
    else:
        # Stochastic solvers based on solver_type
        if solver_type == 'euler':
            # x, y, z = solve_system(beta, epsilon, V, time_step)
            x, y, z = solve_system_seuler(beta, epsilon, time_step, 200, 
                                         V0=2, V1=2, VM2=6, k2=0.1, VM3=20, kx=0.3, ky=0.2, kz=0.1, 
                                         VM5=30, k5=1, kd=0.6, V4=2.5, k=10, kf=1, m=4, n=2, p=1, V=V, seed=seed)
        elif solver_type == 'srk':
            x, y, z = solve_system_srk(beta, epsilon, time_step, 200, 
                                       V0=2, V1=2, VM2=6, k2=0.1, VM3=20, kx=0.3, ky=0.2, kz=0.1, 
                                       VM5=30, k5=1, kd=0.6, V4=2.5, k=10, kf=1, m=4, n=2, p=1, V=V, seed=seed)
        elif solver_type == 'heun':
            x, y, z = solve_system_heun(beta, epsilon, time_step, 200, 
                                        V0=2, V1=2, VM2=6, k2=0.1, VM3=20, kx=0.3, ky=0.2, kz=0.1, 
                                        VM5=30, k5=1, kd=0.6, V4=2.5, k=10, kf=1, m=4, n=2, p=1, V=V, seed=seed)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}")

    np.random.seed(seed)
    start_time = 100 + np.random.rand() * 90
    end_time = start_time + 10
    start_idx = int(start_time / time_step)
    end_idx = int(end_time / time_step)
    
    sample = np.column_stack((x[start_idx:end_idx:sampling_interval], 
                              y[start_idx:end_idx:sampling_interval], 
                              z[start_idx:end_idx:sampling_interval]))
    
    return sample

def load_beta_epsilon(system_type):
    if system_type == 'B':
        data = np.loadtxt('./data_set/BetaEpsilon/BurstingPoints.dat')
    elif system_type == 'OSC':
        data = np.loadtxt('./data_set/BetaEpsilon/OSCPoints.dat')
    elif system_type == 'SSS':
        data = np.loadtxt('./data_set/BetaEpsilon/SSSPoints.dat')
    elif system_type == 'Bosc':
        data = np.loadtxt('./data_set/BetaEpsilon/BoscPoints.dat')
    else:
        raise ValueError('Invalid system type. Choose B, OSC, or SSS.')
    
    return data

# def generate_data_old(system_type, num_samples_per_pair, sampling_num, V, time_step, epsilon, beta):

#     sampling_interval = int(round((10 / time_step) / sampling_num))
    
#     x, y, z = solve_system(beta, epsilon, V, time_step)
    
#     start_time = 100 + np.random.rand() * 90
#     end_time = start_time + 10
#     start_idx = int(start_time / time_step)
#     end_idx = int(end_time / time_step)
    
#     sample = np.column_stack((x[start_idx:end_idx:sampling_interval], y[start_idx:end_idx:sampling_interval], z[start_idx:end_idx:sampling_interval]))
    
#     return sample

# def generate_sample_data_old(args):
#     system_type, V, sampling_num, time_step, beta, epsilon, parm_idx, sample_idx = args
#     sample_data = generate_data_old(system_type, 1, sampling_num, V, time_step, beta, epsilon)
#     base_dir = f'./data_set/{system_type}/euler/V{V}/parm{parm_idx}/'

#     os.makedirs(base_dir, exist_ok=True)
    
#     sample_file_path = os.path.join(base_dir, f'sample{sample_idx}.npy')    
#     np.save(sample_file_path, {'samples': sample_data, 'beta': beta, 'epsilon': epsilon})
    
#     return f'Sample {sample_idx} (beta={beta}, epsilon={epsilon}) saved for {system_type}, V = {V}'
