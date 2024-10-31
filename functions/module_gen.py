from .stochastic_solver import *
from .deterministic_solver import *
import time
import os

def generate_sample_data(args):
    system_type, V, sampling_step, time_step, beta, epsilon, parm_idx, sample_idx, solver_type = args

    # Handle 'inf' value for V when creating directory names
    V_dir_name = 'inf' if V == float('inf') else f'{V}'
    
    # Generate sample data with system_type parameter
    sample_data = generate_data(solver_type, sampling_step, V, time_step, beta, epsilon, system_type)
    
    # Create the directory path including solver_type
    base_dir = f'./data_set/{system_type}/{solver_type}/V{V_dir_name}/parm{parm_idx}/'
    os.makedirs(base_dir, exist_ok=True)
    
    # Save the sample data
    sample_file_path = os.path.join(base_dir, f'sample{sample_idx}.npy')    
    np.save(sample_file_path, {'samples': sample_data, 'beta': beta, 'epsilon': epsilon})
    
    return f'Sample {sample_idx} (beta={beta}, epsilon={epsilon}) saved for {system_type}/{solver_type}, V = {V}'


def generate_data(solver_type, sampling_step, V, time_step, beta, epsilon, system_type):
    sampling_interval = int(round((10 / time_step) / sampling_step))
    seed = int(time.time() * 1000) % (2**32 - 1)
    
    # Get parameters based on system type
    params = get_system_parameters(system_type)
    
    if V == float('inf'):
        # Deterministic solvers
        if solver_type == 'euler':
            x, y, z = solve_system_euler(beta, epsilon, time_step, 200, **params)
        elif solver_type == 'rk4':
            x, y, z = solve_system_rk4(beta, epsilon, time_step, 200, **params)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}")
    else:
        # Stochastic solvers
        if solver_type == 'euler':
            x, y, z = solve_system_seuler(beta, epsilon, time_step, 200, V=V, seed=seed, **params)
        elif solver_type == 'srk':
            x, y, z = solve_system_srk(beta, epsilon, time_step, 200, V=V, seed=seed, **params)
        elif solver_type == 'heun':
            x, y, z = solve_system_heun(beta, epsilon, time_step, 200, V=V, seed=seed, **params)
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

def get_system_parameters(system_type):
    """
    Get system parameters based on the first digit of system_type.
    Args:
        system_type (str): System type identifier (e.g., '1B', '2OSC', '3PD')
    Returns:
        dict: Dictionary containing system parameters
    """
    system_class = int(system_type[0])  # Get first digit
    
    if system_class == 1:
        return {
            'V0': 2, 'V1': 2, 'VM2': 6, 'k2': 0.1, 'VM3': 20,
            'kx': 0.3, 'ky': 0.2, 'kz': 0.1, 'VM5': 30,
            'k5': 1, 'kd': 0.6, 'V4': 2.5, 'k': 10,
            'kf': 1, 'm': 4, 'n': 2, 'p': 1
        }
    elif system_class == 2:
        return {
            'V0': 2, 'V1': 2, 'VM2': 6, 'k2': 0.1, 'VM3': 20,
            'kx': 0.5, 'ky': 0.2, 'kz': 0.2, 'VM5': 30,
            'k5': 0.3, 'kd': 0.5, 'V4': 5, 'k': 10,
            'kf': 1, 'm': 2, 'n': 4, 'p': 2
        }
    elif system_class == 3:
        return {
            'V0': 2, 'V1': 2, 'VM2': 6, 'k2': 0.1, 'VM3': 30,
            'kx': 0.6, 'ky': 0.3, 'kz': 0.1, 'VM5': 50,
            'k5': 0.3194, 'kd': 1, 'V4': 3, 'k': 10,
            'kf': 1, 'm': 2, 'n': 4, 'p': 1
        }
    else:
        raise ValueError(f"Invalid system class: {system_class}")
    
def load_beta_epsilon(system_type):
    if system_type == '1B':
        data = np.loadtxt('./data_set/BetaEpsilon/BurstingPoints.dat')
    elif system_type == '1OSC':
        data = np.loadtxt('./data_set/BetaEpsilon/OSCPoints.dat')
    elif system_type == '1SSS':
        data = np.loadtxt('./data_set/BetaEpsilon/SSSPoints.dat')
    elif system_type == '1Bosc':
        data = np.loadtxt('./data_set/BetaEpsilon/BoscPoints.dat')

    elif system_type == '2OSC':
        data = np.loadtxt('./data_set/BetaEpsilon/OSPointsP2.dat')    
    elif system_type == '2SSS':
        data = np.loadtxt('./data_set/BetaEpsilon/SSSPointsP2.dat')    
    elif system_type == '2QP':
        data = np.loadtxt('./data_set/BetaEpsilon/QPPointsP2.dat')    

    elif system_type == '3OSC':
        data = np.loadtxt('./data_set/BetaEpsilon/OSPointsP3.dat')    
    elif system_type == '3SSS':
        data = np.loadtxt('./data_set/BetaEpsilon/SSSPointsP3.dat')    
    elif system_type == '3PD':
        data = np.loadtxt('./data_set/BetaEpsilon/PDPointsP3.dat')    
    elif system_type == '3Chaos':
        data = np.loadtxt('./data_set/BetaEpsilon/ChaosPointsP3.dat')    

    else:
        raise ValueError('Invalid system type.')
    
    return data