# functions/solve_system.py

from numba import njit
import numpy as np
from tqdm import tqdm
import os

# Define the parameters
    # Define the parameters
V0, V1, V4 = 2, 2, 2.5
kf, k = 1, 10
VM2, k2 = 6, 0.1
VM3, m = 20, 4
kx, ky, kz = 0.3, 0.2, 0.1
VM5, k5 = 30, 1
kd, p, n = 0.6, 1, 2

# Define parameters
dt = 1e-6
tmax = 200
Nn = round(tmax / dt)

# Time array
tT = np.arange(0, tmax + dt, dt)

@njit
def solve_system(beta, epsilon, V):
    # Random variates
    xi = [np.random.normal(0, np.sqrt(dt), Nn) for _ in range(12)]
    # Initial conditions
    x = np.ones(Nn + 1)
    y = np.ones(Nn + 1)
    z = np.ones(Nn + 1)
    # Euler discretization
    for i in range(Nn):
        V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
        V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
        V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)
        x[i + 1] = (x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V0) * xi[0][i] + np.sqrt(V1 * beta) * xi[1][i] - np.sqrt(V2) * xi[2][i] + np.sqrt(V3) * xi[3][i] + np.sqrt(kf * y[i]) * xi[4][i] - np.sqrt(k) * xi[5][i]))
        y[i + 1] = (y[i] + (V2 - kf * y[i] - V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V2) * xi[6][i] - np.sqrt(V3) * xi[7][i] - np.sqrt(kf * y[i]) * xi[8][i]))
        z[i + 1] = (z[i] + (V4 * beta - epsilon * z[i] - V5) * dt + 1 / np.sqrt(V) * (np.sqrt(V4 * beta) * xi[9][i] - np.sqrt(V5) * xi[10][i] - np.sqrt(epsilon * z[i]) * xi[11][i]))
    return x, y, z


# @njit
# def solve_system_old(beta, epsilon, V, dt):
#     # Define the parameters
#     V0, V1, V4 = 2, 2, 2.5
#     kf, k = 1, 10
#     VM2, k2 = 6, 0.1
#     VM3, m = 20, 4
#     kx, ky, kz = 0.3, 0.2, 0.1
#     VM5, k5 = 30, 1
#     kd, p, n = 0.6, 1, 2
#     tmax = 200
#     Nn = round(tmax / dt)

#     xi = [np.random.normal(0, np.sqrt(dt), Nn) for _ in range(12)]
#     x, y, z = np.ones(Nn + 1), np.ones(Nn + 1), np.ones(Nn + 1)
#     for i in range(Nn):
#         V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
#         V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
#         V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)
#         x[i + 1] = (x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V0) * xi[0][i] + np.sqrt(V1 * beta) * xi[1][i] - np.sqrt(V2) * xi[2][i] + np.sqrt(V3) * xi[3][i] + np.sqrt(kf * y[i]) * xi[4][i] - np.sqrt(k) * xi[5][i]))
#         y[i + 1] = (y[i] + (V2 - kf * y[i] - V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V2) * xi[6][i] - np.sqrt(V3) * xi[7][i] - np.sqrt(kf * y[i]) * xi[8][i]))
#         z[i + 1] = (z[i] + (V4 * beta - epsilon * z[i] - V5) * dt + 1 / np.sqrt(V) * (np.sqrt(V4 * beta) * xi[9][i] - np.sqrt(V5) * xi[10][i] - np.sqrt(epsilon * z[i]) * xi[11][i]))
#     return x, y, z

def generate_data(system_type, num_samples_per_pair, sampling_num, V, dt, epsilon, beta):
    sampling_interval = int(round((10 / dt) / sampling_num))
    
    # 시스템 방정식을 해결하여 데이터를 생성 (예시 코드)
    x, y, z = solve_system(beta, epsilon, V)
    
    # 100~200초 사이에서 임의의 10초 구간 선택
    start_time = 100 + np.random.rand() * 90
    end_time = start_time + 10
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)
    
    # 샘플링 데이터 생성
    sample = np.column_stack((x[start_idx:end_idx:sampling_interval], y[start_idx:end_idx:sampling_interval], z[start_idx:end_idx:sampling_interval]))
    
    return sample


# def generate_data_old(system_type, num_samples_per_pair, sampling_num, V, dt, epsilon, beta, part_idx):
#     sampling_interval = int(round((10 / dt) / sampling_num))
    
#     for sample_idx in range(num_samples_per_pair):
#         x, y, z = solve_system(beta, epsilon, V, dt)
        
#         start_time = 100 + np.random.rand() * 90
#         end_time = start_time + 10
#         start_idx = int(start_time / dt)
#         end_idx = int(end_time / dt)
        
#         sample = np.column_stack((x[start_idx:end_idx:sampling_interval], y[start_idx:end_idx:sampling_interval], z[start_idx:end_idx:sampling_interval]))
        
#         base_dir = f'./data_set/{system_type}/V{V}/epsilon_{epsilon}_beta_{beta}/'
#         os.makedirs(base_dir, exist_ok=True)
        
#         sample_file_path = os.path.join(base_dir, f'part{part_idx}_sample{sample_idx}.npy')
        
#         np.save(sample_file_path, {'samples': sample, 'epsilon': epsilon, 'beta': beta})

# def generate_data_old2(system_type, num_samples_per_pair, sampling_num, V, dt):
#     # Load parameter file based on system type
#     if system_type == 'B':
#         data = np.loadtxt('./data_set/EpsilonBeta/BurstingPoints.dat')
#     elif system_type == 'OSC':
#         data = np.loadtxt('./data_set/EpsilonBeta/OSCPoints.dat')
#     elif system_type == 'SSS':
#         data = np.loadtxt('./data_set/EpsilonBeta/SSSPoints.dat')
#     else:
#         raise ValueError('Invalid system type. Choose B, OSC, or SSS.')
    
#     num_pairs = data.shape[0]
#     all_data = []
#     #sampling_interval =  int( (10 / dt) / sampling_num )
#     sampling_interval = int(round((10 / dt) / sampling_num))
#     for pair_idx in tqdm(range(num_pairs), desc=f"Processing {system_type}"):
#         beta, epsilon = data[pair_idx]
#         sample_data = []
        
#         for _ in range(num_samples_per_pair):
#             x, y, z = solve_system(beta, epsilon, V, dt)
            
#             # Select a random 10-second interval between 100 and 200 seconds
#             start_time = 100 + np.random.rand() * 90
#             end_time = start_time + 10
#             start_idx = int(start_time / dt)  # Use dt here
#             end_idx = int(end_time / dt)      # Use dt here
            
#             sample = np.column_stack((x[start_idx:end_idx:sampling_interval], y[start_idx:end_idx:sampling_interval], z[start_idx:end_idx:sampling_interval]))
#             sample_data.append(sample)
        
#         all_data.append({
#             'beta': beta,
#             'epsilon': epsilon,
#             'samples': sample_data
#         })
    
#     # Save the data
#     filename = f'./data_set/{system_type}_V{V}.npy'
#     np.save(filename, all_data)
#     print(f"Data saved to {filename}")