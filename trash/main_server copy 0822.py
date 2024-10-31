# main_server.py

import multiprocessing as mp
from functions.stochastic_solver import generate_data
import numpy as np
from tqdm import tqdm
import os

def generate_partial_data(args):
    system_type, V, num_samples, sampling_num, time_step, part_idx = args
    parts_dir = './data_set/parts/'  # 부분 데이터를 저장할 상위 폴더
    os.makedirs(parts_dir, exist_ok=True)
    part_file_path = os.path.join(parts_dir, f'{system_type}_V{V}_part{part_idx}.npy')
    
    print(f'Generating data for system type: {system_type}, V = {V}, Part: {part_idx}')
    partial_data = generate_data(system_type, num_samples, sampling_num, V, time_step)
    np.save(part_file_path, partial_data)
    return f'Part {part_idx} saved for {system_type}, V = {V}'

def combine_and_save(system_type, V, num_parts):
    parts_dir = './data_set/parts/'  # 부분 데이터가 저장된 폴더
    all_data = []
    
    for part_idx in range(num_parts):
        part_file_path = os.path.join(parts_dir, f'{system_type}_V{V}_part{part_idx}.npy')
        part_data = np.load(part_file_path)
        all_data.append(part_data)
        os.remove(part_file_path)  # 임시 파일 삭제
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_file_path = f'./data_set/{system_type}_V{V}.npy'
    np.save(combined_file_path, combined_data)
    print(f'Combined data saved for {system_type}, V = {V}')

if __name__ == "__main__":
    time_step = 1e-6
    total_samples = 100
    sampling_num = 1000
    num_parts = 20
    num_samples_per_part = total_samples // num_parts
    
    system_types = ['B']  # ['B', 'OSC', 'SSS']
    V_set = [10000] # [1000 10000]
    
    config_list = [(system_type, V, num_samples_per_part, sampling_num, time_step, part_idx)
                   for system_type in system_types
                   for V in V_set
                   for part_idx in range(num_parts)]
    
    num_cpus = mp.cpu_count()
    
    with mp.Pool(num_cpus) as pool:
        results = list(tqdm(pool.imap(generate_partial_data, config_list),
                            total=len(config_list),
                            desc="Overall Progress"))
    
    for result in results:
        print(result)
    
    # 각 system_type과 V에 대해 부분 데이터를 결합 및 저장
    for system_type in system_types:
        for V in V_set:
            combine_and_save(system_type, V, num_parts)
    
    print('Data generation and combination complete for all system types.')
