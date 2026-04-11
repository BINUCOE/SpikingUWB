import os
import numpy as np
import torch
from tqdm import tqdm
from time import time as t
import csv


def setup_device(cuda_name, gpu, seed):
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False

    torch.set_num_threads(os.cpu_count() - 1)
    print("Running on Device = ", device)
    return device


def create_directories(task_tpye, plot_name):
    directories = [
        f"{task_tpye}/{plot_name}",
        f"{task_tpye}/{plot_name}/weights",
        f"{task_tpye}/{plot_name}/performance",
        f"{task_tpye}/{plot_name}/assaiments",
        "model_weights/model",
        "model_weights/weight"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_accuracy_to_csv(plot_name, accuracy):
    acc_folder_path = 'acc'
    if not os.path.exists(acc_folder_path):
        os.makedirs(acc_folder_path)
    csv_file_name = os.path.join(acc_folder_path, f'{plot_name}_acc.csv')
    with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['All', 'Proportion'])
        writer.writerows(zip(accuracy["all"], accuracy["proportion"]))


def save_weight(plot_name, assignments, proportions):
    np.save(r'model_weights/weight/assignments_{}_last.npy'.format(plot_name), assignments.cpu().numpy())
    np.save(r'model_weights/weight/proportions_{}_last.npy'.format(plot_name), proportions.cpu().numpy())


def get_task_name(input_neuron):
    task_mapping = {
        100: '100_1',
        225: '225_2',
        400: '400_2',
        784: '784_2',
        900: '900_3',
        1600: '1600_4',
        2500: '2500_5',
        3600: '3600_6',
    }
    return task_mapping.get(input_neuron, 'unknown_task')
