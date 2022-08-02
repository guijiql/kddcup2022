import numpy as np
import random
import torch
import os

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric + ': ' + str(round(value, 4))) + ' '
    return result_str

def make_dirs():
    for task in ['task1', 'task2', 'task3']:
        os.makedirs(f'./saved/{task}', exist_ok=True)
        os.makedirs(f'./config/{task}', exist_ok=True)
        os.makedirs(f'./cache/{task}', exist_ok=True)