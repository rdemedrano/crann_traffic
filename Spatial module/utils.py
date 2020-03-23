"""
UTILS MODULE
"""
import torch

import os
import numpy as np
from collections import defaultdict
import json

def prepare_datasets(data, n_inp, n_out, dim_x, dim_y, train_size = 0.6, val_size = 0.2):
    """
    Preparation of data with a train/val/test split as validation scheme
    """
    data_total = []
    n_data = len(data)-1 - (n_inp+n_out)
    indexes = np.arange(n_data)
    for index in indexes:
        data_total.append(data[index:index+(n_inp+n_out)])
        
    data_total = torch.cat(data_total, dim = 0).view(-1, (n_inp+n_out), dim_x,dim_y)
     
    len_train = int(train_size * len(indexes))
    len_val = int((val_size + train_size) * len(indexes))
    
    data_train = data_total[0:len_train]
    data_val = data_total[len_train:len_val]
    data_test = data_total[len_val:]
    
    data_train, data_val, data_test, min_value, max_value = normalize_data(data_train, data_val, data_test)
    
    X_train, Y_train = torch.split(data_train, n_inp, dim = 1)
    X_val, Y_val = torch.split(data_val, n_inp, dim = 1)
    X_test, Y_test = torch.split(data_test, n_inp, dim = 1)
    
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, min_value, max_value)
   
    

def normalize(data, min_data, max_data):
    """
    Min-Max normalization by spatial point
    """
    N,T,W,H = data.size()
    
    return ((data.view(-1,W*H) - min_data)/(max_data - min_data)).view(N,T,W,H)

def normalize_data(train_set, validation_set, test_set):
    """
    Datasets normalization. To avoid data leakage, only train set information is used in the process
    """
    N,T,W,H = train_set.size()
    
    min_values = torch.min(train_set.view(-1, W*H), 0).values
    max_values = torch.max(train_set.view(-1, W*H), 0).values
    
    train_set_norm = normalize(train_set, min_values, max_values)
    validation_set_norm = normalize(validation_set, min_values, max_values)
    test_set_norm = normalize(test_set, min_values, max_values)
    
    return (train_set_norm, validation_set_norm, test_set_norm, min_values, max_values)    



def denormalize_data(data, min_data, max_data):
    """
    Given normalization contants, undo the normalization
    """
    return min_data + data.mul(max_data - min_data)


def mae(x_pred, x_target, dim=0):
    """
    MAE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).abs().mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).abs().mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).abs().mean((0,2))
    else:
        raise ValueError("Not a valid dimension")

def rel_error(x_pred, x_target, dim=0):
    """
    WMAPE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return 100*(mae(x_pred, x_target, dim = dim)/(x_target.abs().mean())).item()
    elif dim == 1:
        return 100*(mae(x_pred, x_target, dim = 1)/(x_target.abs().mean((0,1))))
    elif dim == 2:
        return 100*(mae(x_pred, x_target)/(x_target.abs().mean((0,2))))
    else:
        raise ValueError("Not a valid dimension")


def rmse(x_pred, x_target, dim=0):
    """
    RMSE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).pow(2).mean().sqrt().item()
    elif dim == 1:
        return x_pred.sub(x_target).pow(2).mean((0,1)).sqrt().squeeze()
    elif dim == 2:
        return x_pred.sub(x_target).pow(2).mean((0,2)).sqrt().squeeze()
    else:
        raise ValueError("Not a valid dimension")

def bias(x_pred, x_target, dim=0):
    """
    Bias calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).mean((0,2))
    else:
        raise ValueError("Not a valid dimension")


class DotDict(dict):
    """
    Dot notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class Logger(object):
    """
    Log information through the process
    """
    def __init__(self, log_dir, chkpt_interval):
        super(Logger, self).__init__()
        os.makedirs(os.path.join(log_dir))
        self.log_path = os.path.join(log_dir, 'logs.json')
        self.model_path = os.path.join(log_dir, 'spatial_model.pth')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)