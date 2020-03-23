"""
UTILS MODULE
"""
import torch

import os
import numpy as np
from collections import defaultdict
import json

def prepare_datasets(data, n_seq, n_pred, train_size = 0.6, val_size = 0.2):
    """
    Preparation of data with a train/val/test split as validation scheme
    """
    data_total = []
    n_data = len(data)-1 - (n_seq+n_pred)
    indexes = np.arange(0,n_data)
    for index in indexes:
        data_total.append(data[index:index+(n_seq+n_pred)])
        
        
    data_total = torch.cat(data_total, dim = 0).view(-1, (n_seq+n_pred))
        
    len_train = int(train_size * len(indexes))
    len_val = int((val_size + train_size) * len(indexes))
    
    data_train = data_total[0:len_train]
    data_val = data_total[len_train:len_val]
    data_test = data_total[len_val:]
    
    data_train, data_val, data_test, min_value, max_value = normalize_data(data_train, data_val, data_test)
    
    X_train, Y_train = torch.split(data_train, n_seq, dim = 1)
    X_val, Y_val = torch.split(data_val, n_seq, dim = 1)
    X_test, Y_test = torch.split(data_test, n_seq, dim = 1)
    
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, min_value, max_value)
   

def normalize(data, min_data, max_data):
    """
    Min-Max normalization
    """
    return (data - min_data)/(max_data - min_data)

def normalize_data(train_set, validation_set, test_set):
    """
    Datasets normalization. To avoid data leakage, only train set information is used in the process
    """
    min_value = torch.min(train_set)
    max_value = torch.max(train_set)
    
    train_set_norm = normalize(train_set, min_value, max_value)
    validation_set_norm = normalize(validation_set, min_value, max_value)
    test_set_norm = normalize(test_set, min_value, max_value)


    
    return (train_set_norm, validation_set_norm, test_set_norm, min_value, max_value)



def denormalize_data(data, min_data, max_data):
    """
    Given normalization contants, undo the normalization
    """
    return min_data + data.mul(max_data - min_data)

def rmse(x_pred, x_target, reduce=True):
    """
    RMSE calculation
    If reduce, overall RMSE
    Else, by spatial point
    """
    if reduce:
        return x_pred.sub(x_target).pow(2).mean().sqrt().item()
    return x_pred.sub(x_target).pow(2).mean(0).sqrt()

def bias(x_pred, x_target, reduce=True):
    """
    Bias calculation
    If reduce, overall bias
    Else, by spatial point
    """
    if reduce:
        return x_pred.sub(x_target).mean().item()
    return x_pred.sub(x_target).mean(0)

def mae(x_pred, x_target, reduce=True):
    """
    MAE calculation
    If reduce, overall MAE
    Else, by spatial point
    """
    if reduce:
        return x_pred.sub(x_target).abs().mean().item()
    return x_pred.sub(x_target).abs().mean(0)

def rel_error(x_pred, x_target, reduce = True):
    """
    Rel-err calculation
    If reduce, overall rel-err
    Else, by spatial point
    """
    if reduce:
        return 100*(mae(x_pred, x_target)/(x_target.abs().mean())).item()
    return 100*(mae(x_pred, x_target, reduce)/(x_target.abs().mean(0)))

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
        self.encoder_path = os.path.join(log_dir, 'encoder_model.pth')
        self.decoder_path = os.path.join(log_dir, 'decoder_model.pth')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model, which):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model, which)
        self.logs['epoch'] += 0.5

    def save(self, model, which):
        self.logs['epoch'] = int(self.logs['epoch'])
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.encoder_path if which == 'encoder' else self.decoder_path)
