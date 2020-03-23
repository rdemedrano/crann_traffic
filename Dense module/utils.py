"""
UTILS MODULE
"""
import torch

import os
import numpy as np
from collections import defaultdict
import json

def prepare_datasets(data, n_img, n_pred, n_timesteps, n_exogenous, 
                     dim_x, dim_y, train_size = 0.6, val_size=0.2):
    """
    Preparation of data with a train/val/test split as validation scheme
    """
    data_time = []
    data_space = []
    data_exo = []
    
    n_data = len(data)-1 - (n_timesteps+n_pred)
    indexes1 = np.arange(n_data)
    indexes2 = indexes1 + (n_timesteps + n_pred) - (n_img + n_pred)
    
    for index1,index2 in zip(indexes1, indexes2):
        data_time.append(data[index1:index1+(n_timesteps+n_pred), dim_x*dim_y])
        data_space.append(data[index2:index2+(n_img+n_pred), 0:dim_x*dim_y]) 
        data_exo.append(data[(index2+n_img):index2+(n_img+n_pred), (dim_x*dim_y+1):]) 
        
    data_time = torch.cat(data_time, dim = 0).view(-1, (n_timesteps+n_pred))
    data_space = torch.cat(data_space, dim = 0).view(-1, (n_img+n_pred), dim_x,dim_y)
    data_exo = torch.cat(data_exo, dim = 0).view(-1, n_pred, n_exogenous)
    
    len_train = int(train_size * len(indexes1))
    len_val = int((val_size + train_size) * len(indexes1))
    
    data_train_time = data_time[0:len_train]
    data_val_time = data_time[len_train:len_val]
    data_test_time = data_time[len_val:]
    
    data_train_space = data_space[0:len_train]
    data_val_space = data_space[len_train:len_val]
    data_test_space = data_space[len_val:]
    
    data_train_exo = data_exo[0:len_train]
    data_val_exo = data_exo[len_train:len_val]
    data_test_exo = data_exo[len_val:]
    
    
    data_train_time, data_val_time, data_test_time, min_value_time, max_value_time =\
                    normalize_data(data_train_time, data_val_time, data_test_time)
    data_train_space, data_val_space, data_test_space, min_value_space, max_value_space =\
                    normalize_data(data_train_space, data_val_space, data_test_space, dimension = "space")                
    data_train_exo, data_val_exo, data_test_exo, min_value_exo, max_value_exo =\
                    normalize_data(data_train_exo, data_val_exo, data_test_exo, dimension = "exo")
  

    X_train_time, Y_train_time = torch.split(data_train_time, n_timesteps, dim = 1)
    X_val_time, Y_val_time = torch.split(data_val_time, n_timesteps, dim = 1)
    X_test_time, Y_test_time = torch.split(data_test_time, n_timesteps, dim = 1)
    
    X_train_space, Y_train = torch.split(data_train_space, n_img, dim = 1)
    X_val_space, Y_val = torch.split(data_val_space, n_img, dim = 1)
    X_test_space, Y_test = torch.split(data_test_space, n_img, dim = 1)
    
    X_train_exo = data_train_exo
    X_val_exo = data_val_exo
    X_test_exo = data_test_exo
    
    return (X_train_time, Y_train_time, X_val_time, Y_val_time, X_test_time, Y_test_time, min_value_time, max_value_time, \
            X_train_space, Y_train, X_val_space, Y_val, X_test_space, Y_test, min_value_space, max_value_space, \
            X_train_exo, X_val_exo, X_test_exo, min_value_exo, max_value_exo)



def normalize(data, min_data, max_data, dimension = "time"):
    """
    Min-Max normalization by dimension
    """
    if dimension == "time":
        return (data - min_data)/(max_data - min_data)
    elif dimension == "space":
        N,T,W,H = data.size()
        return ((data.view(-1,W*H) - min_data)/(max_data - min_data)).view(N,T,W,H)
    elif dimension == "exo":
        N, T, V = data.size()
        return ((data.view(-1,V) - min_data)/(max_data - min_data)).view(N,T,V)
    else:
        raise ValueError("No possible dimension found")
    
    
def normalize_data(train_set, validation_set, test_set, dimension = "time"):
    """
    Datasets normalization by dimension. To avoid data leakage, only train set information is used in the process
    """
    if dimension == "time":
        min_value = torch.min(train_set)
        max_value = torch.max(train_set)
    elif dimension == "space":
        N,T,W,H = train_set.size()
        min_value = torch.min(train_set.view(-1, W*H), 0).values
        max_value = torch.max(train_set.view(-1, W*H), 0).values
    elif dimension == "exo":
        N, T, V = train_set.size()
        min_value = torch.min(train_set.view(-1, V), 0).values
        max_value = torch.max(train_set.view(-1, V), 0).values
    else:
        raise ValueError("No possible dimension found")
        
    train_set_norm = normalize(train_set, min_value, max_value, dimension)
    validation_set_norm = normalize(validation_set, min_value, max_value, dimension)
    test_set_norm = normalize(test_set, min_value, max_value, dimension)

    
    return (train_set_norm, validation_set_norm, test_set_norm, min_value, max_value)


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


def evaluate_temp_att(encoder, decoder, batch, n_pred, device):
     """
     Inference of temporal attention mechanism
     """
     output = torch.Tensor().to(device)
     
     h = encoder.init_hidden(batch.size(0))
            
     encoder_output, h = encoder(batch,h)
     decoder_hidden = h
     decoder_input = torch.zeros(batch.size(0), 1, device = device)
            
     for k in range(n_pred):
         decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
         decoder_input = decoder_output
         output = torch.cat((output, decoder_output),1)

     return output

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
        self.model_path = os.path.join(log_dir, 'spatial_model.pt')
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