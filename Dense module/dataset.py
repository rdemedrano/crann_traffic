"""
DATASET MODULE
"""
import torch

import os
import numpy as np

from utils import prepare_datasets

def data_transform(opt):
    """
    From raw data to train, val and test dataset correctly normalised and min-max values to undo
    the transformation
    """
    data = torch.Tensor(np.genfromtxt(os.path.join('..',opt.datadir, opt.dataset)))
    
    X_train_time, Y_train_time, X_val_time, Y_val_time, X_test_time, Y_test_time, min_value_time, max_value_time, \
    X_train_space, Y_train, X_val_space, Y_val, X_test_space, Y_test, min_value_space, max_value_space, \
    X_train_exo, X_val_exo, X_test_exo, min_value_exo, max_value_exo = \
            prepare_datasets(data, opt.n_inp_sp, opt.n_out_sp, opt.n_inp_tem, opt.n_exo, opt.dim_x, opt.dim_y)
    
    return (X_train_time, Y_train_time, X_val_time, Y_val_time, X_test_time, Y_test_time, min_value_time, max_value_time, \
            X_train_space, Y_train, X_val_space, Y_val, X_test_space, Y_test, min_value_space, max_value_space, \
            X_train_exo, X_val_exo, X_test_exo, min_value_exo, max_value_exo)