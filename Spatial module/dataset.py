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
    X_train, Y_train, X_val, Y_val, X_test, Y_test, min_value, max_value = prepare_datasets(data, opt.n_inp, 
                                                                                        opt.n_out, opt.dim_x, opt.dim_y)
    
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, min_value, max_value)