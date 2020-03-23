"""
TRAINING MODULE
"""
import torch
import torch.optim as optim

import os
import numpy as np
import configargparse
import json
from datetime import datetime
from collections import defaultdict
from tqdm import trange

from dataset import data_transform
from utils import DotDict, Logger, denormalize_data, rmse, bias, rel_error
from sp_att_mech import AttentionCNN



"""
#######################################################################################################################
# VARIABLES AND OPTIONS
#######################################################################################################################
"""
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='sp_data.csv')
# -- exp
p.add('--outputdir', type=str, help='path to save exp', default='output/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
# -- model
p.add('--dim_x', type=int, help='x dimension for image creation', default=5)
p.add('--dim_y', type=int, help='x dimension for image creation', default=6)
p.add('--n_inp', type=int, help='number of input timesteps', default=24)
p.add('--n_out', type=int, help='number of output timesteps', default=24)
p.add('--n_points', type=int, help='number of spatial points/sensors', default=30)
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-2)
p.add('--beta1', type=float, default=.9, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-8, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=3.365e-4)
# -- learning
p.add('--batch_size', type=int, default=64, help='batch size')
p.add('--patience', type=int, default=10, help='number of epoch to wait before trigerring lr decay')
p.add('--n_epochs', type=int, default=200, help='number of epochs to train for')
# -- gpu
p.add('--device', type=int, default=0, help='-1: cpu; > -1: cuda device id')

# parse
opt = DotDict(vars(p.parse_args()))
if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



"""
#######################################################################################################################
# DATA PREPARATION
#######################################################################################################################
"""
X_train, Y_train, X_val, Y_val, X_test, Y_test, min_value, max_value = data_transform(opt)

train_dataset = []
for i in range(len(X_train)):
    train_dataset.append([X_train[i], Y_train[i]])
    
val_dataset = []
for i in range(len(X_val)):
    val_dataset.append([X_val[i], Y_val[i]]) 
    
test_dataset = []
for i in range(len(X_test)):
   test_dataset.append([X_test[i], Y_test[i]]) 
    

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = opt.batch_size,
                                           shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = len(X_val),
                                           shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = len(X_test),
                                           shuffle = False)



"""
#######################################################################################################################
# MODEL
#######################################################################################################################
"""
model = AttentionCNN(in_channels=opt.n_inp, out_channels=opt.n_out, dim_x=opt.dim_x, dim_y=opt.dim_y)
model.to(device)



"""
#######################################################################################################################
# OPTIMIZER
#######################################################################################################################
"""
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),  lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)

if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)



"""
#######################################################################################################################
# LOGGER
#######################################################################################################################
"""
logger = Logger(opt.outputdir, 25)
with open(os.path.join(opt.outputdir, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)



"""
#######################################################################################################################
# TRAINING AND VALIDATION
#######################################################################################################################
"""
lr = opt.lr
tr = trange(opt.n_epochs, position=0, leave=True)

for t in tr:     
    # Training
    model.train()
    logs_train = defaultdict(float)
    for i, (x,y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()  
        logs_train['mse'] += loss.item()
        
    # Logs training
    logs_train['mse'] /= (i+1)
    logger.log('train', logs_train)

    # Evaluation
    model.eval()
    logs_val = defaultdict(float)
    with torch.no_grad():
        for x,y in val_loader:
            if torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)
            else:
                x = x
                y = y
            y_pred, _ = model(x)
            loss_val = loss_fn(y_pred, y)
            logs_val['mse'] = loss_val.item()
            
        # Logs evaluation
        logger.log('val', logs_val)
        
    # General information
    tr.set_postfix(train_mse = logs_train['mse'], val_mse=logs_val['mse'], 
                   train_rmse = np.sqrt(logs_train['mse']), val_rmse=np.sqrt(logs_val['mse']),
                   lr = lr)
    logger.checkpoint(model)
    
    # Learning rate decay
    if opt.patience > 0:
        lr_scheduler.step(logs_val['mse'])
        lr = optimizer.param_groups[0]['lr']
    if lr <= 1e-5:
        break
    

"""
#######################################################################################################################
# TEST
#######################################################################################################################
"""
model.eval()
logs_test = defaultdict(float)
with torch.no_grad():        
    for x,y in test_loader:
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        y_pred, _ = model(x)
        
        y_pred_dnorm = denormalize_data(y_pred.view(-1, opt.n_inp, opt.n_points).cpu(), min_value, max_value)
        y_dnorm = denormalize_data(y.view(-1, opt.n_inp, opt.n_points).cpu(), min_value, max_value)
        
        loss_test = loss_fn(y_pred_dnorm, y_dnorm)
        
        logs_test['mse'] = loss_test.item()
        logs_test['rmse'] = np.sqrt(loss_test.item())
        logs_test['bias'] = bias(y_pred_dnorm, y_dnorm)
        logs_test['err-rel'] = rel_error(y_pred_dnorm, y_dnorm)
        
        logger.log('test', logs_test)
        
print("\n\n================================================")
print(" *  Test MSE: ", logs_test['mse'],
      "\n *  Test RMSE: ", logs_test['rmse'],
      "\n *  Test Bias: ", logs_test['bias'],
      "\n *  Test Rel-Err (%): ", logs_test['err-rel'])
print("================================================\n")


logger.save(model)
