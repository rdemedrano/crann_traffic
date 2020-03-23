"""
TRAINING MODULE
"""
import torch
import torch.optim as optim

import sys
import os
import numpy as np
import configargparse
import json
from datetime import datetime
from collections import defaultdict
from tqdm import trange

from dataset import data_transform
from utils import DotDict, Logger, evaluate_temp_att, denormalize_data, rmse, bias, rel_error
from dense import MLP
sys.path.append('../Spatial module/') 
from sp_att_mech import AttentionCNN
sys.path.append('../Temporal module/') 
from bahdanau_att import EncoderLSTM, BahdanauDecoder


"""
#######################################################################################################################
# VARIABLES AND OPTIONS
#######################################################################################################################
"""
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='dense_data.csv')
# -- exp
p.add('--outputdir', type=str, help='path to save exp', default='output/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
# -- model
# ---- spatial module
p.add('--dim_x', type=int, help='x dimension for image creation', default=5)
p.add('--dim_y', type=int, help='x dimension for image creation', default=6)
p.add('--n_inp_sp', type=int, help='number of input timesteps', default=24)
p.add('--n_out_sp', type=int, help='number of output timesteps', default=24)
p.add('--n_points', type=int, help='number of spatial points/sensors', default=30)
# ---- temporal module
p.add('--n_inp_tem', type=int, help='number of input timesteps', default=24*14)
p.add('--n_out_tem', type=int, help='number of output timesteps', default=24)
p.add('--in_dim_tem', type=int, help='number of input features', default=1)
p.add('--out_dim_tem', type=int, help='number of output features', default=1)
p.add('--n_hidden_tem', type=int, help='hidden dimension of enc-dec', default=100)
p.add('--n_layers_tem', type=int, help='number of layers for enc-dec', default=1)
# ---- dense module
p.add('--n_exo', type=int, help='number of exogenous features', default=6)
p.add('--n_hidden_dns', type=int, help='hidden dimension of dense', default=0)
p.add('--n_layers_dns', type=int, help='number of layers for dense', default=1)
p.add('--n_ar', type=int, help='number of autoregressive terms', default=4)
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-3)
p.add('--beta1', type=float, default=.9, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-8, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=5e-3)
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
X_train_time, Y_train_time, X_val_time, Y_val_time, X_test_time, Y_test_time, min_value_time, max_value_time, \
X_train_space, Y_train, X_val_space, Y_val, X_test_space, Y_test, min_value_space, max_value_space, \
X_train_exo, X_val_exo, X_test_exo, min_value_exo, max_value_exo = data_transform(opt)


train_dataset = []
for i in range(len(Y_train)):
    train_dataset.append([X_train_time[i], X_train_space[i], X_train_exo[i], Y_train[i]])
    
val_dataset = []
for i in range(len(Y_val)):
    val_dataset.append([X_val_time[i], X_val_space[i], X_val_exo[i], Y_val[i]]) 
    
test_dataset = []
for i in range(len(Y_test)):
   test_dataset.append([X_test_time[i], X_test_space[i], X_test_exo[i], Y_test[i]]) 
    

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = opt.batch_size,
                                           shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = len(Y_val),
                                           shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = len(Y_test),
                                           shuffle = False)


"""
#######################################################################################################################
# MODELS
#######################################################################################################################
"""
# DENSE
inputs = opt.n_out_sp*(opt.n_points + opt.n_exo + 1) 
outputs = opt.n_out_sp*opt.n_points

model = MLP(n_inputs=inputs + opt.n_ar*opt.n_points, n_outputs=outputs, n_layers=opt.n_layers_dns, n_hidden=opt.n_hidden_dns)
model.to(device)

# ESPACIAL
spatial_model = AttentionCNN(in_channels=opt.n_inp_sp, out_channels=opt.n_out_sp, dim_x=opt.dim_x, dim_y=opt.dim_y)
spatial_model.to(device)
spatial_model.load_state_dict(torch.load("../Trained models/spatial_model.pth"))

# TEMPORAL
temporal_encoder = EncoderLSTM(opt.in_dim_tem, opt.n_hidden_tem, device=device)
temporal_encoder.to(device)
temporal_encoder.load_state_dict(torch.load("../Trained models/encoder.pth"))
temporal_decoder = BahdanauDecoder(opt.n_hidden_tem, opt.out_dim_tem)
temporal_decoder.to(device)
temporal_decoder.load_state_dict(torch.load("../Trained models/decoder.pth"))


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


spatial_model.eval()
temporal_encoder.eval()
temporal_decoder.eval()

for param in spatial_model.parameters():
    param.requires_grad = False
    
for param in temporal_encoder.parameters():
    param.requires_grad = False
    
for param in temporal_decoder.parameters():
    param.requires_grad = False

for t in tr:     
    # Training    
    model.train()
    logs_train = defaultdict(float)
    for i, (x_time, x_space, x_exo, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x_time = x_time.view(-1, opt.n_inp_tem, opt.in_dim_tem).to(device)
            x_space = x_space.to(device)
            x_exo = x_exo.to(device)
            y = y.to(device)
        else:
            x_time = x_time
            x_space = x_space
            x_exo = x_exo
            y = y
        optimizer.zero_grad()
        y_time = evaluate_temp_att(temporal_encoder, temporal_decoder, x_time, opt.n_out_sp, device)
        y_space = spatial_model(x_space)[0]
        x = torch.cat((y_time.unsqueeze(2), y_space.squeeze().view(-1,opt.n_out_sp,opt.n_points), x_exo), dim = 2).view(-1, inputs)
        x = torch.cat((x,x_space[:,-opt.n_ar:].view(-1,opt.n_ar*opt.n_points)), dim = 1)
        y_pred = model(x).view(-1,opt.n_out_sp,opt.dim_x,opt.dim_y)
        loss = loss_fn(y_pred, y)
        loss.backward()
    
        optimizer.step()
        logs_train['mse'] += loss.item()
        
    # Logs training
    logs_train['mse'] /= (i+1)
    logger.log('train', logs_train)
    

    model.eval()
    logs_val = defaultdict(float)
    with torch.no_grad():
        for x_time, x_space, x_exo, y in val_loader:
            if torch.cuda.is_available():
                x_time = x_time.view(-1,  opt.n_inp_tem, opt.in_dim_tem).to(device)
                x_space = x_space.to(device)
                x_exo = x_exo.to(device)
                y = y.to(device)
            else:
                x_time = x_time
                x_space = x_space
                x_exo = x_exo
                y = y
            y_time = evaluate_temp_att(temporal_encoder, temporal_decoder, x_time, opt.n_out_sp, device)
            y_space, _ = spatial_model(x_space)
            x = torch.cat((y_time.unsqueeze(2), y_space.squeeze().view(-1,opt.n_out_sp,opt.n_points), x_exo), dim = 2).view(-1, inputs)
            x = torch.cat((x,x_space[:,-opt.n_ar:].view(-1,opt.n_ar*opt.n_points)), dim = 1)
            y_pred = model(x).view(-1,opt.n_out_sp,opt.dim_x,opt.dim_y)
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
    if lr < 1e-5:
        break
            
"""
#######################################################################################################################
# TEST
#######################################################################################################################
"""
model.eval()
logs_test = defaultdict(float)
with torch.no_grad():        
    for x_time, x_space, x_exo, y in test_loader:
        if torch.cuda.is_available():
            x_time = x_time.view(-1, opt.n_inp_tem, opt.in_dim_tem).to(device)
            x_space = x_space.to(device)
            x_exo = x_exo.to(device)
            y = y.to(device)
        else:
            x_time = x_time
            x_space = x_space
            x_exo = x_exo
            y = y
        y_time = evaluate_temp_att(temporal_encoder, temporal_decoder, x_time, opt.n_out_sp, device)
        y_space, _ = spatial_model(x_space)
        x = torch.cat((y_time.unsqueeze(2), y_space.squeeze().view(-1,opt.n_out_sp,opt.n_points), x_exo), dim = 2).view(-1, inputs)
        x = torch.cat((x,x_space[:,-opt.n_ar:].view(-1,opt.n_ar*opt.n_points)), dim = 1)
        y_pred = model(x).view(-1,opt.n_out_sp,opt.dim_x,opt.dim_y)
        
        y_dnorm = denormalize_data(y.view(-1, opt.n_out_sp, opt.n_points).cpu(), min_value_space, max_value_space)
        y_pred_dnorm = denormalize_data(y_pred.view(-1, opt.n_out_sp, opt.n_points).cpu(), min_value_space, max_value_space)
        
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
