"""
TRAINING MODULE
"""
import torch
import torch.optim as optim


import os
import numpy as np
import configargparse
import json
import random
from datetime import datetime
from collections import defaultdict
from tqdm import trange

from dataset import data_transform
from utils import DotDict, Logger, denormalize_data, rmse, bias, rel_error
from bahdanau_att import EncoderLSTM, BahdanauDecoder

"""
#######################################################################################################################
# VARIABLES AND OPTIONS
#######################################################################################################################
"""
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='tem_data.csv')
# -- exp
p.add('--outputdir', type=str, help='path to save exp', default='output/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
# -- model
p.add('--n_inp', type=int, help='number of input timesteps', default=24*14)
p.add('--n_out', type=int, help='number of output timesteps', default=24)
p.add('--n_points', type=int, help='number of spatial points/sensors', default=30)
p.add('--in_dim', type=int, help='number of input features', default=1)
p.add('--out_dim', type=int, help='number of output features', default=1)
p.add('--n_hidden', type=int, help='hidden dimension of enc-dec', default=100)
p.add('--n_layers', type=int, help='number of layers for enc-dec', default=1)
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-2)
p.add('--beta1', type=float, default=.9, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-8, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=3e-5)
p.add('--d_e', type=float, help='dropout encoder', default=0)
p.add('--d_d', type=float, help='dropour decoder', default=0)
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
encoder = EncoderLSTM(opt.in_dim, opt.n_hidden, opt.n_layers, opt.d_e, device)
decoder = BahdanauDecoder(opt.n_hidden, opt.out_dim, opt.n_layers, opt.d_d)

encoder.to(device)
decoder.to(device)


"""
#######################################################################################################################
# AUXILIAR FUNCTIONS
#######################################################################################################################
"""
def evaluate(encoder, decoder, batch, device = device):
     output = torch.Tensor().to(device)
     
     h = encoder.init_hidden(batch.size(0))
            
     encoder_output, h = encoder(batch,h)
     decoder_hidden = h
     decoder_input = torch.zeros(batch.size(0), 1, device = device)
            
     for k in range(opt.n_out):
         decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
         decoder_input = decoder_output
         output = torch.cat((output, decoder_output),1)

     return output
 
    
"""
#######################################################################################################################
# OPTIMIZER
#######################################################################################################################
"""
loss_fn = torch.nn.MSELoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)

if opt.patience > 0:
    lr_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, patience=opt.patience)
    lr_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, patience=opt.patience)

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
    # training
    encoder.train()
    decoder.train()
    logs_train = defaultdict(float)
    for i_1 ,(x,y) in enumerate(train_loader):
        
        x = x.view(-1,opt.n_inp,opt.in_dim).to(device)
        y = y.to(device)
            
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
            
        h = encoder.init_hidden(x.size(0))
                
        encoder_output, h = encoder(x,h)
        decoder_hidden = h
        decoder_input = torch.zeros(x.size(0), 1, device = device)
            
        teacher_forcing = True if random.random() < 0.5 else False
        y_pred = torch.Tensor().to(device)
        att = torch.Tensor().to('cpu')
        for i_2 in range(y.size(1)):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            att = torch.cat((att, attn_weights.detach().unsqueeze(2).cpu()),2)

            if teacher_forcing:
                decoder_input = y[:,i_2].view(-1,1)
            else:
                decoder_input = decoder_output
            y_pred = torch.cat((y_pred, decoder_output),1)
            
        loss = loss_fn(y_pred, y)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        logs_train['mse'] += loss.item()            
            
    # logs training
    logs_train['mse'] /= (i+1)
    logger.log('train', logs_train)
            
    encoder.eval()
    decoder.eval()
    logs_val = defaultdict(float)
    with torch.no_grad():
        for x,y in val_loader:
            x = x.view(-1,opt.n_inp,opt.in_dim).to(device)
            y = y.to(device)
            
            y_pred = evaluate(encoder, decoder, x)
            loss_val = loss_fn(y_pred, y)
            logs_val['mse'] = loss_val.item()
            
        # logs evaluation
        logger.log('val', logs_val)
        
     # general information
    tr.set_postfix(train_mse = logs_train['mse'], val_mse=logs_val['mse'], 
                   train_rmse = np.sqrt(logs_train['mse']), val_rmse=np.sqrt(logs_val['mse']),
                   lr = lr)
    logger.checkpoint(encoder, 'encoder')
    logger.checkpoint(decoder, 'decoder')
     
    # learning rate decay
    if opt.patience > 0:
        lr_scheduler1.step(logs_val['mse'])
        lr = encoder_optimizer.param_groups[0]['lr']
        
        lr_scheduler2.step(logs_val['mse'])
        lr = decoder_optimizer.param_groups[0]['lr']
    if lr <= 1e-5:
        break        


"""
#######################################################################################################################
# TEST
#######################################################################################################################
"""
encoder.eval()
decoder.eval()
logs_test = defaultdict(float)
with torch.no_grad():    
    for x,y in test_loader:
        x = x.view(-1,opt.n_inp,opt.in_dim).to(device)
        y = y.to(device)
     
        y_pred = evaluate(encoder, decoder, x)
        
        y_pred_dnorm = denormalize_data(y_pred, min_value, max_value)
        y_dnorm = denormalize_data(y, min_value, max_value)
        
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


logger.save(encoder, 'encoder')
logger.save(decoder, 'decoder')