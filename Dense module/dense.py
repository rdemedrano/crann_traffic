"""
MODELS MODULE
"""
import torch.nn as nn

class MLP(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Dense module
    
    --------------
    | Attributes |
    --------------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    n_layers : int
        Number of layers
    n_hidden : int
        Dimension of hidden layers

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, n_inputs, n_outputs, n_layers=1, n_hidden=0, dropout = 0):
        super(MLP, self).__init__()       
        if n_layers < 1:
          raise ValueError('Number of layers needs to be at least 1.')  
        elif n_layers == 1:
            self.module = nn.Linear(n_inputs, n_outputs)
        else:
            modules = [nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
            n_layers -= 1
            while n_layers > 1:
                modules += [nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
                n_layers -= 1
            modules.append(nn.Linear(n_hidden, n_outputs))
            self.module = nn.Sequential(*modules)

        
    def forward(self, x):
        return self.module(x)
