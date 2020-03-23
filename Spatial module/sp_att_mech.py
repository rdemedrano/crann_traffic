"""
MODELS MODULE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCNN(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Spatial module with spatio-temporal attention
    
    --------------
    | Attributes |
    --------------
    in_channels : int
        Number of input timesteps
    out_channels : int
        Number of output timesteps
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, in_channels, out_channels, dim_x, dim_y):
        super(AttentionCNN, self).__init__()
        #Variables
        self.out_channels = out_channels
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        # Conv blocks
        self.conv_block1 = ConvBlock(in_channels, 64, 5)
        
        # Attention
        self.att1 = AttentionBlock(dim_x, dim_y, 24, method = 'hadamard')
        
        # Output
        self.regressor = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.regressor(out)
        out, att = self.att1(out)     
        return out, att
    
        
class ConvBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Convolutional blocks of num_conv convolutions with out_features channels
    
    --------------
    | Attributes |
    --------------
    in_features : int
        Number of input channels
    out_features : int
        Number of middle and output channels
    num_conv : int
        Number of convolutions
    
    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, in_features, out_features, num_conv):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.op(x)
    
        
class AttentionBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Attentional block for spatio-temporal attention mechanism
    
    --------------
    | Attributes |
    --------------
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images
    timesteps : int
        Number of input timesteps
    method : str
        Attentional function to calculate attention weights

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, dim_x, dim_y, timesteps, method='hadamard'):
        super(AttentionBlock, self).__init__()
        # Variables
        self.method = method
        self.weight = nn.Parameter(torch.FloatTensor(timesteps, dim_x*dim_y, dim_x*dim_y))
        torch.nn.init.xavier_uniform_(self.weight)
        if method == 'general':
            self.fc = nn.Linear(timesteps*(dim_x*dim_y)**2, timesteps*(dim_x*dim_y)**2, bias = False)
        elif method == 'concat':
            self.fc = nn.Linear(timesteps*(dim_x*dim_y)**2, timesteps*(dim_x*dim_y)**2, bias = False)
    
    def forward(self, x, y=0):
        N, T, W, H = x.size()
        if self.method == 'hadamard':
            xp = x.view(N,T,-1).repeat(1,1,W*H).view(N,T,W*H,W*H)
            wp = self.weight.expand_as(xp)
            alig_scores = wp.mul(xp)
        elif self.method =='general':
            xp = x.view(N,T,-1).repeat(1,1,W*H).view(N,T,W*H,W*H)
            wp = self.weight.expand_as(xp)
            alig_scores = self.fc((wp.mul(xp)).view(N,-1))
        elif self.method == 'concat':
            xp = x.view(N,T,-1).repeat(1,1,W*H).view(N,T,W*H,W*H)
            wp = self.weight.expand_as(xp)
            alig_scores = torch.tanh(self.fc((wp + xp).view(N,-1)))
        elif self.method == 'dot':
            xp = x.view(N,T,-1).repeat(1,1,W*H).view(N,T,W*H,W*H)
            alig_scores = self.weight.matmul(xp)

        att_weights = F.softmax(alig_scores.view(N, T, W*H, W*H), dim = 3)
        out = att_weights.matmul(x.view(N,T,-1).unsqueeze(3))
        return out.view(N,T,W,H), att_weights
