"""
MODELS MODULE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Encoder for temporal module
    
    --------------
    | Attributes |
    --------------
    input_size : int
        Number of input features
    hidden_size : int
        Dimension of hidden space
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder
    device : int/str
        Device in which hiddens are stored

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, device = 'cuda'):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)
        
    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))

class BahdanauDecoder(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Decoder an attention mechanism for temporal module
    
    --------------
    | Attributes |
    --------------
    hidden_size : int
        Dimension of hidden space
    output_size : int
        Number of output features
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size + self.output_size, self.hidden_size, batch_first=True)
        self.fc_prediction = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
    
        # Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0].view(-1, 1, self.hidden_size))+
                       self.fc_encoder(encoder_outputs))
        
        alignment_scores = x.matmul(self.weight.unsqueeze(2))
        
        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(inputs.size(0),-1), dim=1)
        
        # Multiplying the Attention weights with encoder outputs to get the context vector
        self.context_vector = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Concatenating context vector with embedded input word
        output = torch.cat((inputs, self.context_vector.squeeze(1)), 1).unsqueeze(1)
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
    
        output = self.fc_prediction(output).squeeze(2)
    
        return output, hidden, attn_weights