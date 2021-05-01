"""
### Authors: Dennis Brown, Shannon McDade, Jacob Parmer
###
### Created: Mar 20, 2021
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, verbose=False):
        super(RNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.verbose = verbose
        if self.verbose:
            print(f'Using {self.device} device')
        

        # Define params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def export_model(self):
        return