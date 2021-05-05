"""
### Author: Jacob Parmer
###
### Created: Mar 20, 2021
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, verbose=False):
        super(RNN, self).__init__()

        self.device = device
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

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden

    def export_model(self):
        return

    def train(self, X, y, epochs=100, lrate=0.01, verbose=False):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        
        target_seq = torch.zeros(64064)
        
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            output, hidden = self(X)
            target_seq = target_seq.to(self.device)
            loss = criterion(output, target_seq.view(-1).long())
            loss.backward()
            optimizer.step()

            if verbose and epoch%10 == 0:
                print(f"epoch: {epoch}/{epochs}")
                print(f"Loss: {loss.item():.5}")

        for line in output:
            np_line = line.detach().numpy()
            print(np.argmax(np_line, axis=0))

        return