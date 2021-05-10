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

from src.time_logs import TimerLog

from pudb import set_trace

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, device, verbose=False):
        super(RNN, self).__init__()

        self.device = device
        self.to(self.device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.verbose = verbose

        if self.verbose:
            print(f'Using {self.device} device')
        

        self.relu = nn.LeakyReLU()

        #self.hidden = self.init_hidden()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.lin1 = nn.Linear(hidden_size, 26)
        self.lin2 = nn.Linear(26, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        self.hidden = self.init_hidden(x.size(1))
        x, _ = self.rnn(x, self.hidden)
        
        x = x.contiguous().view(x.size(0), -1)

        x = self.relu(self.lin1(x))
        x = self.lin2(x)

        return self.softmax(x)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h, c

    def train(self, X, y, epochs=100, lrate=0.01, verbose=False):

        timer = TimerLog()

        if verbose:
            print(f"Starting training for {epochs} epochs...")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        
        for epoch in range(1, epochs+1):
            for i in range(len(X)):
                target_seq = y[i]
                optimizer.zero_grad()
                output = self(X[i])
                target_seq = target_seq.to(self.device)
                loss = criterion(output, target_seq.view(-1).long())
                loss.backward()
                optimizer.step()

                if i == len(X)-1 and verbose:
                    print(f"Sample Output:\n{np.argmax(output.detach().numpy(), axis=1)}")
                    print(f"Targets:\n{target_seq}")

            if verbose:
                print(f"epoch: {epoch}/{epochs}")
                print(f"Loss: {loss.item():.5}")

        for line in output:
            np_line = line.detach().numpy()
            print(np.argmax(np_line, axis=0))

        if verbose:
            print(f"Finished model training in {timer.get_elapsed()} seconds.")

        return