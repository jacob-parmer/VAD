"""
### Authors: Dennis Brown, Shannon McDade, Jacob Parmer
###
### Created: Mar 20, 2021
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class RNN(nn.Module):

    def __init__(self, verbose=False):
        super(RNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.verbose = verbose
        if self.verbose:
            print(f'Using {self.device} device')

    def export_model(self):
        return