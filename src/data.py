"""
### Authors: Dennis Brown, Shannon McDade, Jacob Parmer
###
### Created: Mar 21, 2021
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import os

class LibriSpeech:

    def __init__(self, root, url, folder_in_archive, download=False, batch_size=64, shuffle=True):
        self.dataset = datasets.LIBRISPEECH(
                            root=root,
                            url=url,
                            folder_in_archive=folder_in_archive,
                            download=download
                        )
        
        if not os.path.exists(self.dataset._path):
            raise RuntimeError(
                "Dataset not found. Please use 'download=True' to download it."
            )

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    
