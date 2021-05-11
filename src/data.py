"""
### Author: Jacob Parmer
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
import webrtcvad
import pandas
import math

from src.time_logs import TimerLog
from src.signals import Signal

import numpy as np
from pudb import set_trace

FRAME_SIZE_MS = 10

class LibriSpeech:

    def __init__(self, root, url, folder_in_archive, download=False, batch_size=64, shuffle=True):
        self.dataset = datasets.LIBRISPEECH(
                            root=root,
                            url=url,
                            folder_in_archive=folder_in_archive,
                            download=download
                        )
        self.name = url
        if not os.path.exists(self.dataset._path):
            raise RuntimeError(
                "Dataset not found. Please use 'download=True' to download it."
            )

        self.labels = []

    def load_data(self, index, n_mels, n_mfcc):

        # 1. Get MFCC features from data at index
        data = Signal(self.dataset[index][0], self.dataset[index][1])
        frame_size = int(data.sample_rate * (FRAME_SIZE_MS / 1000.0))

        data.split_into_frames(frame_size=frame_size) # webrtc only supports 10, 20, 30 ms frames
        X = data.get_MFCC(hop_length=frame_size, n_mels=n_mels, n_mfcc=n_mfcc).transpose(2,0).transpose(1,2)

        # 2. Get labels for each frame in MFCC features
        label_dir = label_dir = os.getcwd() + "/LibriSpeech/labels/" + self.name # I think this will break on windows? Checking needed.
        file_dir = label_dir + "/" + str(self.dataset[index][3]) + "/" + str(self.dataset[index][4])
        file_name = file_dir + f"/{self.dataset[index][3]}-{self.dataset[index][4]}-{str(self.dataset[index][5]).zfill(4)}.csv"

        y = torch.tensor(pandas.read_csv(file_name, delimiter=",", header=None).values)

        return X, y

    # -------- PRIVATE MEMBERS ----------- #
    def _label_data(self, dataset_name, verbose=False):
        if verbose:
            print("Beginning VAD labeling...")
            print("-------------------------")

        vad = webrtcvad.Vad(3)

        label_dir = os.getcwd() + "/LibriSpeech/labels/" + dataset_name

        for i, data in enumerate(self.dataset):
            sig = Signal(data[0], data[1])

            sig.split_into_frames(frame_size=int(sig.sample_rate * (FRAME_SIZE_MS / 1000.0))) # webrtc only supports 10, 20, 30 ms frames

            labels = [1 if vad.is_speech(np.int16(f * 32768).tobytes(), sample_rate=sig.sample_rate) else 0 for f in sig.split_waveform]

            # Write labels to .csv files
            file_label_dir = label_dir + "/" + str(self.dataset[i][3]) + "/" + str(self.dataset[i][4])
            if not os.path.exists(file_label_dir):
                os.makedirs(file_label_dir)

            file_name = file_label_dir + f"/{self.dataset[i][3]}-{self.dataset[i][4]}-{str(self.dataset[i][5]).zfill(4)}.csv"

            if verbose:
                print(f"Writing labels to file {file_name}...")

            label_file = open(file_name, "w")
            label_file.write(str(labels[0]))
            for label in labels[1:]:
                label_file.write("," + str(label))

            label_file.close()
    
        return


def build_librispeech(mode, verbose=False):

    if verbose:
        print("Loading LibriSpeech Data...")
        print("If LibriSpeech is not already downloaded, this might take a while...")

    librispeech = {}
    if mode == 'training':
        datasets = ["dev-clean", "dev-other", "train-clean-100", "train-clean-360"]
    elif mode == 'testing':
        datasets = ["test-clean", "test-other"]
    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'")


    for name in datasets:
        # 1. Creates dictionary of librispeech objects, each corresponding to data in LibriSpeech/ folder
        librispeech[name] = LibriSpeech(
                                root=os.getcwd(),
                                url=name,
                                folder_in_archive="LibriSpeech",
                                download=True
                            )

        # 2. Labels librispeech data if labels don't already exist
        if not os.path.exists(os.getcwd() + "/LibriSpeech/labels/" + name):
            librispeech[name]._label_data(dataset_name=name, verbose=verbose)

    if verbose:
        print("Done!")

    return librispeech
    
