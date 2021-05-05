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
        
        if not os.path.exists(self.dataset._path):
            raise RuntimeError(
                "Dataset not found. Please use 'download=True' to download it."
            )

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    # -------- PRIVATE MEMBERS ----------- #
    def _label_data(self, dataset_name, verbose=False):
        if verbose:
            print("Beginning VAD labeling...")
            print("-------------------------")

        vad = webrtcvad.Vad(3)

        #os.makedirs(os.getcwd() + "/LibriSpeech/labels/" + dataset_name)
        #features = get_features(self.dataset)

        for i, data in enumerate(self.dataset):
            sig = Signal(data[0], data[1])

            sig.split_into_frames(frame_size=int(sig.sample_rate * (FRAME_SIZE_MS / 1000.0))) # webrtc only supports 10, 20, 30 ms frames

            labels = [1 if vad.is_speech(np.int16(f * 32768).tobytes(), sample_rate=sig.sample_rate) else 0 for f in sig.waveform]
            print(i, labels)
            #print(self.dataset[i])
        return

    def _get_labels(self, dataset_name, verbose=False):

        return


def build_librispeech(verbose=False):

    if verbose:
        print("Loading LibriSpeech Data...")
        print("If LibriSpeech is not already downloaded, this might take a while...")

    librispeech = {}
    download_names = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100",
                        "train-clean-360"] # READD , "train-other-500" LATER

    
    for name in download_names:
        # 1. Creates dictionary of librispeech objects, each corresponding to data in LibriSpeech/ folder    
        librispeech[name] = LibriSpeech(
                                root=os.getcwd(),
                                url=name,
                                folder_in_archive="LibriSpeech",
                                download=True
                            )

        # 2. Labels librispeech data if labels don't already exist, otherwise pull them from labels folder
        breakpoint()
        if not os.path.exists(os.getcwd() + "/LibriSpeech/labels/" + name):
            librispeech[name]._label_data(dataset_name=name, verbose=verbose)
        else:
            librispeech[name]._get_labels(dataset_name=name, verbose=verbose)

    if verbose:
        print("Done!")

    return librispeech

def get_features(dataset, device, verbose=False):

    timer = TimerLog()

    if verbose:
        print("Starting Feature Extraction...")

    dataset_size = len(dataset)
    features = torch.empty(0)
    for i, data in enumerate(dataset):
        sig = Signal(data[0], data[1])
        sig_features = sig.get_mfilterbank().unsqueeze_(-1).transpose(2,0)
                        
        features = torch.cat((features, sig_features), 0)

        if verbose:
            print(f"#{i}/{dataset_size}")
            sig.print_stats()

        if i == 1000:
            break

    features.to(device)
    
    if verbose:
        print(f"Finished Feature Extraction. Total Time Elapsed: {timer.get_elapsed()}")

    return features
    
