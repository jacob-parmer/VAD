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
        self.labels = []

    def get_labels(self, dataset_name, verbose=False):

        label_dir = os.getcwd() + "/LibriSpeech/labels/" + dataset_name

        if verbose:
            print(f"Reading labels from dataset {dataset_name}")

        for i in range(len(self.dataset)):
            file_label_dir = label_dir + "/" + str(self.dataset[i][3]) + "/" + str(self.dataset[i][4])
            file_name = file_label_dir + f"/{self.dataset[i][3]}-{self.dataset[i][4]}-{str(self.dataset[i][5]).zfill(4)}.csv"

            labels_tensor = torch.tensor(pandas.read_csv(file_name, delimiter=",", header=None).values)
            self.labels.append(labels_tensor)

            if verbose:
                print(f"Reading labels... {i}/{len(self.dataset)}")

        return self.labels

    def get_features(self, device, verbose=False):

        timer = TimerLog()

        if verbose:
            print("Starting Feature Extraction...")

        dataset_size = len(self.dataset)
        features = []
        for i, data in enumerate(self.dataset):
            sig = Signal(data[0], data[1])
            sig.split_into_frames(frame_size=int(sig.sample_rate * (FRAME_SIZE_MS / 1000.0))) # webrtc only supports 10, 20, 30 ms frames
            sig_features = sig.get_MFCC().transpose(2,0).transpose(1,2)

            features.append(sig_features)

            if verbose:
                print(f"#{i}/{dataset_size}")
                sig.print_stats()
        
        if verbose:
            print(f"Finished Feature Extraction. Total Time Elapsed: {timer.get_elapsed()}")

        return features

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

            labels = [1 if vad.is_speech(np.int16(f * 32768).tobytes(), sample_rate=sig.sample_rate) else 0 for f in sig.waveform]

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

        # 2. Labels librispeech data if labels don't already exist
        if not os.path.exists(os.getcwd() + "/LibriSpeech/labels/" + name):
            librispeech[name]._label_data(dataset_name=name, verbose=verbose)


    if verbose:
        print("Done!")

    return librispeech
    
