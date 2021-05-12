"""
### Author: Jacob Parmer
###
### Created: Mar 21, 2021
"""

import torch
from torchaudio import datasets
import os
import webrtcvad
import pandas
import numpy as np

from src.signals import Signal

FRAME_SIZE_MS = 10

class LibriSpeech:
    """
    Handles librispeech data, which includes audio waveforms, sample rate, the text being spoken, and index information.
    """

    def __init__(self, root, url, folder_in_archive, download=False):
        """
        Creates Librispeech object.
    
        If download is set to true, torch's dataset handler will automatically download the .tar to the machine and extract.
        Note that it doesn't delete the tar after unpacking, so the size ends up being 2x what it needs to be

        Params:
            (str) root: Path to root directory of the project
            (str) url: Name of the dataset to be pulled, e.g. "dev-clean", "dev-other".
                       Can be found here - https://www.openslr.org/12
            (str) folder_in_archive: Subdirectory to find dataset - should be 'LibriSpeech'.
            (bool) download: Sets automatic download if files not found            
        """
        self.dataset = datasets.LIBRISPEECH(
                            root=root,
                            url=url,
                            folder_in_archive=folder_in_archive,
                            download=download
                        )
        self.name = url
        if not os.path.exists(self.dataset._path):
            raise RuntimeError(
                "Dataset not found. Please use 'download=True' to download it, or manually grab files from https://www.openslr.org/12 and paste them into LibriSpeech/ folder."
            )

    def load_data(self, index, n_mels, n_mfcc):
        """
        Loads features and target values for specified data

        This was added so that data could be passed into the model on a one-by-one basis, rather than loading them all at the beginning and rolling from there.
        These datasets get pretty big apparently, so loading the features from them all at once causes memory issues.

        I'm pretty sure this function will fail on windows because of the way I'm pulling the file.
        Either fix this or just dockerize the application so that I don't have to deal with it. :)
        
        Params:
            (int) index: Position in the dataset to load data from
            (int) n_mels: Number of mel bins to return from MFCC feature extraction
            (int) n_mfcc: Number of cepstral coefficients to return from MFCC feature extraction

        Returns:
            (tensor) X: MFCC data of shape [?,?,?]
            (tensor) y: Label data of shape [?,?,?]
        """
        # 1. Get MFCC features from data at index
        data = Signal(self.dataset[index][0], self.dataset[index][1])
        frame_size = int(data.sample_rate * (FRAME_SIZE_MS / 1000.0))
        X = data.get_MFCC(hop_length=frame_size, n_mels=n_mels, n_mfcc=n_mfcc).transpose(2,0).transpose(1,2)

        # 2. Get labels for each frame in MFCC features
        label_dir = label_dir = os.getcwd() + "/LibriSpeech/labels/" + self.name 
        file_dir = label_dir + "/" + str(self.dataset[index][3]) + "/" + str(self.dataset[index][4])
        file_name = file_dir + f"/{self.dataset[index][3]}-{self.dataset[index][4]}-{str(self.dataset[index][5]).zfill(4)}.csv"

        y = torch.tensor(pandas.read_csv(file_name, delimiter=",", header=None).values)

        return X, y

    # -------- PRIVATE MEMBERS ----------- #
    def _label_data(self, dataset_name, verbose=False):
        """
        Creates labels and saves them as .csv files to LibriSpeech/labels/ directory.

        Utilizes webrtcvad package to generate labels, currently set to mode 3, which aggressively filters out non-speech.
        Could possibly lead to higher FRR.

        Params:
            (str) dataset_name: Name of the dataset to be labelled, e.g. "dev-clean".
            (bool) verbose: Displays running information to CLI while running if this is enabled

        Returns:
            .csv files in LibriSpeech/labels/dataset_name subdirectory
        """
        if verbose:
            print("Beginning VAD labeling...")
            print("-------------------------")

        vad = webrtcvad.Vad(3)

        label_dir = os.getcwd() + "/LibriSpeech/labels/" + dataset_name

        for i, data in enumerate(self.dataset):
            sig = Signal(data[0], data[1])

            split_waveform = sig.split_into_frames(frame_size=int(sig.sample_rate * (FRAME_SIZE_MS / 1000.0))) # webrtc only supports 10, 20, 30 ms frames

            labels = [1 if vad.is_speech(np.int16(f * 32768).tobytes(), sample_rate=sig.sample_rate) else 0 for f in split_waveform]

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
    """
    Creates and returns librispeech objects for each dataset in datasets variable.
    Also creates labels if they don't already exist.

    Params:
        (str) mode: 'training' or 'testing', however the program is being run
        (bool) verbose: Displays running information to CLI while running if this is enabled

    Returns:
        (dict) librispeech: Contains all the data from each of the loaded datasets
    """
    if verbose:
        print("Loading LibriSpeech Data...")
        print("If LibriSpeech is not already downloaded, this might take a while...")

    librispeech = {}
    if mode == 'training':
        datasets = ["train-other-500", "train-clean-360"] # Extras to add? "train-clean-100", "dev-clean", "dev-other"
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
    
