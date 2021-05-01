"""
### Authors: Dennis Brown, Shannon McDade, Jacob Parmer
###
### Created: Mar 20, 2021
"""

from src.signals import Signal
from src.display import Display
from src.model import RNN
from src.data import LibriSpeech
from src.time_logs import TimerLog

import torch
from torch import nn

import argparse
import os
import builtins

# Displays time in program before every print statement.
_print = print
stopwatch = TimerLog()

# ------------- HELPER FUNCTIONS --------------- #
def timed_print(*args):
    print_str = ""
    for arg in args:
        print_str += str(arg) + " "

    _print(f"{stopwatch.get_elapsed()}\t| {print_str}")

def get_features(dataset):

    timer = TimerLog()

    if args.verbose:
        print("Starting Feature Extraction...")

    dataset_size = len(dataset)
    features = torch.empty(0)
    for i, data in enumerate(dataset):
        sig = Signal(data[0], data[1])
        sig_features = sig.get_mfilterbank().unsqueeze_(-1).transpose(2,0)
        #sig_features = sig_features.transpose(1,2)
                        
        features = torch.cat((features, sig_features), 0)

        if args.verbose:
            print(f"#{i}/{dataset_size}")
            sig.print_stats()

        if i == 3:
            break

    if args.verbose:
        print(f"Finished Feature Extraction. Total Time Elapsed: {timer.get_elapsed()}")

    return features

# ---------- MAIN PROGRAM EXECUTION ----------- #
def main(args):

    if args.timed:
        builtins.print = timed_print

    if args.mode == "training":
        if args.verbose:
            print("Beginning VAD model training...")
            print("-------------------------------")

        train_clean_100 = LibriSpeech(
                            root=os.getcwd(),
                            url="train-clean-100",
                            folder_in_archive="LibriSpeech",
                            download=True
                        )

        features = get_features(train_clean_100.dataset)
        model = RNN(input_size=len(features[0][0]),
                    output_size=1,
                    hidden_dim=12, 
                    n_layers=1, 
                    verbose=args.verbose
                    )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

        for epoch in range(1, args.epochs+1):
            optimizer.zero_grad()

    elif args.mode == "testing":
        if args.verbose:
            print("Beginning VAD model testing...")
            print("------------------------------")

        test_clean = LibriSpeech(
                        root=os.getcwd(),
                        url="test-clean",
                        folder_in_archive="LibriSpeech",
                        download=True
                    )

        breakpoint()

        features = get_features(test_clean.dataset)
        model = RNN(input_size=len(features[0][0]),
                    output_size=1,
                    hidden_dim=12,
                    n_layers=1,
                    verbose=args.verbose
                    )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

        for epoch in range(1, args.epochs+1):
            optimizer.zero_grad()
            features.to(model.device)
            output, hidden = model(features)



            
    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debugging logs")
    parser.add_argument("--timed", "-t", action="store_true", help="Display execution time in debugging logs")
    parser.add_argument("--mode", "-m",  help="Set program to training or testing mode")
    parser.add_argument("--lrate", "-l", default=0.01, help="Set learning rate for model")
    parser.add_argument("--epochs", "-e", default=100, help="Set number of epochs for model training")
    args = parser.parse_args()

    main(args)



