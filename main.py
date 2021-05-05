"""
### Author: Jacob Parmer
###
### Created: Mar 20, 2021
"""

from src.signals import Signal
from src.display import Display
from src.model import RNN
from src.time_logs import TimerLog
from src.data import LibriSpeech, get_features, build_librispeech

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

    _print(f"{stopwatch.get_elapsed():.6f}\t| {print_str}")

# ---------- MAIN PROGRAM EXECUTION ----------- #
def main(args):

    if args.timed:
        builtins.print = timed_print

    librispeech = build_librispeech(verbose=args.verbose)
    
    # ----- TRAIN ----- #
    if args.mode == "training":
        if args.verbose:
            print("Beginning VAD model training...")
            print("-------------------------------")

        features = get_features(librispeech["train-clean-100"].dataset, device=args.device)

        model = RNN(input_size=features.size(2),
                    output_size=1,
                    hidden_dim=2, 
                    n_layers=1, 
                    device=args.device,
                    verbose=args.verbose
                    )

        y = list()
        model.train(features, y, epochs=args.epochs, lrate=args.lrate, verbose=args.verbose)          


    # ----- TEST ----- #
    elif args.mode == "testing":
        if args.verbose:
            print("Beginning VAD model testing...")
            print("------------------------------")

        features = get_features(librispeech["test_clean"].dataset, device=args.device)

        model = RNN(input_size=features.size(2),
                    output_size=1,
                    hidden_dim=2,
                    n_layers=1,
                    device=args.device,
                    verbose=args.verbose
                    )


    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debugging logs")
    parser.add_argument("--timed", "-t", action="store_true", help="Display execution time in debugging logs")
    parser.add_argument("--mode", "-m",  help="Set program to training or testing mode")
    parser.add_argument("--lrate", "-l", default=0.01, type=float, help="Set learning rate for model")
    parser.add_argument("--epochs", "-e", default=100, type=int, help="Set number of epochs for model training")
    parser.add_argument("--device", "-d", default='cuda' if torch.cuda.is_available() else 'cpu', help="Set program to run on CPU or GPU")
    args = parser.parse_args()

    main(args)



