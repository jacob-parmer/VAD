"""
### Author: Jacob Parmer
###
### Created: Mar 20, 2021
"""
import argparse
import os
import builtins

import torch

from src.signals import Signal
from src.model import RNN
from src.time_logs import TimerLog
from src.data import LibriSpeech, build_librispeech

N_MELS = 256
MODEL_NAME = "model/RNN.pt"


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

    librispeech = build_librispeech(mode=args.mode, verbose=args.verbose)
    
    # ----- TRAIN ----- #
    if args.mode == "training":
        if args.verbose:
            print("Beginning VAD model training...")
            print("-------------------------------")

        model = RNN(input_size=N_MELS,
                    hidden_size=10, 
                    num_layers=1, 
                    device=args.device,
                    verbose=args.verbose
                    )

        for key in librispeech:

            if args.verbose:
                print(f"Dataset: {key}")

            model.train(librispeech[key], epochs=args.epochs, lrate=args.lrate, verbose=args.verbose)          

        output_filename = ""
        if args.verbose:
            print(f"Success! Finished training in {stopwatch.get_elapsed()} seconds.")
            print(f"Saving model to {MODEL_NAME}")

        torch.save(model, MODEL_NAME)


    # ----- TEST ----- #
    elif args.mode == "testing":
        if args.verbose:
            print("Beginning VAD model testing...")
            print("------------------------------")

        model = torch.load(MODEL_NAME)

        for key in librispeech:

            if args.verbose:
                print(f"Dataset: {key}")

            model.test(librispeech[key], verbose=args.verbose)          

    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debugging logs")
    parser.add_argument("--timed", "-t", action="store_true", help="Display execution time in debugging logs")
    parser.add_argument("--mode", "-m",  help="Set program to training or testing mode")
    parser.add_argument("--lrate", "-l", default=0.01, type=float, help="Set learning rate for model")
    parser.add_argument("--epochs", "-e", default=1, type=int, help="Set number of epochs for model training")
    parser.add_argument("--device", "-d", default='cuda' if torch.cuda.is_available() else 'cpu', help="Set program to run on CPU or GPU")
    args = parser.parse_args()

    main(args)



