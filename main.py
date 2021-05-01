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

import argparse
import os
import builtins

# Displays time in program before every print statement.
_print = print
stopwatch = TimerLog()

def timed_print(*args):
    print_str = ""
    for arg in args:
        print_str += str(arg) + " "

    _print(f"{stopwatch.get_elapsed()}\t| {print_str}")


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

        dataset_size = len(train_clean_100.dataset)

        for i, data in enumerate(train_clean_100.dataset):
            sig = Signal(data[0], data[1])

            features = {"MFCC": sig.get_MFCC(),
                        "pitch": sig.get_pitch()
                        }

            if args.verbose:
                print(f"#{i}/{dataset_size}")
                sig.print_stats()
            


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

        dataset_size = len(test_clean.dataset)

        for i, data in enumerate(test_clean.dataset):
            sig = Signal(data[0], data[1])

            features = {"MFCC": sig.get_MFCC(),
                        "pitch": sig.get_pitch()
                        }

            if args.verbose:
                print(f"#{i}/{dataset_size}")
                sig.print_stats()
            
    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debugging logs")
    parser.add_argument("--timed", "-t", action="store_true", help="Display execution time in debugging logs")
    parser.add_argument("--mode", "-m",  help="Set program to training or testing mode")
    args = parser.parse_args()

    main(args)