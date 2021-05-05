"""
### Author: Jacob Parmer
###
### Created: Mar 21, 2021
"""

import time

class TimerLog:

    def __init__(self, verbose=True):
        self.time_created = time.time()
        self.timer = time.time()
        self.verbose = verbose

    def start(self):
        self.timer = time.time()

    def stop(self, help_text):
        elapsed = time.time() - self.timer
        if self.verbose:
            print(f"{help_text} {elapsed} seconds")

    def get_elapsed(self):
        return time.time() - self.timer
        