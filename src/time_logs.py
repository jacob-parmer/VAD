"""
### Author: Jacob Parmer
###
### Created: Mar 21, 2021
"""

import time

class TimerLog:
    """
    Shows information on run-time since object creation or start() function call.
    A glorified stopwatch.
    """

    def __init__(self):
        """
        Initializes timing object, with timer and time_created set to creation system time.
        """

        self.time_created = time.time()
        self.timer = time.time()

    def start(self):
        """
        Sets timer to current system time.
        """
        self.timer = time.time()

    def get_elapsed(self):
        """
        Gets amount of time past since object creation or last start() call by finding difference
        between current system time and start time.
        """
        return time.time() - self.timer
        