"""
### Author: Jacob Parmer
###
### Created: Mar 20, 2021
"""

import matplotlib.pyplot as plt
import numpy as np

class Display:
    """
    Visualizes data using matplotlib.
    """
    def __init__(self):
        """
        Initializes the display object.
        """
        return

    def plot_waveform(self, waveform, sample_rate, labels, title=None):
        """
        Plots signal data and overlays labels on top. 
        Useful for visually verifying accuracy of model.

        Params:
            (tensor) waveform: Signal audio data
            (int) sample_rate: Self explanatory
            (tensor) labels: Labels of audio data
            (str) title: Title to display on plot

        Returns:
            plt.show(): outputs a matplotlib display with audio and label data.
        """
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(111)

        waveform = waveform.view(-1)
        waveform = _normalize(waveform)
        time = _time(waveform, sample_rate)
        time_labels = _time_labels(waveform, sample_rate, labels)

        ax.set_title(title or "Labeled Audio", color='#282846')
        ax.set_ylabel('Frequency', color='#282846')
        ax.set_xlabel('Time', color='#282846')
        ax.plot(time, waveform, color='#007580')
        ax.plot(time_labels, labels - 0.5, color='#FED049')
        plt.show()


def _normalize(waveform):
    """
    Normalizes the audio data for cleaner display.

    Params:
        (tensor) waveform: Signal audio data - should be 1d

    Returns:
        normalized waveform
    """
    return waveform / np.max(np.abs(waveform.detach().numpy()),axis=0)

def _time(waveform, sample_rate):
    """
    Gets time values for waveform so that axis can be time in seconds instead of array positions.

    Params:
        (tensor) waveform: Signal audio data - should be 1d
        (int) sample_rate: Self explanatory

    Returns:
        array of timing values that's the same size as waveform.
    """
    return np.linspace(0, len(waveform) / sample_rate, num=len(waveform))

def _time_labels(waveform, sample_rate, labels):
    """
    Gets labels at time values so that it can be plot on top of the waveform.

    Params:
        (tensor) waveform: Signal audio data - should be 1d
        (int) sample_rate: Self explanatory
        (tensor) labels: Labels of audio data
    
    Returns:
        array of labels at every time index, same size as waveform
    """
    return np.linspace(0, len(waveform) / sample_rate, num=len(labels))
