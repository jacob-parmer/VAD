"""
### Author: Jacob Parmer
###
### Created: Mar 31, 2021
"""

import librosa
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt

class Signal:
    """
    Represents an audio signal.
    """

    def __init__(self, waveform, sample_rate):
        """
        Initializes the audio signal, with both waveform and sample rate data.

        Params:
            (tensor) waveform: [1, length] tensor representative of audio data
            (int) sample_rate: Self explanatory
        """

        self.waveform = waveform
        self.sample_rate = sample_rate
        return

    def split_into_frames(self, frame_size=1024):
        """
        Splits the audio signal into discrete segments. This is primarily for ease of labelling the data.
        Has to pad the last bit of the waveform so that it has a consistent size to the other frames.

        Params:
            (int) frame_size: Self explanatory

        Returns:
            (tensor) torch.reshape(padded_waveform, (-1, frame_size)): 2d representation of waveform with shape [?, ?]
        """
        tot_length = self.waveform.size(1)
        pad_size = (frame_size - (tot_length % frame_size)) - 1 # For some reason functional.pad adds pad_size+1, so -1 here (hacky solution, fix later?)

        if pad_size != -1:
            pad = (1,pad_size) 
            padded_waveform = torch.nn.functional.pad(self.waveform, pad, "constant", 0)

        return torch.reshape(padded_waveform, (-1, frame_size))

    # ------------------------- FEATURE EXTRACTIONS --------------------------- #
    """
        Not going to write out in-depth comments on the rest of these functions,
        since they were largely copy-pasted and much more information could be found at:
        https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#feature-extractions
    """
    def get_MFCC(self, hop_length, n_fft=2048, win_length=None, n_mels=256, n_mfcc=256, verbose=False):
        mfcc_transform = T.MFCC(sample_rate=self.sample_rate,
                                n_mfcc=n_mfcc,
                                melkwargs={'n_fft':n_fft, 'n_mels':n_mels, 'hop_length': hop_length}
                                )

        mfcc = mfcc_transform(self.waveform)

        if verbose:
            _plot_spectrogram(mfcc[0], title="Mel Frequency Cepstrum")

        return mfcc

    def get_mfilterbank(self, n_fft=1024, n_mels=64, verbose=False):
        mel_filters = F.create_fb_matrix(int(n_fft // 2 + 1),
                                         n_mels=n_mels,
                                         f_min=0.,
                                         f_max=self.sample_rate/2.,
                                         sample_rate=self.sample_rate,
                                         norm='slaney'
                                         )

        if verbose:
            _plot_mel_fbank(mel_filters, "Mel Filter Bank")

        return mel_filters

    def get_spectrogram(self, n_fft=1024, win_length=None, hop_length=512, verbose=False):
        spectrogram = T.Spectrogram(n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    center=True,
                                    pad_mode="reflect",
                                    power=2.0
                                    )
        spec = spectrogram(self.waveform)
        
        if verbose:
            _plot_spectrogram(spec[0], title='Spectrogram (dB)')

        return spec


    def get_pitch(self, verbose=False):
        pitch_feature = F.compute_kaldi_pitch(self.waveform, self.sample_rate)
        pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]

        if verbose:
            _plot_kaldi_pitch(self.waveform, self.sample_rate, pitch, nfcc)

        return pitch, nfcc


    # -------------- DISPLAY -------------- #
    def print_stats(self):
        print(f"Sample Rate: {self.sample_rate}")
        print(f"Shape: {tuple(self.waveform.shape)}")
        print(f"Dtype: {self.waveform.dtype}")
        print(f"- Max:      {self.waveform.max().item():6.3f}")
        print(f"- Min:      {self.waveform.min().item():6.3f}")
        print(f"- Mean:     {self.waveform.mean().item():6.3f}")
        print(f"- Std Dev:  {self.waveform.std().item():6.3f}")
        print(self.waveform)


# ------------------- HELPER FUNCTIONS ------------------------#
def _plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

def _plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show()

def _plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Kaldi Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time,  waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
    axis.set_ylim((-1.3, 1.3))

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, nfcc.shape[1])
    ln2 = axis2.plot(
        time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    axis.legend(lns, labels, loc=0)
    plt.show()