import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib

CUTOFF_FREQUENCY = 3000.0

def movingMean(x, window_size):
    """
    Function that calculates the moving mean of a 1D array given a specific window.
    :param window_size: Window size of a 1D array | 
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def interpretWav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):
    """
    Function for interpreting wav data.
    :param raw_bytes: Raw audio extracted from the multichannel wav file
    :param n_frames: Number of frames of the multichannel wav file
    :param n_channels: Number of channels of the multichannel wav file
    :param sample_width: Number of bits required to represent the value of the multichannel wav file
    :param interleaved: Interleaved of the multichannel wav file
    :return channels
    """
    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # Channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # Channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

def lowPassFiltering(recorded_file):
    """
    Main function that starts the low pass filtering process.
    :param recorded_file: Lung sound file recorded earlier
    """
    with contextlib.closing(wave.open(recorded_file,'rb')) as spf:
        sampRate = spf.getframerate()
        sampWidth = spf.getsampwidth()
        numChannels = spf.getnchannels()
        numFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        rawAudio = spf.readframes(numFrames*numChannels)
        spf.close()
        channels = interpretWav(rawAudio, numFrames, numChannels, sampWidth, True)

        # Get window size
        frequencyRatio = (CUTOFF_FREQUENCY/sampRate)
        N = int(math.sqrt(0.196196 + frequencyRatio**2)/frequencyRatio)

        # Use moving average (only on first channel)
        filtered = movingMean(channels[0], N).astype(channels.dtype)

        wavFile = wave.open("filtered.wav", "w")
        wavFile.setparams((1, sampWidth, sampRate, numFrames, spf.getcomptype(), spf.getcompname()))
        wavFile.writeframes(filtered.tobytes('C'))
        wavFile.close()