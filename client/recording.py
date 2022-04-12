import filtering
import sounddevice
import numpy as np
from scipy.io.wavfile import write

CYCLES_PER_SECOND = 22050
AUDIO_LENGTH = 10

def record_sound():
    """
    For recording a lung sound clip using intergrated microphone
    """
    print("Recording started!")

    # Record audio data from your sound device into a NumPy array
    recording = sounddevice.rec(int(AUDIO_LENGTH*CYCLES_PER_SECOND), samplerate=CYCLES_PER_SECOND, channels=2, dtype=np.int16)
    # Check if the recording is finished
    sounddevice.wait()

    print("Recording stopped!")

    # Write a NumPy array as a WAV file in Int16 format.
    write("sound.wav", CYCLES_PER_SECOND, recording)

    # Start low pass filtering
    filtering.lowPassFiltering("sound.wav")

    return True