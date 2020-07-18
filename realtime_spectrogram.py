import librosa
import numpy
import scipy.signal
import pyaudio
import time

# Define global variables.
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 1024

SCREEN_WIDTH = 178
ENERGY_THRESHOLD = 0.02

# Choose the frequency range of your log-spectrogram.
N_FFT = 4096
F_LO = librosa.note_to_hz('C2')
F_HI = librosa.note_to_hz('C9')
bank = librosa.filters.mel(RATE, N_FFT, SCREEN_WIDTH, fmin=F_LO, fmax=F_HI)
bank *= scipy.signal.get_window('hann', N_FFT//2+1)[None,:]
bank /= ENERGY_THRESHOLD

def melspectrum_from_audio(audio_data):
    # This function takes one audio buffer as a numpy array and returns a string to be printed to the terminal.
    # Compute real FFT.
    x_fft = numpy.fft.rfft(audio_data, n=N_FFT)

    # Compute mel spectrum.
    melspectrum = bank.dot(abs(x_fft)).astype(int)

    return melspectrum

def show_spectrogram(melspectrum):
    # Initialize output characters to display.
    char_list = [' ']*SCREEN_WIDTH

    for i in range(SCREEN_WIDTH):
        # Draw frequency axis guidelines.
        if (i+1) % 30 == 0:
            char_list[i] = '|'

        # If there is energy in this frequency bin, display an asterisk.
        strength = melspectrum[i]
        if strength == 1:
            char_list[i] = '.'
        elif strength == 2:
            char_list[i] = '+'
        elif strength == 3:
            char_list[i] = '#'
        elif strength >= 4:
            char_list[i] = '@'

    # Return string.
    return ''.join(char_list)

def callback(in_data, frame_count, time_info, status):
    audio_data = numpy.frombuffer(in_data, dtype=numpy.float32)
    spectrum = melspectrum_from_audio(audio_data)
    output_str = show_spectrogram(spectrum)
    print(output_str)
    return in_data, pyaudio.paContinue

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,   # Do record input.
                output=False, # Do not play back output.
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(0.100)

stream.stop_stream()
stream.close()

p.terminate()
