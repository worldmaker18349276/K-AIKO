import sys
import time
import numpy as np
import scipy
import pyaudio
import realtime_analysis as ra

CHANNELS = 1
RATE = 44100
WIN_LENGTH = 1024
HOP_LENGTH = 512
SAMPLES_PER_BUFFER = 1024

# generate click sound for test
click_bps = 1
click_length = int(RATE/click_bps)
click_signal = ra.click(sr=RATE, length=click_length).astype(np.float32)
click_signal = np.tile(click_signal, 2 * SAMPLES_PER_BUFFER // click_length + 2)

# output stream callback
playback_index = 0
def output_callback(in_data, frame_count, time_info, status):
    global playback_index

    out_data = click_signal[playback_index:playback_index+frame_count].tobytes()
    playback_index = (playback_index + frame_count) % click_length

    return out_data, pyaudio.paContinue

# input stream callback
dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
               ra.power_freq(RATE, WIN_LENGTH),
               ra.onset_strength(),
               ra.onset_detect(RATE, HOP_LENGTH, delta=1))
next(dect)
lag = 0.1

onset_times = [-1]
record_frame = 0
def input_callback(in_data, frame_count, time_info, status):
    global record_frame
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    
    for i in range(0, frame_count, HOP_LENGTH):
        record_frame += 1
        if dect.send(audio_data[i:i+HOP_LENGTH]):
            time = record_frame*HOP_LENGTH/RATE - lag
            onset_times.append(time)
    
    return in_data, pyaudio.paContinue

# terminal output callback
screen_width = 100
bar_offset = 10
drop_speed = 1.0 # sreen per sec
hit_err = (0.02, 0.06, 0.1)
screen_delay = 0.03
def show_callback(current_time):
    current_time -= screen_delay
    view = [" "]*screen_width

    dt = 1 / screen_width / drop_speed
    index_start =            0 - bar_offset
    index_end   = screen_width - bar_offset
    time_start = current_time + index_start * dt
    time_end   = current_time +   index_end * dt
    for click_time in range(int(time_start), int(time_end)+1):
        index = int((click_time - current_time) / dt)
        if index_start <= index < index_end:
            view[index + bar_offset] = "+"

    view[0] = "("
    view[bar_offset] = "|"
    view[-1] = ")"

    hit_time = onset_times[-1]
    if abs(hit_time - current_time) < 0.3:
        err = hit_time - round(hit_time)
        if abs(err) < hit_err[0]:
            view[bar_offset] = "I"
        elif abs(err) < hit_err[1]:
            view[bar_offset] = "]" if err < 0 else "["
        elif abs(err) < hit_err[2]:
            view[bar_offset] = ">" if err < 0 else "<"
        else:
            view[bar_offset] = ":"

    sys.stdout.write("".join(view) + "\r")
    sys.stdout.flush()

# execute test
reference_time = time.time()

p = pyaudio.PyAudio()

input_stream = p.open(format=pyaudio.paFloat32,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      output=False,
                      frames_per_buffer=SAMPLES_PER_BUFFER,
                      stream_callback=input_callback)
input_stream.start_stream()

output_stream = p.open(format=pyaudio.paFloat32,
                       channels=CHANNELS,
                       rate=RATE,
                       input=False,
                       output=True,
                       frames_per_buffer=SAMPLES_PER_BUFFER,
                       stream_callback=output_callback)
output_stream.start_stream()

while output_stream.is_active() and input_stream.is_active():
    time.sleep(0.005)
    show_callback(time.time() - reference_time)

output_stream.stop_stream()
input_stream.stop_stream()
output_stream.close()
input_stream.close()

p.terminate()
