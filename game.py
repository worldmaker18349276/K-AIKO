import sys
import time
import numpy as np
import scipy
import pyaudio
import realtime_analysis as ra

CHANNELS = 1
RATE = 44100
WIN_LENGTH = 512*4
HOP_LENGTH = 512
SAMPLES_PER_BUFFER = 1024
# frame resolution: 11.6 ms

beatmap = [1.0,  2.0,  3.0,  3.5,  4.0,
           5.0,  6.0,  7.0,  7.5,  8.0,
           9.0,  10.0, 11.0, 11.5, 12.0,
           13.0, 14.0, 15.0, 15.5, 16.0]
duration = 16.5


# output stream callback
signal_gen = ra.clicks(beatmap, sr=RATE, duration=duration)
next(signal_gen, None)

def output_callback(in_data, frame_count, time_info, status):
    out_data = next(signal_gen, None)
    return out_data, pyaudio.paContinue


# input stream callback
beatmap_judgements = (0.14, 0.10, 0.06, 0.02)
beatmap_threshold = 2.0
beatmap_scores = []
def judge(beatmap, judgements, threshold, scores):
    beatmap = enumerate(beatmap)
    beat_index, beat_time = next(beatmap, (None, None))

    time, strength, detected = yield
    while True:
        while beat_index is not None and beat_time + judgements[0] <= time:
            scores.append((beat_index, 0, 0, time))
            beat_index, beat_time = next(beatmap, (None, None))

        if detected:
            key = 2 if strength >= threshold else 1
            
            if beat_index is not None and beat_time - judgements[0] <= time:
                err = time - beat_time
                accuracy = next((i for i, acc in reversed(list(enumerate(judgements))) if abs(err) < acc), 0)
                
                scores.append((beat_index, accuracy, key, time))
                
                beat_index, beat_time = next(beatmap, (None, None))
                
            else:
                scores.append((None, 0, key, time))

        time, strength, detected = yield

pre_max = 0.03
post_max = 0.03
pre_avg = 0.03
post_avg = 0.03
wait = 0.03
delta = 0.1
dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
               ra.power_spectrum(RATE, WIN_LENGTH),
               ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
               ra.onset_detect(RATE, HOP_LENGTH,
                               pre_max=pre_max, post_max=post_max,
                               pre_avg=pre_avg, post_avg=post_avg,
                               wait=wait, delta=delta),
               judge(beatmap, beatmap_judgements, beatmap_threshold, beatmap_scores))
next(dect)

def input_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    
    for i in range(0, frame_count, HOP_LENGTH):
        dect.send(audio_data[i:i+HOP_LENGTH])
    
    return in_data, pyaudio.paContinue


# screen output callback
screen_width = 100
bar_offset = 10
drop_speed = 100.0 # pixels per sec

def draw_view(beatmap, screen_width, bar_offset, drop_speed):
    dt = 1 / drop_speed
    windowed_beatmap = ra.window(((int(t/dt),)*2 + (i,) for i, t in enumerate(beatmap)), screen_width, bar_offset)
    next(windowed_beatmap)

    current_time = yield None
    while True:
        view = [" "]*screen_width

        current_pixel = int(current_time/dt)
        
        hitted = [i for i, acc, key, t in beatmap_scores if i is not None and key is not 0]
        for beat_pixel, _, beat_index in windowed_beatmap.send(current_pixel):
            if beat_index not in hitted:
                view[beat_pixel - current_pixel + bar_offset] = "+" # "░▄▀█", "□ ▣ ■"

        view[0] = "("
        view[bar_offset] = "|"
        view[-1] = ")"

        if len(beatmap_scores) > 0:
            beat_index, accuracy, key, hit_time = beatmap_scores[-1]
            if abs(hit_time - current_time) < 0.2:
                if key == 1:
                    view[bar_offset] = "I"
                elif key == 2:
                    view[bar_offset] = "H"
                    
        current_time = yield "".join(view)

drawer = draw_view(beatmap, screen_width=screen_width, bar_offset=bar_offset, drop_speed=drop_speed)
next(drawer)

def view_callback(current_time):
    view = drawer.send(current_time)
    sys.stdout.write(view + "\r")
    sys.stdout.flush()
    pass


# execute test
view_fps = 200
view_delay = -0.03

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
    view_callback(time.time() - reference_time + view_delay)
    time.sleep(1/view_fps)

output_stream.stop_stream()
input_stream.stop_stream()
output_stream.close()
input_stream.close()

p.terminate()

# print scores
print()
for score in beatmap_scores:
    print(score)

