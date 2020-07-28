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
beatmap_scores = []
def judge(beatmap, judgements, scores, offset=0):
    beatmap = iter(beatmap)
    beat_time = next(beatmap, None)

    time, detected = yield
    while True:
        time += offset
        while beat_time is not None and beat_time + judgements[0] <= time:
            scores.append((0, None))
            beat_time = next(beatmap, None)

        if detected and beat_time is not None and beat_time - judgements[0] <= time:
            err = time - beat_time
            for i, judgement in list(enumerate(judgements))[::-1]:
                if abs(err) < judgement:
                    scores.append((i, err))
                    break
            else:
                scores.append((0, err))
            beat_time = next(beatmap, None)

        time, detected = yield

pre_max = 0.03
post_max = 0.00
pre_avg = 0.10
post_avg = 0.10
wait = 0.03
delta = 0.1
dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
               ra.power_spectrum(RATE, WIN_LENGTH),
               ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
               ra.onset_detect(RATE, HOP_LENGTH,
                               pre_max=pre_max, post_max=post_max,
                               pre_avg=pre_avg, post_avg=post_avg,
                               wait=wait, delta=delta),
               judge(beatmap, beatmap_judgements, beatmap_scores))
next(dect)

def input_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    
    for i in range(0, frame_count, HOP_LENGTH):
        dect.send(audio_data[i:i+HOP_LENGTH])
    
    return in_data, pyaudio.paContinue


# screen output callback
screen_width = 100
bar_offset = 10
drop_speed = 1.0 # sreen per sec

def draw_view(beatmap, screen_width, bar_offset, drop_speed):
    dt = 1 / screen_width / drop_speed
    windowed_beatmap = ra.window(((int(t/dt),)*2 + (i,) for i, t in enumerate(beatmap)), screen_width, bar_offset)
    next(windowed_beatmap)

    current_time = yield None
    while True:
        view = [" "]*screen_width

        current_pixel = int(current_time/dt)
        for beat_pixel, _, beat_index in windowed_beatmap.send(current_pixel):
            # if beat_index >= len(beatmap_scores) or beatmap_scores[beat_index][0] is None:
            view[beat_pixel - current_pixel + bar_offset] = "+" # "░▄▀█", "○◎◉●"

        view[0] = "("
        view[bar_offset] = "|"
        view[-1] = ")"

        if len(beatmap_scores) > 0 and beatmap_scores[-1][1] is not None:
            beat_time = beatmap[len(beatmap_scores)-1]
            (score, err) = beatmap_scores[-1]
            if abs(beat_time + err - current_time) < 0.3:
                if score == 3:
                    view[bar_offset] = "I"
                elif score == 2:
                    view[bar_offset] = "]" if err < 0 else "["
                elif score == 1:
                    view[bar_offset] = ">" if err < 0 else "<"
                else:
                    view[bar_offset] = ":"

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

