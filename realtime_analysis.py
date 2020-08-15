import functools
import itertools
import numpy
import scipy
import scipy.fftpack
import scipy.signal

def click(sr, freq=1000.0, decay_time=0.1, amplitude=1.0, length=None):
    if length is None:
        length = decay_time
    t = numpy.linspace(0, length, int(length*sr), endpoint=False, dtype=numpy.float32)
    return amplitude * 2**(-10*t/decay_time) * numpy.sin(2 * numpy.pi * freq * t)

def power2db(power, scale=(1e-5, 1e8)):
    return 10.0 * numpy.log10(numpy.maximum(scale[0], power*scale[1]))

def frame(win_length, hop_length):
    x = numpy.zeros(win_length, dtype=numpy.float32)
    x[-hop_length:] = yield None
    while True:
        x_last = yield x
        x[:-hop_length] = x[hop_length:]
        x[-hop_length:] = x_last

def power_spectrum(sr, win_length):
    window = scipy.signal.get_window("hann", win_length)

    x = yield None
    while True:
        x = yield 2/win_length/sr * numpy.abs(numpy.fft.rfft(x*window))**2
        # (J * df).sum() == (x**2).mean()

def onset_strength(df, dt):
    curr = yield None
    prev = numpy.zeros_like(curr)
    while True:
        prev, curr = curr, (yield numpy.maximum(0.0, (curr - prev)/dt  * df).sum(0))

def onset_detect(sr, hop_length,
                 pre_max=0.03, post_max=0.00,
                 pre_avg=0.10, post_avg=0.10,
                 wait=0.03, delta=0.07):
    pre_max  = int(pre_max  * sr / hop_length)
    post_max = int(post_max * sr / hop_length)
    pre_avg  = int(pre_avg  * sr / hop_length)
    post_avg = int(post_avg * sr / hop_length)
    wait = int(wait * sr / hop_length)

    center = max(pre_max, pre_avg)
    delay = max(post_max, post_avg)
    buffer = numpy.zeros(center+delay+1, dtype=numpy.float32)
    max_buffer = buffer[center-pre_max:center+post_max+1]
    avg_buffer = buffer[center-pre_avg:center+post_avg+1]
    index = -delay
    prev_index = -wait

    buffer[-1] = yield None
    while True:
        index += 1
        strength = buffer[center]
        detected = True
        detected = detected and index > prev_index + wait
        detected = detected and strength == max_buffer.max()
        detected = detected and strength >= avg_buffer.mean() + delta

        if detected:
            prev_index = index
        buffer[:-1] = buffer[1:]
        buffer[-1] = yield (index * hop_length / sr, strength, detected)

def pipe(*iters):
    for it in iters:
        next(it)
    data = yield None
    while True:
        data = yield functools.reduce(lambda data, it: it.send(data), iters, data)

def pair(*iters):
    for it in iters:
        next(it)
    data = yield None
    while True:
        data = yield tuple(map(lambda subdata, it: it.send(subdata), data, iters))

def transform(func=lambda i, a: a):
    index = 0
    data = yield None
    while True:
        index += 1
        data = yield func(index, data)

def inwhich(*iters):
    it = pipe(*iters)
    data = yield None
    while True:
        side = it.send(data)
        data = yield (data, side)

def whenever(func, cond=bool):
    index = 0
    data = yield None
    while True:
        index += 1
        if cond(data):
            func(index, data)
        data = yield data

def window(it, timespan, offset=0, key=lambda item: item):
    it = iter(it)
    playing = []
    waiting = next(it, None)

    time = yield None
    while True:
        t0 = time - offset
        tf = time - offset + timespan

        while waiting is not None and key(waiting)[0] < tf:
            playing.append(waiting)
            waiting = next(it, None)

        while len(playing) > 0 and key(playing[0])[1] < t0:
            playing.pop(0)

        time = yield playing

def merge(signals, duration, sr, buffer_length=1024):
    buffer = numpy.zeros(buffer_length, dtype=numpy.float32)
    windowed_signals = window(((int(t*sr), int(t*sr)+signal.shape[0], signal) for t, signal in signals), buffer_length)
    next(windowed_signals)

    yield None

    for index in range(0, int(duration*sr), buffer_length):
        buffer[:] = 0

        for start, end, signal in windowed_signals.send(index):
            i = max(start, index)
            j = min(end, index+buffer_length)
            buffer[i-index:j-index] += signal[i-start:j-start]

        yield buffer

def load(file, sr=None, buffer_length=1024):
    # file: wave.Wave_read
    width = file.getsampwidth()
    nchannels = file.getnchannels()
    samplerate = file.getframerate()
    sample_length = file.getnframes()

    if sr is None:
        sr = samplerate
    scale = 2.0 ** (1 - 8*width)
    fmt = "<i{:d}".format(width)

    buffer_length_ = buffer_length * samplerate // sr
    buf_ = numpy.zeros(buffer_length_*3, dtype=numpy.float32)
    N = sample_length // buffer_length_
    tail_length = ((sample_length - 1) % buffer_length_ + 1) * buffer_length // buffer_length_

    yield None

    for n in range(-1, N+1):
        start_ = buffer_length_ * (n % 3)
        buf_[start_:start_+buffer_length_] = 0.0

        if n <= N:
            x = file.readframes(buffer_length_)
            x = scale * numpy.frombuffer(x, fmt).astype(numpy.float32)
            if nchannels > 1:
                x = x.reshape((-1, nchannels)).mean(axis=1)
            buf_[start_:start_+x.shape[-1]] = x

        if n == -1:
            continue

        buf = scipy.signal.resample(buf_, buffer_length*3) if sr != samplerate else buf_
        start = buffer_length * ((n + 2) % 3)
        if n == N:
            buf[start+tail_length:start+buffer_length] = 0.0
        yield buf[start:start+buffer_length]

