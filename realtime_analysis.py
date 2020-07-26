import functools
import itertools
import numpy
import scipy
import scipy.fftpack
import scipy.signal

def hz2mel(f):
    return 2595.0 * numpy.log10(1.0 + f / 700.0)
    # if f < 1000.0:
    #     return f * 0.015
    # else:
    #     return numpy.log(f) * 14.54507850578556 - 85.4738428315498

def mel2hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)
    # if m < 1000.0:
    #     return m / 0.015
    # else:
    #     return 1000.0 * numpy.exp(0.06875177742094912 * m - 1.0312766613142368)

def click(sr, freq=1000.0, decay_time=0.1, length=None):
    if length is None:
        length = int(1.0*sr)
    t = numpy.linspace(0, length/sr, length, endpoint=False, dtype=numpy.float32)
    return 2**(-10*t/decay_time) * numpy.sin(2 * numpy.pi * freq * t)

def clicks(times, duration, sr, buffer_length=1024, freq=1000.0, decay_time=0.1):
    signals = ((t, click(sr, freq, decay_time, int(decay_time*sr))) for t in times)
    return merge(signals, duration, sr, buffer_length)

def mel_freq(fmin, fmax, n_mels, extend=0):
    mel_min = hz2mel(fmin)
    mel_max = hz2mel(fmax)
    m, dm = numpy.linspace(mel_min, mel_max, n_mels, retstep=True, dtype=numpy.float32)
    if extend != 0:
        m_ = mel_max + numpy.arange(1, extend+1, dtype=numpy.float32)*dm
        m = numpy.concatenate((m, m_))
    return mel2hz(m)

def mel_weights(sr, n_fft, n_mels):
    fft_f = numpy.linspace(0, sr/2, n_fft//2+1, dtype=numpy.float32)
    mel_f = mel_freq(0, sr/2, n_mels, 2)

    weights = numpy.empty((n_mels, n_fft//2+1), dtype=numpy.float32)
    for i in range(1, n_mels+1):
        lower = (fft_f - mel_f[i-1]) / (mel_f[i] - mel_f[i-1])
        upper = (mel_f[i+1] - fft_f) / (mel_f[i+1] - mel_f[i])
        weights[i-1] = numpy.maximum(0, numpy.minimum(lower, upper))
        weights[i-1] *= 2 / (mel_f[i+1] - mel_f[i-1])

    return weights

def power2db(power):
    return 10.0 * numpy.log10(numpy.maximum(1e-5, power))

def frame(win_length, hop_length):
    x = numpy.zeros(win_length, dtype=numpy.float32)
    x[-hop_length:] = yield None
    while True:
        x_last = yield x
        x[:-hop_length] = x[hop_length:]
        x[-hop_length:] = x_last

def energy(win_length):
    window = scipy.signal.get_window("hann", win_length)
    
    x = yield None
    while True:
        x = yield power2db((x**2 * window).sum())

def power_freq(sr, win_length):
    window = scipy.signal.get_window("hann", win_length)

    x = yield None
    while True:
        x = yield power2db(numpy.abs(numpy.fft.rfft(x*window))**2)

def power_mel(sr, win_length, n_mels=128):
    window = scipy.signal.get_window("hann", win_length)
    weights = mel_weights(sr, win_length, n_mels)

    x = yield None
    while True:
        x = yield power2db(weights.dot(numpy.abs(numpy.fft.rfft(x*window))**2))

def onset_strength():
    curr = yield None
    prev = numpy.zeros_like(curr)
    while True:
        prev, curr = curr, (yield numpy.maximum(0.0, curr - prev).mean(0))

def onset_detect(sr, hop_length, delay=None,
                 pre_max=0.03, post_max=0.00,
                 pre_avg=0.10, post_avg=0.10,
                 wait=0.03, delta=0.07):
    if delay is None:
        delay = max(post_max, post_avg)

    delay = int(delay * sr / hop_length) + 1
    pre_max  = int(pre_max  * sr / hop_length)
    post_max = int(post_max * sr / hop_length) + 1
    pre_avg  = int(pre_avg  * sr / hop_length)
    post_avg = int(post_avg * sr / hop_length) + 1
    wait = int(wait * sr / hop_length)

    if delay < post_max or delay < post_avg:
        raise ValueError

    center = max(pre_max, pre_avg)
    buf = numpy.zeros(center+delay, dtype=numpy.float32)
    prev = wait+1
    index = 1-delay
    buf[-1] = yield None
    while True:
        index += 1
        detected = True
        detected = detected and buf[center] == buf[center-pre_max:center+post_max].max()
        detected = detected and buf[center] >= buf[center-pre_avg:center+post_avg].mean() + delta
        detected = detected and prev > wait

        prev = 0 if detected else prev+1
        buf[:-1] = buf[1:]
        buf[-1] = yield (index * hop_length / sr, detected)


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

def window(ranges, timespan, offset=0):
    playing = []
    waiting = next(ranges, None)

    time = yield None
    while True:
        t0 = time - offset
        tf = time - offset + timespan

        while waiting is not None and waiting[0] < tf:
            playing.append(waiting)
            waiting = next(ranges, None)

        while len(playing) > 0 and playing[0][1] < t0:
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

        yield buffer.tobytes()

