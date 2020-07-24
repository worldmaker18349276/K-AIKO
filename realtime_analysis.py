import functools
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

def mel_fft(sr, win_length, n_mels=128):
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

def onset_detect(sr, hop_length, lag=None,
                 pre_max=0.03, post_max=0.00,
                 pre_avg=0.10, post_avg=0.10,
                 wait=0.03, delta=0.07):
    if lag is None:
        lag = max(post_max, post_avg)

    lag = int(lag * sr / hop_length) + 1
    pre_max  = int(pre_max  * sr / hop_length)
    post_max = int(post_max * sr / hop_length) + 1
    pre_avg  = int(pre_avg  * sr / hop_length)
    post_avg = int(post_avg * sr / hop_length) + 1
    wait = int(wait * sr / hop_length)

    if lag < post_max or lag < post_avg:
        raise ValueError

    center = max(pre_max, pre_avg)
    buf = numpy.zeros(center+lag, dtype=numpy.float32)
    prev = wait+1
    buf[-1] = yield None
    while True:
        detected = True
        detected = detected and buf[center] == buf[center-pre_max:center+post_max].max()
        detected = detected and buf[center] >= buf[center-pre_avg:center+post_avg].mean() + delta
        detected = detected and prev > wait

        prev = 0 if detected else prev+1
        buf[:-1] = buf[1:]
        buf[-1] = yield detected


def pipe(*iters):
    for it in iters:
        next(it)
    input = yield None
    while True:
        input = yield functools.reduce(lambda input, it: it.send(input), iters, input)

def pair(*iters):
    for it in iters:
        next(it)
    inputs = yield None
    while True:
        inputs = yield tuple(map(lambda input, it: it.send(input), inputs, iters))

def transform(func=lambda a: a):
    arr = yield None
    while True:
        arr = yield func(arr)


def peak_pick(buffer_length, pre_max, post_max, pre_avg, post_avg, delta, wait):
    max_length = pre_max + post_max
    max_origin = (1 - max_length) // 2
    avg_length = pre_avg + post_avg
    avg_origin = (1 - avg_length) // 2
    tail = max(max_origin-max_length, avg_origin-avg_length)

    mov_max = scipy.ndimage.filters.maximum_filter1d(x, max_length, mode="constant", origin=max_origin, cval=0)
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, avg_length, mode="constant", origin=avg_origin, cval=0)
    x = x[:-tail]
    mov_max = mov_max[:-tail]
    mov_avg = mov_avg[:-tail]

    detections = x * (x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)