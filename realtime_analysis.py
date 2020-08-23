import functools
import itertools
import numpy
import scipy
import scipy.fftpack
import scipy.signal


class DataNode:
    def __init__(self, function):
        self.function = function

    def send(self, value):
        return self.function(value)

    def __next__(self):
        return self.function(None)

    def __iter__(self):
        return self

    @staticmethod
    def from_generator(generator):
        @functools.wraps(generator)
        def node_builder(*args, **kwargs):
            gen = generator(*args, **kwargs)
            init_data = next(gen)

            node = DataNode(gen.send)
            if isinstance(init_data, dict):
                for name in init_data:
                    setattr(node, name, init_data[name])

            return node

        return node_builder


@DataNode.from_generator
def delay(*prepend):
    """A data node delays signal and prepends given values.

    Parameters
    ----------
    prepend : list
        The list of prepended values.

    Attributes
    ----------
    delay : int
        The delayed period of this node.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : any
        The delayed signal.
    """
    buffer = list(prepend)
    data = yield dict(delay=len(buffer))
    while True:
        buffer.append(data)
        data = yield buffer.pop(0)

@DataNode.from_generator
def take(number):
    """A data node takes finite signals.

    Parameters
    ----------
    number : int
        The number of period to take.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : any
        The output signal.
    """
    data = yield None
    for _ in range(number):
        data = yield data

@DataNode.from_generator
def pipe(*nodes):
    """A data node processes data sequentially.

    Parameters
    ----------
    nodes : list
        The data nodes to pipe.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : any
        The processed signal.
    """
    data = yield None
    while True:
        data = yield functools.reduce(lambda data, node: node.send(data), nodes, data)

@DataNode.from_generator
def pair(*nodes):
    """A data node processes data parallelly.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to pair.

    Receives
    --------
    x : tuple
        The input signal; its length should equal to number of nodes.

    Yields
    ------
    x : tuple
        The processed signal; its length should equal to number of nodes.
    """
    data = yield None
    while True:
        data = yield tuple(node.send(subdata) for node, subdata in zip(nodes, data))

@DataNode.from_generator
def branch(*nodes):
    """A data node processes data additionally.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to branch.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : any
        The input signal.
    """
    data = None
    while True:
        data = yield data
        for node in nodes:
            node.send(data)

@DataNode.from_generator
def merge(*nodes):
    """A data node processes additional data.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to merge.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : tuple
        The input signal and additional data.
    """
    data = yield None
    while True:
        data = yield (data,) + tuple(next(node) for node in nodes)

@DataNode.from_generator
def drip(items, schedule):
    """A data node to fetch scheduled items chronologically.

    Parameters
    ----------
    items : list
        The items to fetch.

    schedule : function
        A function to schedule each item, which should return start/end time in a tuple.

    Receives
    --------
    x : any
        The current time to fetch items, which should greater than previous received time.

    Yields
    ------
    x : list
        The items occurred in the given time.
    """
    it = iter(sorted((schedule(item), item) for item in items))

    buffer = []
    waiting = next(it, None)

    time = yield None
    while True:
        while waiting is not None and waiting[0][0] < time:
            buffer.append(waiting)
            waiting = next(it, None)

        buffer = [playing for playing in buffer if playing[0][1] >= time]

        time = yield [item for _, item in buffer]


@DataNode.from_generator
def empty(sr, buffer_length=1024, duration=None):
    """A data node produces empty data.

    Parameters
    ----------
    sr : int
        The sample rate.
    buffer_length : int, optional
        The length of data, default is `1024`.
    duration : float, optional
        The duration of signal.

    Yields
    ------
    x : ndarray
        The empty signal with length `buffer_length`.
    """
    yield None
    if duration is None:
        while True:
            yield numpy.zeros(buffer_length, dtype=numpy.float32)
    else:
        for _ in range(int(duration*sr/buffer_length)+1):
            yield numpy.zeros(buffer_length, dtype=numpy.float32)

@DataNode.from_generator
def load(file, sr=None, buffer_length=1024):
    """A data node to load .wav file with given sample rate.

    Parameters
    ----------
    file : wave.Wave_read
        The wave file to load.
    sr : int, optional
        The sample rate to load, default is the same as given file.
    buffer_length : int, optional
        The length of output signal, default is `1024`.

    Yields
    ------
    x : ndarray
        The loaded signal.
    """
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
        yield numpy.copy(buf[start:start+buffer_length])

@DataNode.from_generator
def save(file, samplerate, width=2):
    scale = 2.0 ** (8*width - 1)
    fmt = "<i{:d}".format(width)

    file.setsampwidth(width)
    file.setnchannels(1)
    file.setframerate(samplerate)
    file.setnframes(0)

    while True:
        file.writeframes(((yield) * scale).astype(fmt).tobytes())

@DataNode.from_generator
def slice(signal, buffer_length=1024):
    """A data node produces data by slicing given signal.

    Parameters
    ----------
    signal : ndarray
        The signal to slice.
    buffer_length : int, optional
        The length of slicing, default is `1024`.

    Yields
    ------
    x : ndarray
        The sliced signal with length `buffer_length`.
    """
    yield None
    for i in range(0, len(signal), buffer_length):
        sliced = signal[i:i+buffer_length]
        yield numpy.pad(sliced, (0, buffer_length - sliced.shape[0]), "constant", constant_values=(0, 0))

@DataNode.from_generator
def attach(scheduled_signals, sr, buffer_length=1024):
    """A data node attaches scheduled signals to input signal.

    Parameters
    ----------
    scheduled_signals : list
        The list of scheduled signals, composed by tuples of scheduled time and data.
    sr : int
        The sample rate to load.
    buffer_length : int, optional
        The length of input signal, default is `1024`.

    Receives
    --------
    x : ndarray
        The input signal.

    Yields
    ------
    x : ndarray
        The processed signal.
    """
    dripping_signals = drip(scheduled_signals, lambda a: (a[0]*sr - buffer_length, a[0]*sr + len(a[1])))

    buffer = None
    for index in itertools.count(0, buffer_length):
        buffer = yield buffer

        for start, signal in dripping_signals.send(index):
            i = max(start, index)
            j = min(start+len(signal), index+buffer_length)
            buffer[i-index:j-index] += signal[i-start:j-start]


@DataNode.from_generator
def frame(win_length, hop_length):
    """A data node to frame signal, prepend by zero.

    Parameters
    ----------
    win_length : int
        The length of framed data.
    hop_length : int
        The length of input data, it should be shorter than input data.

    Receives
    --------
    x : ndarray
        The input signal.

    Yields
    ------
    x : ndarray
        The framed signal.
    """
    x = numpy.zeros(win_length, dtype=numpy.float32)
    x[-hop_length:] = yield None
    while True:
        x_last = yield numpy.copy(x)
        x[:-hop_length] = x[hop_length:]
        x[-hop_length:] = x_last

@DataNode.from_generator
def power_spectrum(sr, win_length, windowing=True, weighting=True):
    """A data node maps signal `x` to power spectrum `J`.

    Without windowing and weighting, they should satisfy

        (J * df).sum() == (x**2).mean()

    where the time resolution `dt = 1/sr` and the frequency resolution `df = sr/win_length`.

    Parameters
    ----------
    sr : int
        The sample rate of input signal.
    win_length : int
        The length of input signal.
    windowing : bool or ndarray, optional
        The window function of signal, `True` for default Hann window, `False` for no windowing.
    weighting : bool or ndarray, optional
        The weight function of spectrum, `True` for default A-weighting, `False` for no weighting.

    Receives
    --------
    x : ndarray
        The input signal.

    Yields
    ------
    J : ndarray
        The power spectrum, with length `win_length//2+1`.
    """
    if isinstance(windowing, bool):
        windowing = get_Hann_window(win_length) if windowing else 1
    if isinstance(weighting, bool):
        weighting = get_A_weight(sr, win_length) if weighting else 1
    weighting *= 2/win_length/sr

    x = yield None
    while True:
        x = yield weighting * numpy.abs(numpy.fft.rfft(x*windowing))**2

@DataNode.from_generator
def onset_strength(df, Dt):
    """A data node maps spectrum `J` to onset strength `st`.

    Parameters
    ----------
    df : float
        The frequency resolution of input spectrum.
    Dt : float
        The time interval between each period.

    Receives
    --------
    J : ndarray
        Input spectrum.

    Yields
    ------
    st : float
        The onset strength between previous and current input spectrum.
    """
    curr = yield None
    prev = numpy.zeros_like(curr)
    while True:
        prev, curr = curr, (yield numpy.maximum(0.0, curr - prev).sum(0) * df / Dt)

@DataNode.from_generator
def pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta):
    """A data node of peak detaction.

    Parameters
    ----------
    pre_max : int
    post_max : int
    pre_avg : int
    post_avg : int
    wait : int
    delta : float

    Attributes
    ----------
    delay : int
        The delay period of this node.

    Receives
    --------
    y : float
        The input signal.

    Yields
    ------
    detected : bool
        Whether the signal reaches its peak.
    """
    center = max(pre_max, pre_avg)
    delay = max(post_max, post_avg)
    buffer = numpy.zeros(center+delay+1, dtype=numpy.float32)
    max_buffer = buffer[center-pre_max:center+post_max+1]
    avg_buffer = buffer[center-pre_avg:center+post_avg+1]
    index = -delay
    prev_index = -wait

    buffer[-1] = yield dict(delay=delay)
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
        buffer[-1] = yield detected

@DataNode.from_generator
def draw_spectrum(length, sr, win_length, scale=0.01):
    """A data node to show given spectrum by braille patterns.

    Parameters
    ----------
    length : int
        The length of string.
    sr : int
        The sample rate of input signal.
    win_length : int
        The length of input signal before fourier transform.

    Receives
    --------
    J : ndarray
        The power spectrum to draw.

    Yields
    ------
    spec : str
        The string representation of spectrum.
    """
    A = numpy.cumsum([0, 2**6, 2**2, 2**1, 2**0])
    B = numpy.cumsum([0, 2**7, 2**5, 2**4, 2**3])

    df = sr/win_length
    n_fft = win_length // 2 + 1

    sec = [0] + [2**i for i in range(length*2)]
    sec = [n_fft * s // sec[-1] for s in sec]
    slices = list(zip(sec[:-1], sec[1:]))

    def vol(J, start, end):
        return power2db(J[start:end].sum() * df * n_fft/(end-start) * scale)

    J = yield None
    while True:
        buf = [max(0, min(4, int(vol(J, start, end) / 80 * 4))) for start, end in slices]
        J = yield "".join(chr(0x2800 + A[a] + B[b]) for a, b in zip(buf[0::2], buf[1::2]))


def click(sr, freq=1000.0, decay_time=0.1, amplitude=1.0, length=None):
    if length is None:
        length = decay_time
    t = numpy.linspace(0, length, int(length*sr), endpoint=False, dtype=numpy.float32)
    return amplitude * 2**(-10*t/decay_time) * numpy.sin(2 * numpy.pi * freq * t)

def power2db(power, scale=(1e-5, 1e8)):
    return 10.0 * numpy.log10(numpy.maximum(scale[0], power*scale[1]))

def get_Hann_window(win_length):
    a = numpy.linspace(0, numpy.pi, win_length)
    window = numpy.sin(a)**2
    gain = (3/8)**0.5 # (window**2).mean()**0.5
    return window / gain

def get_A_weight(sr, win_length):
    f = numpy.arange(win_length//2+1) * (sr/win_length)

    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    weight  = (f**4 * f4**2)**2
    weight /= (f**2 + f1**2)**2
    weight /= (f**2 + f2**2)
    weight /= (f**2 + f3**2)
    weight /= (f**2 + f4**2)**2

    # normalize on 1000 Hz
    f0 = 1000.0
    weight0  = (f0**4 * f4**2)**2
    weight0 /= (f0**2 + f1**2)**2
    weight0 /= (f0**2 + f2**2)
    weight0 /= (f0**2 + f3**2)
    weight0 /= (f0**2 + f4**2)**2
    # weight0 == 10**-0.1

    weight /= weight0

    return weight

