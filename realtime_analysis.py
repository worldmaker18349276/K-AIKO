import functools
import itertools
import contextlib
import wave
import numpy
import scipy
import scipy.fftpack
import scipy.signal
import pyaudio
# import audioread
# with audioread.audio_open(filename) as f:
#     print(f.channels, f.samplerate, f.duration)
#     for buf in f:
#         do_something(buf)


class DataNode:
    def __init__(self, gen):
        self.gen = gen
        self.started = False
        self.stoped = False

    def send(self, value=None):
        if not self.started:
            raise RuntimeError("try to access un-initialized data node")
        if self.stoped:
            raise RuntimeError("try to access finalized data node")

        return self.gen.send(value)

    def __next__(self):
        return self.send()

    def __iter__(self):
        return self

    def __enter__(self):
        if self.started:
            return self
        self.started = True

        try:
            init_data = next(self.gen)

            if isinstance(init_data, dict):
                for name in init_data:
                    setattr(self, name, init_data[name])

            return self

        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, type=None, value=None, traceback=None):
        if not self.started:
            raise RuntimeError("try to finalize un-initialized data node")
        if self.stoped:
            return False
        self.stoped = True

        if type is None or type is StopIteration:
            self.gen.close()
            return False

        try:
            if value is None:
                value = type()
            self.gen.throw(type, value, traceback)

        except BaseException as exc:
            if exc is value:
                return False
            if isinstance(exc, StopIteration):
                return True
            raise

        else:
            raise RuntimeError("generator didn't stop after throw()")

    @staticmethod
    def from_generator(func):
        @functools.wraps(func)
        def node_builder(*args, **kwargs):
            return DataNode(func(*args, **kwargs))

        return node_builder

    @staticmethod
    def wrap(arg):
        if isinstance(arg, DataNode):
            return arg

        elif hasattr(arg, "__next__"):
            def iterator():
                yield
                while True:
                    yield next(arg)
            return DataNode(iterator())

        else:
            def pure_func():
                data = yield
                while True:
                    data = yield arg(data)
            return DataNode(pure_func())


@DataNode.from_generator
def delay(prepend):
    """A data node delays signal and prepends given values.

    Parameters
    ----------
    prepend : list or int
        The list of prepended values or number of delay with prepending `None`.

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
    buffer = [None]*prepend if isinstance(prepend, int) else list(prepend)
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
    data = yield
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
    nodes = list(map(DataNode.wrap, nodes))
    with contextlib.ExitStack() as stack:
        for node in nodes:
            stack.enter_context(node)
        data = yield
        while True:
            data = yield functools.reduce((lambda data, node: node.send(data)), nodes, data)

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
    nodes = list(map(DataNode.wrap, nodes))
    with contextlib.ExitStack() as stack:
        for node in nodes:
            stack.enter_context(node)
        data = yield
        while True:
            data = yield tuple(node.send(subdata) for node, subdata in zip(nodes, data))

@DataNode.from_generator
def branch(*nodes):
    """A data node processes data additionally.

    Parameters
    ----------
    nodes : list of DataNode
        The sequence of data nodes to branch.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : any
        The input signal.
    """
    node = pipe(*nodes)
    with node:
        data = yield
        while True:
            node.send(data)
            data = yield data

@DataNode.from_generator
def merge(*nodes):
    """A data node processes additional data.

    Parameters
    ----------
    nodes : list of DataNode
        The sequence of data nodes to merge.

    Receives
    --------
    x : any
        The input signal.

    Yields
    ------
    x : tuple
        The input signal and additional data.
    """
    node = pipe(*nodes)
    with node:
        data = yield
        while True:
            data = yield (data, next(node))


@DataNode.from_generator
def load(filename, buffer_length=1024, samplerate=44100, start=None, end=None):
    """A data node to load .wav file with given sample rate.

    Parameters
    ----------
    filename : str
        The wave file to load.
    buffer_length : int, optional
        The length of output signal, default is `1024`.
    samplerate : int, optional
        The sample rate to load, default is `44100`.
    start : float, optional
        The start time to load.
    end : float, optional
        The end time to load.

    Yields
    ------
    x : ndarray
        The loaded signal.
    """
    with wave.open(filename, "rb") as file:
        width = file.getsampwidth()
        nchannels = file.getnchannels()
        file_samplerate = file.getframerate()
        sample_length = file.getnframes()

        scale = 2.0 ** (1 - 8*width)
        fmt = "<i{:d}".format(width)

        buffer_length_ = buffer_length * file_samplerate // samplerate
        buf_ = numpy.zeros(buffer_length_*3, dtype=numpy.float32)
        N = sample_length // buffer_length_
        tail_length = ((sample_length - 1) % buffer_length_ + 1) * buffer_length // buffer_length_

        yield

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

            buf = scipy.signal.resample(buf_, buffer_length*3) if samplerate != file_samplerate else buf_
            start = buffer_length * ((n + 2) % 3)
            if n == N:
                buf[start+tail_length:start+buffer_length] = 0.0
            yield numpy.copy(buf[start:start+buffer_length])

@DataNode.from_generator
def save(filename, samplerate=44100, width=2):
    with wave.open(filename, "wb") as file:
        scale = 2.0 ** (8*width - 1)
        fmt = "<i{:d}".format(width)

        file.setsampwidth(width)
        file.setnchannels(1)
        file.setframerate(samplerate)
        file.setnframes(0)

        while True:
            file.writeframes(((yield) * scale).astype(fmt).tobytes())

@DataNode.from_generator
def empty(buffer_length=1024, samplerate=44100, duration=None):
    """A data node produces empty data.

    Parameters
    ----------
    buffer_length : int, optional
        The length of data, default is `1024`.
    samplerate : int, optional
        The sample rate, default is `44100`.
    duration : float, optional
        The duration of signal.

    Yields
    ------
    x : ndarray
        The empty signal with length `buffer_length`.
    """
    yield
    if duration is None:
        while True:
            yield numpy.zeros(buffer_length, dtype=numpy.float32)
    else:
        for _ in range(int(duration*samplerate/buffer_length)+1):
            yield numpy.zeros(buffer_length, dtype=numpy.float32)

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
    yield
    for i in range(0, len(signal), buffer_length):
        sliced = signal[i:i+buffer_length]
        yield numpy.pad(sliced, (0, buffer_length - sliced.shape[0]), "constant", constant_values=(0, 0))

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

    time = yield
    while True:
        while waiting is not None and waiting[0][0] < time:
            buffer.append(waiting)
            waiting = next(it, None)

        buffer = [playing for playing in buffer if playing[0][1] >= time]

        time = yield [item for _, item in buffer]

@DataNode.from_generator
def attach(scheduled_signals, buffer_length=1024, samplerate=44100):
    """A data node attaches scheduled signals to input signal.

    Parameters
    ----------
    scheduled_signals : list
        The list of scheduled signals, composed by tuples of scheduled time and data.
    buffer_length : int, optional
        The length of input signal, default is `1024`.
    samplerate : int, optional
        The sample rate to load, default is `44100`.

    Receives
    --------
    x : ndarray
        The input signal.

    Yields
    ------
    x : ndarray
        The processed signal.
    """
    def schedule(item):
        time, data = item
        return (int(time*samplerate) - buffer_length, int(time*samplerate) + len(data))
    dripping_signals = drip(scheduled_signals, schedule)

    with dripping_signals:
        buffer = yield
        for index in itertools.count(0, buffer_length):
            for time, signal in dripping_signals.send(index):
                start = int(time*samplerate)
                i = max(start, index)
                j = min(start+len(signal), index+buffer_length)
                buffer[i-index:j-index] += signal[i-start:j-start]
            buffer = yield buffer


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
    x[-hop_length:] = yield
    while True:
        x_last = yield numpy.copy(x)
        x[:-hop_length] = x[hop_length:]
        x[-hop_length:] = x_last

@DataNode.from_generator
def power_spectrum(win_length, samplerate=44100, windowing=True, weighting=True):
    """A data node maps signal `x` to power spectrum `J`.

    Without windowing and weighting, they should satisfy

        (J * df).sum() == (x**2).mean()

    where the time resolution `dt = 1/samplerate` and the frequency resolution `df = samplerate/win_length`.

    Parameters
    ----------
    win_length : int
        The length of input signal.
    samplerate : int, optional
        The sample rate of input signal, default is `44100`.
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
        weighting = get_A_weight(samplerate, win_length) if weighting else 1
    weighting *= 2/win_length/samplerate

    x = yield
    while True:
        x = yield weighting * numpy.abs(numpy.fft.rfft(x*windowing))**2

@DataNode.from_generator
def onset_strength(df):
    """A data node maps spectrum `J` to onset strength `st`.

    Parameters
    ----------
    df : float
        The frequency resolution of input spectrum.

    Receives
    --------
    J : ndarray
        Input spectrum.

    Yields
    ------
    st : float
        The onset strength between previous and current input spectrum.
    """
    curr = yield
    prev = numpy.zeros_like(curr)
    while True:
        prev, curr = curr, (yield numpy.maximum(0.0, curr - prev).sum(0) * df)

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
def draw_spectrum(length, win_length, samplerate=44100, decay=1.0):
    """A data node to show given spectrum by braille patterns.

    Parameters
    ----------
    length : int
        The length of string.
    win_length : int
        The length of input signal before fourier transform.
    samplerate : int, optional
        The sample rate of input signal, default is `44100`.
    decay : float, optional
        The decay volume per period, default is `1.0`.

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

    df = samplerate/win_length
    n_fft = win_length//2+1
    f = numpy.logspace(numpy.log10(10.0), numpy.log10(min(20000.0, samplerate/2)), length*2)
    sec = numpy.concatenate(([0], (f/df).round().astype(int)))
    slices = list(zip(sec[:-1], (sec+1)[1:]))

    buf = [0.0]*(length*2)
    J = yield
    while True:
        vols = [power2db(J[start:end].sum() * df * n_fft/(end-start)) / 60.0 * 4.0 for start, end in slices]
        # buf = [min(4.0, v) for v, prev in zip(vols, buf)]
        buf = [max(0.0, prev-decay, min(4.0, v)) for v, prev in zip(vols, buf)]
        J = yield "".join(chr(0x2800 + A[int(a)] + B[int(b)]) for a, b in zip(buf[0::2], buf[1::2]))


@contextlib.contextmanager
def record(node, buffer_length=1024, samplerate=44100):
    def input_callback(in_data, frame_count, time_info, status):
        try:
            data = numpy.frombuffer(in_data, dtype=numpy.float32)
            node.send(data)
            return in_data, pyaudio.paContinue

        except StopIteration:
            return b'', pyaudio.paComplete

    session = pyaudio.PyAudio()
    try:
        input_stream = session.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=samplerate,
                                    input=True,
                                    output=False,
                                    frames_per_buffer=buffer_length,
                                    stream_callback=input_callback,
                                    start=False)

        with contextlib.closing(input_stream), node:
            yield input_stream

    finally:
        session.terminate()

@contextlib.contextmanager
def play(node, buffer_length=1024, samplerate=44100):
    def output_callback(in_data, frame_count, time_info, status):
        try:
            data = node.send(None)
            return data.tobytes(), pyaudio.paContinue

        except StopIteration:
            return b'', pyaudio.paComplete

    session = pyaudio.PyAudio()
    try:
        output_stream = session.open(format=pyaudio.paFloat32,
                                     channels=1,
                                     rate=samplerate,
                                     input=False,
                                     output=True,
                                     frames_per_buffer=buffer_length,
                                     stream_callback=output_callback,
                                     start=False)

        with contextlib.closing(output_stream), node:
            yield output_stream

    finally:
        session.terminate()


def filter(x, distr):
    return numpy.fft.irfft(numpy.fft.rfft(x) * distr)

def pulse(samplerate=44100, freq=1000.0, decay_time=0.01, amplitude=1.0, length=None):
    if length is None:
        length = decay_time
    t = numpy.linspace(0, length, int(length*samplerate), endpoint=False, dtype=numpy.float32)
    return amplitude * 2**(-t/decay_time) * numpy.sin(2 * numpy.pi * freq * t)

def power2db(power, scale=(1e-5, 1e6)):
    return 10.0 * numpy.log10(numpy.maximum(scale[0], power*scale[1]))

def get_Hann_window(win_length):
    a = numpy.linspace(0, numpy.pi, win_length)
    window = numpy.sin(a)**2
    gain = (3/8)**0.5 # (window**2).mean()**0.5
    return window / gain

def get_A_weight(samplerate, win_length):
    f = numpy.arange(win_length//2+1) * (samplerate/win_length)

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
    weight[f<10] = 0.0
    weight[f>20000] = 0.0

    return weight

