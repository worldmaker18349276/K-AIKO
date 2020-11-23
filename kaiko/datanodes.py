import time
import functools
import itertools
import contextlib
import threading
import numpy
import scipy
import scipy.fftpack
import scipy.signal
import pyaudio
import wave
import audioread


class DataNode:
    def __init__(self, generator):
        self.generator = generator
        self.initialized = False
        self.finalized = False

    def send(self, value=None):
        if not self.initialized:
            raise RuntimeError("try to access un-initialized data node")
        if self.finalized:
            raise StopIteration

        try:
            res = self.generator.send(value)
        except:
            self.finalized = True
            raise
        else:
            return res

    def join(self, value=None):
        try:
            while True:
                value = yield self.send(value)
        except StopIteration:
            return value

    def __next__(self):
        return self.send(None)

    def __iter__(self):
        return self

    def __enter__(self):
        if self.finalized:
            raise RuntimeError("try to initialize finalized data node")
        if self.initialized:
            return self
        self.initialized = True

        try:
            next(self.generator)
            return self

        except StopIteration:
            self.finalized = True

    def __exit__(self, type=None, value=None, traceback=None):
        if not self.initialized:
            raise RuntimeError("try to finalize un-initialized data node")
        if self.finalized:
            return False

        self.generator.close()
        self.finalized = True
        return False

    @staticmethod
    def wrap(node_like):
        if isinstance(node_like, DataNode):
            return node_like

        elif hasattr(node_like, '__iter__'):
            def iter(it):
                yield
                for data in it:
                    yield data
            return DataNode(iter(node_like))

        else:
            def pure(func):
                data = yield
                while True:
                    data = yield func(data)
            return DataNode(pure(node_like))

def datanode(gen):
    @functools.wraps(gen)
    def node_builder(*args, **kwargs):
        return DataNode(gen(*args, **kwargs))
    return node_builder


# basic data nodes
@datanode
def rewrap(node, context=contextlib.suppress()):
    """A data node processing data by another data node.

    Parameters
    ----------
    node : DataNode
        The data node.
    context : contextmanager, optional
        The context manager wraps the node.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The processed signal.
    """
    with context, node:
        yield from node.join((yield))

@datanode
def delay(prepend):
    """A data node delays signals and prepends given values.

    Parameters
    ----------
    prepend : int or DataNode
        The number of delay with prepending `None`, or data node of prepended values.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The delayed signal.
    """
    if isinstance(prepend, int):
        prepend = itertools.repeat(None, prepend)
    prepend = DataNode.wrap(prepend)

    with prepend:
        buffer = list(prepend)

    data = yield
    while True:
        buffer.append(data)
        data = yield buffer.pop(0)

@datanode
def skip(node, prefeed):
    """A data node skips signals by feeding given values when initializing.

    Parameters
    ----------
    prefeed : int or DataNode
        The number of skips with prefeeding `None`, or data node of prefeeded values.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The advance signal.
    """
    node = DataNode.wrap(node)
    if isinstance(prefeed, int):
        prefeed = itertools.repeat(None, prefeed)
    prefeed = DataNode.wrap(prefeed)

    with prefeed:
        buffer = list(prefeed)

    with node:
        for dummy in buffer:
            node.send(dummpy)

        yield from node.join((yield))

@datanode
def take(predicate):
    """A data node takes finite signals.

    Parameters
    ----------
    predicate : int or DataNode
        The number of period to take, or data node of predicate.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The output signal.
    """
    if isinstance(predicate, int):
        predicate = itertools.repeat(True, predicate)
    predicate = DataNode.wrap(predicate)

    with predicate:
        data = yield
        while predicate.send(data):
            data = yield data

@datanode
def pipe(*nodes):
    """A data node processes data sequentially.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to pipe.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The processed signal.
    """
    nodes = [DataNode.wrap(node) for node in nodes]
    with contextlib.ExitStack() as stack:
        for node in nodes:
            stack.enter_context(node)

        data = yield
        while True:
            res = data
            for node in nodes:
                res = node.send(res)
            data = yield res

@datanode
def pair(*nodes):
    """A data node processes data parallelly.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to pair.

    Receives
    --------
    data : tuple
        The input signal; its length should equal to number of nodes.

    Yields
    ------
    data : tuple
        The processed signal; its length should equal to number of nodes.
    """
    nodes = [DataNode.wrap(node) for node in nodes]
    with contextlib.ExitStack() as stack:
        for node in nodes:
            stack.enter_context(node)

        data = yield
        while True:
            data = yield tuple(node.send(subdata) for node, subdata in zip(nodes, data))

@datanode
def chain(*nodes):
    """A data node processes data with chaining nodes.

    Parameters
    ----------
    nodes : list of DataNode
        The data nodes to chain.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The processed signal.
    """
    nodes = [DataNode.wrap(node) for node in nodes]
    with contextlib.ExitStack() as stack:
        for node in nodes:
            stack.enter_context(node)

        data = yield
        for node in nodes:
            data = yield from node.join(data)

@datanode
def branch(*nodes):
    """A data node processes data additionally.

    Parameters
    ----------
    nodes : list of DataNode
        The sequence of data nodes to branch.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : any
        The input signal.
    """
    node = pipe(*nodes)

    with node:
        data = yield

        while True:
            try:
                node.send(data)
            except StopIteration:
                break
            data = yield data

@datanode
def merge(*nodes):
    """A data node processes additional data.

    Parameters
    ----------
    nodes : list of DataNode
        The sequence of data nodes to merge.

    Receives
    --------
    data : any
        The input signal.

    Yields
    ------
    data : tuple
        The input signal and additional data.
    """
    node = pipe(*nodes)
    with node:
        data = yield
        while True:
            data = yield (data, node.send())

@datanode
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

    buffer[-1] = yield
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


# for fixed-width data
@datanode
def frame(win_length, hop_length):
    """A data node to frame signal, prepend by zero.

    Parameters
    ----------
    win_length : int
        The length of framed data.
    hop_length : int
        The length of input data.

    Receives
    --------
    data : ndarray
        The input signal.

    Yields
    ------
    data : ndarray
        The framed signal.
    """
    if win_length < hop_length:
        data = yield
        while True:
            data = yield numpy.copy(data[-win_length:])
        return

    data_last = yield
    data = numpy.zeros((win_length, *data_last.shape[1:]), dtype=numpy.float32)
    data[-hop_length:] = data_last

    while True:
        data_last = yield numpy.copy(data)
        data[:-hop_length] = data[hop_length:]
        data[-hop_length:] = data_last

@datanode
def power_spectrum(win_length, samplerate=44100, windowing=True, weighting=True):
    """A data node maps signal `x` to power spectrum `J`.

    Without windowing and weighting, they should satisfy

        (J * df).sum(axis=0) == (x**2).mean(axis=0)

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
    if x.ndim > 1:
        windowing = windowing[:, None] if numpy.ndim(windowing) > 0 else windowing
        weighting = weighting[:, None] if numpy.ndim(weighting) > 0 else weighting

    while True:
        x = yield weighting * numpy.abs(numpy.fft.rfft(x*windowing, axis=0))**2

@datanode
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
        prev, curr = curr, (yield numpy.mean(numpy.maximum(0.0, curr - prev).sum(axis=0)) * df)

@datanode
def draw_spectrum(length, win_length, samplerate=44100, decay=1/4):
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
        The decay volume per period, default is `1/4`.

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
    n = numpy.linspace(1, 88, length*2+1)
    f = 440 * 2**((n-49)/12)
    sec = numpy.minimum(n_fft-1, (f/df).round().astype(int))
    slices = [slice(start, stop) for start, stop in zip(sec[:-1], (sec+1)[1:])]

    volume_of = lambda J: power2db(J.mean() * samplerate / 2, scale=(1e-5, 1e6)) / 60.0
    vols = [0.0]*(length*2)
    J = yield
    while True:
        vols = [max(0.0, prev-decay, min(1.0, volume_of(J[slic]))) for slic, prev in zip(slices, vols)]
        J = yield "".join(chr(0x2800 + A[int(a*4)] + B[int(b*4)]) for a, b in zip(vols[0::2], vols[1::2]))


# for variable-width data
@datanode
def chunk(node, chunk_shape=1024):
    """Make a data node be able to produce fixed width data.

    Parameters
    ----------
    node : DataNode
        The data node to chunk.
    chunk_shape : int or tuple, optional
        The shape of chunk, default is `1024`.

    Yields
    ------
    data : ndarray
        The chunked signal with shape `chunk_shape`.
    """
    node = DataNode.wrap(node)

    with node:
        try:
            yield
            chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
            index = 0

            data = node.send()
            jndex = 0

            while True:
                length = min(chunk.shape[0]-index, data.shape[0]-jndex)
                chunk[index:index+length] = data[jndex:jndex+length]
                index += length
                jndex += length

                if index == chunk.shape[0]:
                    yield chunk
                    chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
                    index = 0

                if jndex == data.shape[0]:
                    data = node.send()
                    jndex = 0

        except StopIteration:
            if index > 0:
                yield chunk

@datanode
def unchunk(node, chunk_shape=1024):
    """Make a data node be able to receive data with any length.

    Parameters
    ----------
    node : DataNode
        The data node to unchunk.
    chunk_shape : int or tuple, optional
        The received shape of given data node, default is `1024`.

    Receives
    ------
    data : ndarray
        The unchunked signal with any length.
    """
    node = DataNode.wrap(node)

    with node:
        try:
            chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
            index = 0

            data = yield
            jndex = 0

            while True:
                length = min(chunk.shape[0]-index, data.shape[0]-jndex)
                chunk[index:index+length] = data[jndex:jndex+length]
                index += length
                jndex += length

                if index == chunk.shape[0]:
                    node.send(chunk)
                    chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
                    index = 0

                if jndex == data.shape[0]:
                    data = yield
                    jndex = 0

        except GeneratorExit:
            if index > 0:
                node.send(chunk)

@datanode
def attach(node):
    with node:
        try:
            data = yield
            index = 0

            signal = node.send()
            jndex = 0

            while True:
                length = min(data.shape[0]-index, signal.shape[0]-jndex)
                data[index:index+length] += signal[jndex:jndex+length]
                index += length
                jndex += length

                if index == data.shape[0]:
                    data = yield data
                    index = 0

                if jndex == signal.shape[0]:
                    signal = node.send()
                    jndex = 0

        except StopIteration:
            if index > 0:
                yield data


def rechannel(channels):
    """A data node to rechannel data.

    Parameters
    ----------
    channels : int or list
        The channel mapping.

    Receives
    ------
    data : ndarray
        The original signal.

    Yields
    ------
    data : ndarray
        The rechanneled signal.
    """
    if channels == 0:
        return lambda data: (data if data.ndim == 1 else numpy.mean(data, axis=1))
    elif isinstance(channels, int):
        return lambda data: (data if data.ndim == 1 else numpy.mean(data, axis=1))[:, None][:, [0]*channels]
    else:
        return lambda data: (data[:, None] if data.ndim == 1 else data)[:, channels]

@datanode
def resample(ratio):
    """A data node to resample data.

    Parameters
    ----------
    ratio : float or tuple
        The resampling factor.

    Receives
    ------
    data : ndarray
        The original signal.

    Yields
    ------
    data : ndarray
        The resampled signal.
    """
    index = 0.0
    up, down = (ratio, 1) if isinstance(ratio, float) else ratio

    data = yield
    while True:
        next_index = index + data.shape[0] * up/down
        length = int(next_index) - int(index)
        data_ = scipy.signal.resample(data, length, axis=0)
        index = next_index % 1.0
        data = yield data_

@datanode
def tslice(node, samplerate=44100, start=None, end=None):
    """A data node sliced by given timespan.

    Parameters
    ----------
    node : DataNode
        The data node to slice.
    samplerate : int, optional
        The sample rate of data, default is `44100`.
    start : float, optional
        The start time, default is no slicing.
    end : float, optional
        The end time, default is no slicing.

    Yields
    ------
    data : ndarray
        The sliced signal.
    """
    node = DataNode.wrap(node)
    index = 0
    start = max(0, round(start*samplerate)) if start is not None else 0
    end = round(end*samplerate) if end is not None else end

    with node:
        for data in node:
            index += data.shape[0]

            if index <= start:
                continue

            if index - data.shape[0] <= start:
                yield
                data = data[start-index:]

            if end is not None and index > end:
                data = data[:end-index]

            yield data

            if end is not None and index > end:
                break


# IO data nodes
@datanode
def load(filename):
    """A data node to load sound file.

    Parameters
    ----------
    filename : str
        The sound file to load.

    Yields
    ------
    data : ndarray
        The loaded signal.
    """

    if filename.endswith(".wav"):
        with wave.open(filename, 'rb') as file:
            nchannels = file.getnchannels()
            width = file.getsampwidth()
            scale = 2.0 ** (1 - 8*width)
            fmt = f'<i{width}'
            def frombuffer(data):
                return scale * numpy.frombuffer(data, fmt).astype(numpy.float32).reshape(-1, nchannels)

            remaining = file.getnframes()
            while remaining > 0:
                data = file.readframes(256)
                remaining -= len(data)//width
                yield frombuffer(data)

    else:
        with audioread.audio_open(filename) as file:
            width = 2
            scale = 2.0 ** (1 - 8*width)
            fmt = f'<i{width}'
            def frombuffer(data):
                return scale * numpy.frombuffer(data, fmt).astype(numpy.float32).reshape(-1, file.channels)

            for data in file:
                yield frombuffer(data)

@datanode
def save(filename, samplerate=44100, channels=1, width=2):
    """A data node to save as .wav file.

    Parameters
    ----------
    filename : str
        The sound file to save.
    samplerate : int, optional
        The sample rate, default is `44100`.
    channels : int, optional
        The number of channels, default is `1`.
    width : int, optional
        The sample width in bytes.

    Receives
    ------
    data : ndarray
        The signal to save.
    """
    with wave.open(filename, 'wb') as file:
        scale = 2.0 ** (8*width - 1)
        fmt = f'<i{width}'
        def tobuffer(data):
            return (data * scale).astype(fmt).tobytes()

        file.setsampwidth(width)
        file.setnchannels(channels)
        file.setframerate(samplerate)
        file.setnframes(0)

        while True:
            file.writeframes(tobuffer((yield)))

@contextlib.contextmanager
def record(manager, node, samplerate=44100, buffer_shape=1024, format='f4', device=-1):
    """A context manager of input stream processing by given node.

    Parameters
    ----------
    manager : pyaudio.PyAudio
        The PyAudio object.
    node : DataNode
        The data node to process recorded sound.
    samplerate : int, optional
        The sample rate of input signal, default is `44100`.
    buffer_shape : int or tuple, optional
        The shape of input signal, default is `1024`.
    format : str, optional
        The sample format of input signal, default is `'f4'`.
    device : int, optional
        The input device index, and `-1` for default input device.

    Yields
    ------
    input_stream : pyaudio.Stream
        The stopped input stream to record sound.
    """
    node = DataNode.wrap(node)
    pa_format = {'f4': pyaudio.paFloat32,
                 'i4': pyaudio.paInt32,
                 'i2': pyaudio.paInt16,
                 'i1': pyaudio.paInt8,
                 'u1': pyaudio.paUInt8,
                 }[format]

    scale = 2.0 ** (8*int(format[1]) - 1)
    normalize = {'f4': (lambda d: d),
                 'i4': (lambda d: d / scale),
                 'i2': (lambda d: d / scale),
                 'i1': (lambda d: d / scale),
                 'u1': (lambda d: (d - 64) / 64),
                 }[format]

    if device == -1:
        device = None

    length, channels = (buffer_shape, 1) if isinstance(buffer_shape, int) else buffer_shape

    def input_callback(in_data, frame_count, time_info, status):
        try:
            data = normalize(numpy.frombuffer(in_data, dtype=format).reshape(buffer_shape))
            node.send(data)

            return b"", pyaudio.paContinue
        except StopIteration:
            return b"", pyaudio.paComplete

    input_stream = manager.open(format=pa_format,
                                channels=channels,
                                rate=samplerate,
                                input=True,
                                output=False,
                                input_device_index=device,
                                frames_per_buffer=length,
                                stream_callback=input_callback,
                                start=False)

    with node:
        try:
            yield input_stream
        finally:
            input_stream.stop_stream()
            input_stream.close()

@contextlib.contextmanager
def play(manager, node, samplerate=44100, buffer_shape=1024, format='f4', device=-1):
    """A context manager of output stream processing by given node.

    Parameters
    ----------
    manager : pyaudio.PyAudio
        The PyAudio object.
    node : DataNode
        The data node to process playing sound.
    samplerate : int, optional
        The sample rate of output signal, default is `44100`.
    buffer_shape : int or tuple, optional
        The length of output signal, default is `1024`.
    format : str, optional
        The sample format of output signal, default is `'f4'`.
    device : int, optional
        The output device index, and `-1` for default output device.

    Yields
    ------
    output_stream : pyaudio.Stream
        The stopped output stream to play sound.
    """
    node = DataNode.wrap(node)
    pa_format = {'f4': pyaudio.paFloat32,
                 'i4': pyaudio.paInt32,
                 'i2': pyaudio.paInt16,
                 'i1': pyaudio.paInt8,
                 'u1': pyaudio.paUInt8,
                 }[format]

    scale = 2.0 ** (8*int(format[1]) - 1)
    normalize = {'f4': (lambda d: d),
                 'i4': (lambda d: d * scale),
                 'i2': (lambda d: d * scale),
                 'i1': (lambda d: d * scale),
                 'u1': (lambda d: d * 64 + 64),
                 }[format]

    if device == -1:
        device = None

    length, channels = (buffer_shape, 1) if isinstance(buffer_shape, int) else buffer_shape

    def output_callback(in_data, frame_count, time_info, status):
        try:
            data = node.send(None)
            out_data = normalize(data).astype(format).tobytes()

            return out_data, pyaudio.paContinue
        except StopIteration:
            return b"", pyaudio.paComplete

    output_stream = manager.open(format=pa_format,
                                 channels=channels,
                                 rate=samplerate,
                                 input=False,
                                 output=True,
                                 output_device_index=device,
                                 frames_per_buffer=length,
                                 stream_callback=output_callback,
                                 start=False)

    with node:
        try:
            yield output_stream
        finally:
            output_stream.stop_stream()
            output_stream.close()

@datanode
def interval(dt, prepare=lambda _:None, first=0.0):
    prepare = DataNode.wrap(prepare)
    stop = threading.Event()

    with prepare:
        yield
        t0 = time.time()

        for i, data in enumerate(prepare):
            if stop.wait(max(0.0, t0+first+i*dt - time.time())):
                break

            yield (time.time()-t0, data)

@contextlib.contextmanager
def thread(node):
    stop = threading.Event()

    def run():
        with node:
            for _ in node:
                if stop.wait(0.0):
                    break

    node.__enter__()
    thread = threading.Thread(target=run)
    try:
        yield thread
    finally:
        stop.set()
        if thread.is_alive():
            thread.join()

@contextlib.contextmanager
def timeit(name="timeit", timeit=True):
    if not timeit:
        yield lambda node: node
        return

    import bisect

    N = 10
    count = 0
    total = 0.0
    total2 = 0.0
    worst = [0.0]*N
    best = [numpy.inf]*N

    @datanode
    def timeit_wrapper(node):
        nonlocal count, total, total2, worst, best

        with node:
            data = None
            while True:
                data = yield data
                t0 = time.perf_counter()
                data = node.send(data)
                t = time.perf_counter() - t0
                count += 1
                total += t
                total2 += t**2
                bisect.insort(worst, t)
                worst.pop(0)
                bisect.insort_left(best, t)
                best.pop()

    try:
        yield timeit_wrapper

    finally:
        if count == 0:
            print(f"{name}: count=0")
            return

        avg = total/count
        std = (total2/count - avg**2)**0.5

        if count < N:
            print(f"{name}: count={count}, avg={avg}±{std}")
        else:
            print(f"{name}: count={count}, avg={avg}±{std} ({sum(best)/N} ~ {sum(worst)/N})")


# not data nodes
def filter(x, distr):
    return numpy.fft.irfft(numpy.fft.rfft(x, axis=0) * distr, axis=0)

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

def get_half_Hann_window(win_length):
    a = numpy.linspace(0, numpy.pi/2, win_length)
    window = numpy.sin(a)**2
    return window

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

