import sys
import os
import time
import functools
import itertools
from collections import OrderedDict
import contextlib
import queue
import threading
import subprocess
import signal
import bisect
import numpy
import scipy
import scipy.signal


def datanode(gen_func):
    @functools.wraps(gen_func)
    def node_func(*args, **kwargs):
        return DataNode(gen_func(*args, **kwargs))
    return node_func

class DataNodeStateError(Exception):
    pass

class DataNode:
    def __init__(self, generator):
        self.generator = generator
        self.initialized = False
        self.finalized = False
        self.result = None

    def send(self, value=None):
        if not self.initialized:
            raise DataNodeStateError("try to access un-initialized data node")
        if self.finalized:
            raise StopIteration(self.result)

        try:
            res = self.generator.send(value)
        except StopIteration as e:
            self.finalized = True
            self.result = e.value
            raise e
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
            raise DataNodeStateError("try to initialize finalized data node")
        if self.initialized:
            return self
        self.initialized = True

        try:
            next(self.generator)

        except StopIteration as e:
            self.finalized = True
            self.result = e.value
            return self

        else:
            return self

    def __exit__(self, type=None, value=None, traceback=None):
        if not self.initialized:
            raise DataNodeStateError("try to finalize un-initialized data node")
        if self.finalized:
            return False

        self.close()
        return False

    def close(self):
        self.generator.close()
        self.finalized = True

    @staticmethod
    @datanode
    def from_iter(iterator):
        yield
        for data in iterator:
            yield data

    @staticmethod
    @datanode
    def from_func(function):
        data = yield
        while True:
            data = yield function(data)

    @staticmethod
    def wrap(node_like):
        if isinstance(node_like, DataNode):
            return node_like

        elif hasattr(node_like, '__iter__'):
            return DataNode.from_iter(node_like)

        elif hasattr(node_like, '__call__'):
            return DataNode.from_func(node_like)

        else:
            raise ValueError

    def exhaust(self, dt=0.0, interruptible=False):
        stop_event = threading.Event()
        def SIGINT_handler(sig, frame):
            stop_event.set()

        with self:
            if interruptible:
                signal.signal(signal.SIGINT, SIGINT_handler)

            while True:
                if stop_event.wait(dt):
                    raise KeyboardInterrupt

                try:
                    self.send(None)
                except StopIteration as e:
                    return e.value


# basic data nodes
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
        try:
            for dummy in buffer:
                node.send(dummpy)

            yield from node.join((yield))
        except StopIteration:
            return

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
        try:
            while predicate.send(data):
                data = yield data
        except StopIteration:
            return

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
                try:
                    res = node.send(res)
                except StopIteration:
                    return
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
            data_ = []
            for node, subdata in zip(nodes, data):
                try:
                    data_.append(node.send(subdata))
                except StopIteration:
                    return
            data = yield tuple(data_)

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
            try:
                data = yield from node.join(data)
            except StopIteration:
                return

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
            try:
                data = yield (data, node.send())
            except StopIteration:
                return


# signal analysis
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
    index = -1-delay
    prev_index = -1-wait

    #               center
    #     pre_avg     |    post_avg
    #   ___________   |   ___________
    #  /           \  v  /           \
    # [x, x, x, x, x, x, x, x, x, x, x]
    #        \_____/     \_____/      \____ new data
    #        pre_max     post_max

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

        except StopIteration:
            return

        except GeneratorExit:
            if index > 0:
                try:
                    node.send(chunk)
                except StopIteration:
                    return

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
def tslice(node, samplerate, start=None, end=None):
    """A data node sliced by given timespan.

    Parameters
    ----------
    node : DataNode
        The data node to slice.
    samplerate : int
        The sample rate of data.
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


# others
class TimedVariable:
    def __init__(self, value=None, duration=numpy.inf):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._scheduled = []
        self._default_value = value
        self._default_duration = duration
        self._item = (value, None, numpy.inf)

    def get(self, time, ret_sched=False):
        with self._lock:
            value, start, duration = self._item
            if start is None:
                start = time

            while not self._queue.empty():
                item = self._queue.get()
                if item[1] is None:
                    item = (item[0], time, item[2])
                self._scheduled.append(item)
            self._scheduled.sort(key=lambda item: item[1])

            while self._scheduled and self._scheduled[0][1] <= time:
                value, start, duration = self._scheduled.pop(0)

            if start + duration <= time:
                value, start, duration = self._default_value, None, numpy.inf

            self._item = (value, start, duration)
            return value if not ret_sched else self._item

    def set(self, value, start=None, duration=None):
        if duration is None:
            duration = self._default_duration
        self._queue.put((value, start, duration))

    def reset(self, start=None):
        self._queue.put((self._default_value, start, numpy.inf))

class Scheduler(DataNode):
    """A data node schedule given data nodes dynamically.

    Receives
    --------
    data : tuple
        The input signal and the meta signal.

    Yields
    ------
    data : any
        The output signal.
    """

    def __init__(self):
        self.queue = queue.Queue()
        super().__init__(self.proxy())

    def proxy(self):
        nodes = OrderedDict()

        try:
            data, *meta = yield

            while True:
                while not self.queue.empty():
                    key, node, zindex = self.queue.get()
                    if key in nodes:
                        nodes[key][0].__exit__()
                        del nodes[key]
                    if node is not None:
                        node.__enter__()
                        zindex_func = zindex if hasattr(zindex, '__call__') else lambda z=zindex: z
                        nodes[key] = (node, zindex_func)

                for key, (node, _) in sorted(nodes.items(), key=lambda item: item[1][1]()):
                    try:
                        data = node.send((data, *meta))
                    except StopIteration:
                        del nodes[key]

                data, *meta = yield data

        finally:
            for node, _ in nodes.values():
                node.__exit__()

    class _NodeKey:
        def __init__(self, parent, node):
            self.parent = parent
            self.node = node

        def is_initialized(self):
            return self.node.initialized

        def is_finalized(self):
            return self.node.finalized

        def remove(self):
            self.parent.remove_node(self)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.remove()

    def add_node(self, node, zindex=(0,)):
        key = self._NodeKey(self, node)
        self.queue.put((key, node, zindex))
        return key

    def remove_node(self, key):
        self.queue.put((key, None, (0,)))

@datanode
def tick(dt, t0=0.0, shift=0.0, stop_event=None):
    if stop_event is None:
        stop_event = threading.Event()
    ref_time = time.perf_counter()

    yield
    for i in itertools.count():
        if stop_event.wait(max(0.0, ref_time+t0+i*dt - time.perf_counter())):
            break

        yield time.perf_counter()-ref_time+shift

@datanode
def timeit(node, log=print):
    if hasattr(time, 'thread_time'):
        get_time = time.thread_time
    elif hasattr(time, 'clock_gettime'):
        get_time = lambda: time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
    else:
        get_time = time.perf_counter

    N = 10
    start = 0.0
    stop = numpy.inf
    count = 0
    total = 0.0
    total2 = 0.0
    worst = [0.0]*N
    best = [numpy.inf]*N

    with node:
        try:
            data = yield

            start = stop = time.perf_counter()

            while True:

                t0 = get_time()
                data = node.send(data)
                t = get_time() - t0
                stop = time.perf_counter()

                count += 1
                total += t
                total2 += t**2
                bisect.insort(worst, t)
                worst.pop(0)
                bisect.insort_left(best, t)
                best.pop()

                data = yield data

        except StopIteration:
            return

        finally:
            stop = time.perf_counter()

            if count == 0:
                log(f"count=0")

            else:
                avg = total/count
                dev = (total2/count - avg**2)**0.5
                eff = total/(stop - start)

                if count < N:
                    log(f"count={count}, avg={avg*1000:5.3f}±{dev*1000:5.3f}ms ({eff: >6.1%})")

                else:
                    best_time = sum(best)/N
                    worst_time = sum(worst)/N

                    log(f"count={count}, avg={avg*1000:5.3f}±{dev*1000:5.3f}ms"
                        f" ({best_time*1000:5.3f}ms ~ {worst_time*1000:5.3f}ms) ({eff: >6.1%})")


# async processes
def _thread_task(thread, stop_event, error):
    yield
    thread.start()
    try:
        yield
        while thread.is_alive():
            yield
    finally:
        stop_event.set()
        if thread.is_alive():
            thread.join()
        if not error.empty():
            raise error.get()

@datanode
def subprocess_task(command):
    yield
    proc = subprocess.Popen(command)
    try:
        yield
        while proc.poll() is None:
            yield
    finally:
        proc.kill()
    return proc.returncode

@datanode
def create_task(func):
    res = queue.Queue()
    error = queue.Queue()
    stop_event = threading.Event()

    def run():
        try:
            res.put(func(stop_event))
        except Exception as e:
            error.put(e)

    thread = threading.Thread(target=run)
    yield from _thread_task(thread, stop_event, error)
    if res.empty():
        raise ValueError("empty result")
    return res.get()

@datanode
def interval(producer=lambda _:None, consumer=lambda _:None, dt=0.0, t0=0.0):
    producer = DataNode.wrap(producer)
    consumer = DataNode.wrap(consumer)

    def run(stop_event):
        ref_time = time.perf_counter()

        for i, data in enumerate(producer):
            delta = ref_time+t0+i*dt - time.perf_counter()
            if stop_event.wait(delta) if delta > 0 else stop_event.is_set():
                break

            try:
                consumer.send(data)
            except StopIteration:
                return

    with producer, consumer:
        with create_task(run) as task:
            yield from task.join((yield))


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

