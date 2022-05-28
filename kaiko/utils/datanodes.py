import time as time_module
import functools
import itertools
from collections import OrderedDict
import dataclasses
import contextlib
import queue
import threading
import subprocess
import signal
import ast
from typing import Callable
import numpy
import scipy
import scipy.signal
import numexpr


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
            self.generator.send(None)
        except StopIteration as e:
            self.finalized = True
            self.result = e.value
            return self
        except:
            self.finalized = True
            raise
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

    def join(self):
        if self.finalized:
            raise DataNodeStateError("try to initialize finalized data node")
        self.initialized = True
        if self.finalized:
            return self.result

        try:
            self.result = yield from self.generator
        except:
            self.finalized = True
            raise
        else:
            return self.result

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

        elif hasattr(node_like, "__iter__"):
            return DataNode.from_iter(node_like)

        elif hasattr(node_like, "__call__"):
            return DataNode.from_func(node_like)

        else:
            raise TypeError

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
        The number of delay with prepending `None`, or data node of prepended
        values.

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
        The number of skips with prefeeding `None`, or data node of prefeeded
        values.

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
                node.send(dummy)

            data = yield
            while True:
                res = node.send(data)
                data = yield res
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
            with node:
                while True:
                    try:
                        data = node.send(data)
                    except StopIteration:
                        break
                    data = yield data


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


# functional nodes
@datanode
def cache(node, key=lambda a: a):
    node = DataNode.wrap(node)
    with node:
        data = yield
        kee = key(data)
        value = node.send(data)

        while True:
            data = yield value
            kee_ = key(data)
            if kee == kee_:
                continue
            kee = kee_
            value = node.send(data)


def map(func, **kw):
    return DataNode.from_func(lambda arg: func(arg, **kw))


def starmap(func, **kw):
    return DataNode.from_func(lambda args: func(*args, **kw))


def cachemap(func, key=lambda a: a, **kw):
    return cache(lambda arg: func(arg, **kw), key=key)


def starcachemap(func, key=lambda *a: a, **kw):
    return cache(lambda args: func(*args, **kw), key=lambda args: key(*args))


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

    where the time resolution `dt = 1/samplerate` and the frequency resolution
    `df = samplerate/win_length`.

    Parameters
    ----------
    win_length : int
        The length of input signal.
    samplerate : int, optional
        The sample rate of input signal, default is `44100`.
    windowing : bool or ndarray, optional
        The window function of signal, `True` for default Hann window, `False`
        for no windowing.
    weighting : bool or ndarray, optional
        The weight function of spectrum, `True` for default A-weighting, `False`
        for no weighting.

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
    weighting *= 2 / win_length / samplerate

    x = yield
    if x.ndim > 1:
        windowing = windowing[:, None] if numpy.ndim(windowing) > 0 else windowing
        weighting = weighting[:, None] if numpy.ndim(weighting) > 0 else weighting

    while True:
        x = yield weighting * numpy.abs(numpy.fft.rfft(x * windowing, axis=0)) ** 2


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
        prev, curr = (
            curr,
            (yield numpy.mean(numpy.maximum(0.0, curr - prev).sum(axis=0)) * df),
        )


@datanode
def pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta):
    r"""A data node of peak detaction.

    The value is picked iff: it is `delta` larger than the average in the given
    range, and is largest one in another given range. Those two ranges are
    determined by four parameters, they mean

    ..code::

                      center
            pre_avg     |    post_avg
          ___________   |   ___________
         /           \  v  /           \
        [x, x, x, x, x, x, x, x, x, x, x]
               \_____/     \_____/      \____ new data
               pre_max     post_max

    In `wait` periods after picked, no peak can be picked. The yielded value is
    a boolean indicating whether `center` is picked, it has a delay
    `max(post_max, post_avg)`.

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
    buffer = numpy.zeros(center + delay + 1, dtype=numpy.float32)
    max_buffer = buffer[center - pre_max : center + post_max + 1]
    avg_buffer = buffer[center - pre_avg : center + post_avg + 1]
    index = -1 - delay
    prev_index = -1 - wait

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

    chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
    index = 0
    jndex = 0

    with node:
        try:
            yield
            data = node.send()

            while True:
                length = min(chunk.shape[0] - index, data.shape[0] - jndex)
                chunk[index : index + length] = data[jndex : jndex + length]
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
    chunk = numpy.zeros(chunk_shape, dtype=numpy.float32)
    index = 0
    jndex = 0

    with node:
        try:
            data = yield

            while True:
                length = min(chunk.shape[0] - index, data.shape[0] - jndex)
                chunk[index : index + length] = data[jndex : jndex + length]
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
    index = 0
    jndex = 0

    with node:
        data = yield

        try:
            signal = node.send()

            while True:
                length = min(data.shape[0] - index, signal.shape[0] - jndex)
                data[index : index + length] += signal[jndex : jndex + length]
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


def rechannel(channels, original=None):
    """A data node to rechannel data.

    Parameters
    ----------
    channels : int
        The number of channels.
    original : int, optional
        The original channel number.

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
        if original is None:
            return DataNode.wrap(
                lambda data: (data if data.ndim == 1 else numpy.mean(data, axis=1))
            )
        elif original == 0:
            return DataNode.wrap(lambda data: data)
        else:
            return DataNode.wrap(lambda data: numpy.mean(data, axis=1))

    else:
        if original is None:
            return DataNode.wrap(
                lambda data: numpy.tile(
                    data[:, None]
                    if data.ndim == 1
                    else numpy.mean(data, axis=1, keepdims=True),
                    (1, channels),
                )
            )
        elif original == 0:
            return DataNode.wrap(lambda data: numpy.tile(data[:, None], (1, channels)))
        else:
            return DataNode.wrap(
                lambda data: numpy.tile(
                    numpy.mean(data, axis=1, keepdims=True), (1, channels)
                )
            )


@datanode
def resample(ratio):
    """A data node to resample data.

    Parameters
    ----------
    ratio : float or tuple
        The resampling factor.

    Receives
    --------
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
        next_index = index + data.shape[0] * up / down
        length = int(next_index) - int(index)
        data_ = scipy.signal.resample(data, length, axis=0) if length > 0 else data
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
    start = max(0, round(start * samplerate)) if start is not None else 0
    end = round(end * samplerate) if end is not None else float("inf")

    with node:
        for data in node:
            index += data.shape[0]

            if index <= start:
                continue

            if index - data.shape[0] <= start:
                yield
                data = data[start - index :]

            if index > end:
                data = data[: end - index]

            yield data

            if index > end:
                break


@datanode
def tunslice(node, samplerate, start=None, end=None):
    """A data node unsliced by given timespan.

    Parameters
    ----------
    node : DataNode
        The data node to unslice.
    samplerate : int
        The sample rate of data.
    start : float, optional
        The start time, default is no slicing.
    end : float, optional
        The end time, default is no slicing.

    Receives
    --------
    data : ndarray
        The unsliced signal.
    """
    node = DataNode.wrap(node)
    index = 0
    start = max(0, round(start * samplerate)) if start is not None else 0
    end = round(end * samplerate) if end is not None else float("inf")

    with node:
        while True:
            data = yield

            index += data.shape[0]

            if index <= start:
                continue

            if index - data.shape[0] <= start:
                data = data[start - index :]

            if index > end:
                data = data[: end - index]

            node.send(data)

            if index > end:
                break


# mixer
@datanode
def tspan(samplerate, start=None, end=None):
    """A data node can only pass within a given timespan.

    Parameters
    ----------
    samplerate : int
        The sample rate of data.
    start : float, optional
        The start time, default is no slicing.
    end : float, optional
        The end time, default is no slicing.

    Receives
    --------
    data : ndarray
        The original signal.

    Yields
    ------
    data : ndarray
        The sliced signal.
    """
    index = 0
    start = max(0, round(start * samplerate)) if start is not None else 0
    end = round(end * samplerate) if end is not None else float("inf")

    data = yield
    while True:
        index += data.shape[0]

        if index <= start:
            data = data[0:0]

        else:
            if index - data.shape[0] <= start:
                data = data[start - index :]
            if index > end:
                data = data[: end - index]

        data = yield data

        if index > end:
            break


def clip(amplitude=1, method="hard"):
    if method == "hard":
        func = lambda x: numpy.clip(x, -amplitude, amplitude)
    elif method == "tanh":
        func = lambda x: numpy.tanh(x / amplitude) * amplitude
    else:
        raise ValueError(f"Invalid method: {method}")

    return DataNode.wrap(func)


@datanode
def fadein(samplerate, duration):
    data = yield
    t = 0.0
    while t < duration:
        dt = data.shape[0] / samplerate
        data *= numpy.clip(
            numpy.linspace(t / duration, (t + dt) / duration, data.shape[0]), 0.0, 1.0
        )[:, None]
        t += dt
        data = yield data

    while True:
        data = yield data


@datanode
def fadeout(samplerate, duration, out_event, before=None):
    data = yield
    t = 0.0
    t0 = before
    while True:
        if (t0 is None or t < t0) and out_event.is_set():
            t0 = t
        dt = data.shape[0] / samplerate

        if t0 is not None and t0 < t + dt:
            data *= numpy.clip(
                numpy.linspace(
                    1.0 - (t - t0) / duration,
                    1.0 - (t + dt - t0) / duration,
                    data.shape[0],
                ),
                0.0,
                1.0,
            )[:, None]

        t += dt
        data = yield data


@datanode
def lfilter(b, a=1):
    zi = scipy.signal.lfilter_zi(b, a)
    data = yield
    while data.shape[0] == 0:
        data = yield data
    zi = zi[(slice(None), *(None,) * (data.ndim - 1))] * data[0:1]
    while True:
        if data.shape[0] > 0:
            data, zi = scipy.signal.lfilter(b, a, data, axis=0, zi=zi)
        data = yield data


def bandpass(N, bands, gains, samplerate):
    if bands and bands[0] is None:
        bands[0] = 0
    if bands and bands[-1] is None:
        bands[-1] = samplerate / 2
    return lfilter(scipy.signal.remez(N, bands, gains, fs=samplerate))


def gammatone(freq, samplerate):
    return lfilter(*scipy.signal.gammatone(freq, "iir", fs=samplerate))


def waveform(expr, samplerate=44100, channels=1, chunk_length=1024, variables=None):
    # &, |, ~
    # <, <=, ==, !=, >=, >
    # +, -, *, /, **, %, <<, >>
    # sin, cos, tan, arcsin, arccos, arctan, arctan2
    # sinh, cosh, tanh, arcsinh, arccosh, arctanh
    # log, log10, log1p, exp, expm1
    # sqrt, abs, conj, real, imag, complex
    # where
    # pi, pi2, inf, e

    if variables is None:
        variables = {}
    constants = {
        **variables,
        "pi": numpy.pi,
        "pi2": numpy.pi * 2,
        "inf": numpy.inf,
        "e": numpy.e,
    }
    expr = expr.format(
        sine=sine_wave_template,
        square=square_wave_template,
        triangle=triangle_wave_template,
        sawtooth=sawtooth_wave_template,
        square_duty=square_duty_wave_template,
    )

    dt = chunk_length / samplerate
    t_ = numpy.linspace(0, dt, chunk_length, dtype=numpy.float64, endpoint=False)
    if channels > 0:
        t_ = t_[:, None] * [[1] * channels]
    _r = numexpr.evaluate(expr, local_dict={"t": t_}, global_dict=constants)
    if numpy.shape(_r) != t_.shape:
        raise TypeError("The returned array shape should be the same as the input.")
    if not numpy.issubdtype(_r.dtype, numpy.float64):
        raise TypeError("The returned data type should be a subtype of float64.")

    @datanode
    def waveform_node():
        t0 = 0.0
        yield
        for _ in itertools.count():
            t = t0 + t_
            yield numexpr.evaluate(expr, local_dict={"t": t}, global_dict=constants)
            t0 += dt

    return waveform_node()


class Template:
    def __init__(self, template):
        self.template = template

    def __format__(self, spec):
        return self.template.format(spec)


sine_wave_template = Template("(sin(({})*pi2))")
square_wave_template = Template("(where(({})%1<0.5,1,-1))")
triangle_wave_template = Template("(arcsin(sin(({})*pi2))/pi*2)")
sawtooth_wave_template = Template("((({})+0.5)%1*2-1)")
square_duty_wave_template = Template("(where(([{0}][0])%1<[{0}][1],1,-1))")


def parse_minsec(s):
    if s == "":
        return None
    min, sec = s.split(":", 1) if ":" in s else ("0", s)
    return int(min) * 60 + float(sec)


@dataclasses.dataclass
class Waveform:
    expr: str

    def generate(self, samplerate, channels, buffer_length=1024):
        base, *effects = self.expr.split("#")
        effect_nodes = []
        for effect in effects:
            if "~" in effect:
                start, end = effect.split("~", 1)
                name = "tspan"
                args = parse_minsec(start), parse_minsec(end)
            else:
                name, args = effect.split(":", 1) if ":" in effect else (effect, "")
                if name not in Waveform.valid_effects:
                    raise ValueError(f"invalid effect name {name}")
                args = ast.literal_eval(f"({args},)") if args else ()
            effect_nodes.append(getattr(Waveform, name)(samplerate, channels, *args))

        return pipe(waveform(base, samplerate, channels, buffer_length), *effect_nodes)

    valid_effects = [
        "tspan",
        "clip",
        "bandpass",
        "gammatone",
    ]

    @staticmethod
    def tspan(samplerate, channels, start=None, end=None):
        return tspan(samplerate, start=start, end=end)

    @staticmethod
    def clip(samplerate, channels, amplitude=1.0, method="hard"):
        return clip(amplitude=amplitude, method=method)

    @staticmethod
    def bandpass(samplerate, channels, bands, gains, N=401):
        return bandpass(N, bands, gains, samplerate)

    @staticmethod
    def gammatone(samplerate, channels, frequency):
        return gammatone(frequency, samplerate)


def collect(node):
    with node:
        return numpy.concatenate(list(node))


# others
class DynamicPipeline(DataNode):
    """Chaining given data nodes dynamically.

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
                        zindex_func = (
                            zindex
                            if hasattr(zindex, "__call__")
                            else lambda z=zindex: z
                        )
                        nodes[key] = (node, zindex_func)

                for key, (node, _) in sorted(
                    nodes.items(), key=lambda item: item[1][1]()
                ):
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
        node = DataNode.wrap(node)
        key = self._NodeKey(self, node)
        self.queue.put((key, node, zindex))
        return key

    def remove_node(self, key):
        self.queue.put((key, None, (0,)))


@datanode
def count(start, step):
    index = 0
    yield
    while True:
        yield start + index * step
        index += 1


@datanode
def time(shift=0.0):
    ref_time = time_module.perf_counter()
    yield
    while True:
        yield time_module.perf_counter() - ref_time + shift


@datanode
def sleep(delta):
    start = time_module.perf_counter()
    yield
    while time_module.perf_counter() - start < delta:
        yield


@datanode
def tick(dt, t0=0.0, shift=0.0, stop_event=None):
    if stop_event is None:
        stop_event = threading.Event()
    ref_time = time_module.perf_counter()

    yield
    for i in itertools.count():
        if stop_event.wait(
            max(0.0, ref_time + t0 + i * dt - time_module.perf_counter())
        ):
            break

        yield time_module.perf_counter() - ref_time + shift


@datanode
def ensure(node, exception):
    node = DataNode.wrap(node)
    data = None
    with node:
        while True:
            try:
                data = yield data
            except GeneratorExit:
                raise exception()

            try:
                data = node.send(data)
            except StopIteration as stop:
                return stop.value


# async processes
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


class ThreadError(Exception):
    pass


def _thread_task(thread, stop_event, res, err):
    yield
    thread.start()

    try:
        yield
        while thread.is_alive():
            yield

    except GeneratorExit:
        stop_event.set()
        if thread.is_alive():
            thread.join()

    else:
        if not err.empty():
            raise ThreadError() from err.get()
        else:
            assert not res.empty()
            return res.get()


@datanode
def create_task(node):
    res = queue.Queue()
    err = queue.Queue()
    stop_event = threading.Event()

    def run():
        try:
            with node:
                while not stop_event.is_set():
                    try:
                        node.send(None)
                    except StopIteration as stop:
                        res.put(stop.value)
                        return
            res.put(None)
        except Exception as e:
            err.put(e)

    thread = threading.Thread(target=run)
    return (yield from _thread_task(thread, stop_event, res, err))


@datanode
def interval(node, dt, t0=0.0):
    node = DataNode.wrap(node)

    res = queue.Queue()
    err = queue.Queue()
    stop_event = threading.Event()

    def run():
        try:
            ref_time = time_module.perf_counter()

            for i in itertools.count():
                delta = ref_time + t0 + i * dt - time_module.perf_counter()
                if stop_event.wait(delta) if delta > 0.0 else stop_event.is_set():
                    res.put(None)
                    return

                try:
                    expired = delta <= 0.0
                    node.send(expired)
                except StopIteration as stop:
                    res.put(stop.value)
                    return

        except Exception as e:
            err.put(e)

    with node:
        thread = threading.Thread(target=run)
        return (yield from _thread_task(thread, stop_event, res, err))


# not data nodes
def filter(x, distr):
    return numpy.fft.irfft(numpy.fft.rfft(x, axis=0) * distr, axis=0)


def pulse(samplerate=44100, freq=1000.0, decay_time=0.01, amplitude=1.0, length=None):
    if length is None:
        length = decay_time * 10
    t = numpy.linspace(
        0, length, int(length * samplerate), endpoint=False, dtype=numpy.float32
    )
    return amplitude * 2 ** (-t / decay_time) * numpy.sin(2 * numpy.pi * freq * t)


def power2db(power, scale=(1e-5, 1e6)):
    return 10.0 * numpy.log10(numpy.maximum(scale[0], power * scale[1]))


def get_Hann_window(win_length):
    a = numpy.linspace(0, numpy.pi, win_length)
    window = numpy.sin(a) ** 2
    gain = (3 / 8) ** 0.5  # (window**2).mean()**0.5
    return window / gain


def get_half_Hann_window(win_length):
    a = numpy.linspace(0, numpy.pi / 2, win_length)
    window = numpy.sin(a) ** 2
    return window


def get_A_weight(samplerate, win_length):
    f = numpy.arange(win_length // 2 + 1) * (samplerate / win_length)

    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    weight = (f ** 4 * f4 ** 2) ** 2
    weight /= (f ** 2 + f1 ** 2) ** 2
    weight /= f ** 2 + f2 ** 2
    weight /= f ** 2 + f3 ** 2
    weight /= (f ** 2 + f4 ** 2) ** 2

    # normalize on 1000 Hz
    f0 = 1000.0
    weight0 = (f0 ** 4 * f4 ** 2) ** 2
    weight0 /= (f0 ** 2 + f1 ** 2) ** 2
    weight0 /= f0 ** 2 + f2 ** 2
    weight0 /= f0 ** 2 + f3 ** 2
    weight0 /= (f0 ** 2 + f4 ** 2) ** 2
    # weight0 == 10**-0.1

    weight /= weight0
    weight[f < 10] = 0.0
    weight[f > 20000] = 0.0

    return weight
