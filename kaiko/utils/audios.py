import queue
import threading
import contextlib
import pyaudio
import numpy
import scipy.signal
import wave
import audioread
from . import datanodes as dn


def print_pyaudio_info(manager):
    print("portaudio version:")
    print("  " + pyaudio.get_portaudio_version_text())
    print()

    print("available devices:")
    apis_list = [manager.get_host_api_info_by_index(i)['name'] for i in range(manager.get_host_api_count())]

    table = []
    for index in range(manager.get_device_count()):
        info = manager.get_device_info_by_index(index)

        ind = str(index)
        name = info['name']
        api = apis_list[info['hostApi']]
        freq = str(info['defaultSampleRate']/1000)
        chin = str(info['maxInputChannels'])
        chout = str(info['maxOutputChannels'])

        table.append((ind, name, api, freq, chin, chout))

    ind_len   = max(len(entry[0]) for entry in table)
    name_len  = max(len(entry[1]) for entry in table)
    api_len   = max(len(entry[2]) for entry in table)
    freq_len  = max(len(entry[3]) for entry in table)
    chin_len  = max(len(entry[4]) for entry in table)
    chout_len = max(len(entry[5]) for entry in table)

    for ind, name, api, freq, chin, chout in table:
        print(f"  {ind:>{ind_len}}. {name:{name_len}}  by  {api:{api_len}}"
              f"  ({freq:>{freq_len}} kHz, in: {chin:>{chin_len}}, out: {chout:>{chout_len}})")

    print()

    default_input_device_index = manager.get_default_input_device_info()['index']
    default_output_device_index = manager.get_default_output_device_info()['index']
    print(f"default input device: {default_input_device_index}")
    print(f"default output device: {default_output_device_index}")

@contextlib.contextmanager
def create_manager():
    manager = pyaudio.PyAudio()
    try:
        yield manager
    finally:
        manager.terminate()

def validate_input_device(manager, device, samplerate, channels, format):
    if device == -1:
        device = manager.get_default_input_device_info()['index']

    format = {
        'f4': pyaudio.paFloat32,
        'i4': pyaudio.paInt32,
        'i2': pyaudio.paInt16,
        'i1': pyaudio.paInt8,
        'u1': pyaudio.paUInt8,
    }[format]

    manager.is_format_supported(samplerate,
        input_device=device, input_channels=channels, input_format=format)

def validate_output_device(manager, device, samplerate, channels, format):
    if device == -1:
        device = manager.get_default_output_device_info()['index']

    format = {
        'f4': pyaudio.paFloat32,
        'i4': pyaudio.paInt32,
        'i2': pyaudio.paInt16,
        'i1': pyaudio.paInt8,
        'u1': pyaudio.paUInt8,
    }[format]

    manager.is_format_supported(samplerate,
        output_device=device, output_channels=channels, output_format=format)


def _stream_task(stream, error):
    yield
    stream.start_stream()
    try:
        yield
        while stream.is_active():
            yield
    finally:
        stream.stop_stream()
        stream.close()
        if not error.empty():
            raise error.get()

@dn.datanode
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
    node = dn.DataNode.wrap(node)
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

    error = queue.Queue()
    length, channels = (buffer_shape, 1) if isinstance(buffer_shape, int) else buffer_shape

    def input_callback(in_data, frame_count, time_info, status):
        try:
            data = normalize(numpy.frombuffer(in_data, dtype=format).reshape(buffer_shape))
            node.send(data)

            return b"", pyaudio.paContinue
        except StopIteration:
            return b"", pyaudio.paComplete
        except Exception as e:
            error.put(e)
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
        yield from _stream_task(input_stream, error)

@dn.datanode
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
    node = dn.DataNode.wrap(node)
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

    error = queue.Queue()
    length, channels = (buffer_shape, 1) if isinstance(buffer_shape, int) else buffer_shape

    def output_callback(in_data, frame_count, time_info, status):
        try:
            data = node.send(None)
            out_data = normalize(data).astype(format).tobytes()

            return out_data, pyaudio.paContinue
        except StopIteration:
            return b"", pyaudio.paComplete
        except Exception as e:
            error.put(e)
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
        yield from _stream_task(output_stream, error)


class IOCancelledError(Exception):
    pass

@dn.datanode
def load(filename, stop_event=None):
    """A data node to load sound file.

    Parameters
    ----------
    filename : str
        The sound file to load.
    stop_event : threading.Event
        The event to cancel loading file.

    Yields
    ------
    data : ndarray
        The loaded signal.
    """
    if stop_event is None:
        stop_event = threading.Event()

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
                if stop_event.is_set():
                    raise IOCancelledError(f"The operation of loading file {filename} has been cancelled.")

    else:
        with audioread.audio_open(filename) as file:
            width = 2
            scale = 2.0 ** (1 - 8*width)
            fmt = f'<i{width}'
            def frombuffer(data):
                return scale * numpy.frombuffer(data, fmt).astype(numpy.float32).reshape(-1, file.channels)

            for data in file:
                yield frombuffer(data)
                if stop_event.is_set():
                    raise IOCancelledError(f"The operation of loading file {filename} has been cancelled.")

@dn.datanode
def save(filename, samplerate=44100, channels=1, width=2, stop_event=None):
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
    stop_event : threading.Event
        The event to cancel saving file.

    Receives
    ------
    data : ndarray
        The signal to save.
    """
    if stop_event is None:
        stop_event = threading.Event()

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
            if stop_event.is_set():
                raise IOCancelledError(f"The operation of saving file {filename} has been cancelled.")

def load_sound(filepath, samplerate=None, channels=None, volume=0.0, start=None, end=None, chunk_length=1024, stop_event=None):
    with audioread.audio_open(filepath) as file:
        file_samplerate = file.samplerate

    filenode = load(filepath, stop_event)

    if start is not None or end is not None:
        filenode = dn.tslice(filenode, file_samplerate, start, end)

    with filenode:
        sound = numpy.concatenate(tuple(filenode), axis=0)

    if volume != 0:
        sound = sound * 10**(volume/20)

    # resample
    if samplerate is not None and file_samplerate != samplerate:
        length = int(sound.shape[0] * samplerate/file_samplerate)
        sound = scipy.signal.resample(sound, length, axis=0)

    # rechannel
    if sound.ndim == 1:
        sound = sound[:,None]

    if isinstance(channels, int):
        if channels == 0:
            sound = numpy.mean(sound, axis=1)

        elif channels != sound.shape[1]:
            sound = numpy.mean(sound, axis=1, keepdims=True)
            sound = sound[:, [0]*channels]

    elif isinstance(channels, list):
        sound = sound[:, channels]

    elif channels is None:
        pass

    else:
        raise ValueError(f"invalid channel map: {repr(channels)}")

    # chunk
    if chunk_length is not None:
        shape = (chunk_length, *sound.shape[1:])
        with dn.chunk([sound], shape) as node:
            sound = list(node)
    else:
        sound = [sound]

    return sound
