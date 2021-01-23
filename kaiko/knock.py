import time
import functools
import contextlib
import threading
import shutil
import signal
import numpy
import pyaudio
import audioread
from . import cfg
from . import datanodes as dn
from . import tui


SIGINT_reassignment_time = 0.005

@contextlib.contextmanager
def nullcontext(value):
    yield value

class AudioMixer:
    def __init__(self, effects_scheduler, output_stream):
        self.effects_scheduler = effects_scheduler
        self.output_stream = output_stream

    @classmethod
    @dn.datanode
    def get_output_node(clz, scheduler, samplerate, buffer_length, nchannels, sound_delay):
        index = 0
        with scheduler:
            yield
            while True:
                time = index * buffer_length / samplerate + sound_delay
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                data = scheduler.send((time, data))
                yield data
                index += 1

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, start_time):
        with scheduler:
            time, data = yield
            while True:
                time_ = time - start_time
                data = scheduler.send((time_, data))
                time, data = yield data

    @classmethod
    @contextlib.contextmanager
    def create(clz, manager, settings):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device
        sound_delay = settings.sound_delay
        debug_timeit = settings.debug_timeit

        scheduler = dn.Scheduler()
        output_node = clz.get_output_node(scheduler, samplerate, buffer_length, nchannels, sound_delay)
        output_ctxt = dn.timeit(output_node, " output") if debug_timeit else nullcontext(output_node)

        with output_ctxt as output_node:
            stream_ctxt = dn.play(manager, output_node,
                                  samplerate=samplerate,
                                  buffer_shape=(buffer_length, nchannels),
                                  format=format,
                                  device=device,
                                  )

            with stream_ctxt as stream:
                yield clz(scheduler, stream)

    def start(self):
        return self.output_stream.start_stream()

    def stop(self):
        return self.output_stream.stop_stream()

    def is_active(self):
        return self.output_stream.is_active()

    def is_stopped(self):
        return self.output_stream.is_stopped()

    def close(self):
        return self.output_stream.close()

    def add_effect(self, node, zindex=(0,)):
        return self.effects_scheduler.add_node(node, zindex=zindex)

    def remove_effect(self, key):
        return self.effects_scheduler.remove_node(key)

    @contextlib.contextmanager
    def submixer(self, start_time, zindex=(0,)):
        clz = type(self)
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, start_time)
        subnode_key = self.add_effect(subnode, zindex=zindex)
        submixer = clz(scheduler, self.output_stream)
        try:
            yield submixer
        finally:
            self.remove_effect(subnode_key)

class KnockDetector:
    def __init__(self, listeners_scheduler, input_stream):
        self.listeners_scheduler = listeners_scheduler
        self.input_stream = input_stream

    @classmethod
    @dn.datanode
    def get_input_node(clz, scheduler, samplerate, hop_length, win_length,
                            pre_max, post_max, pre_avg, post_avg, wait, delta,
                            knock_delay, knock_energy):
        prepare = max(post_max, post_avg)

        window = dn.get_half_Hann_window(win_length)
        onset = dn.pipe(
            dn.frame(win_length=win_length, hop_length=hop_length),
            dn.power_spectrum(win_length=win_length,
                              samplerate=samplerate,
                              windowing=window,
                              weighting=True),
            dn.onset_strength(1))
        picker = dn.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta)

        with scheduler, onset, picker:
            data = yield
            buffer = [(knock_delay, 0.0)]*prepare
            index = 0
            while True:
                strength = onset.send(data)
                detected = picker.send(strength)
                time = index * hop_length / samplerate + knock_delay
                strength = strength / knock_energy

                buffer.append((time, strength))
                time, strength = buffer.pop(0)

                scheduler.send(((time, strength, detected), None))
                data = yield

                index += 1

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, start_time):
        with scheduler:
            (time, strength, detected), _ = yield
            while True:
                time_ = time - start_time
                scheduler.send(((time_, strength, detected), None))
                (time, strength, detected), _ = yield

    @classmethod
    @contextlib.contextmanager
    def create(clz, manager, settings):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        time_res = settings.detector_time_res
        freq_res = settings.detector_freq_res
        hop_length = round(samplerate*time_res)
        win_length = round(samplerate/freq_res)

        pre_max  = round(settings.detector_pre_max  / time_res)
        post_max = round(settings.detector_post_max / time_res)
        pre_avg  = round(settings.detector_pre_avg  / time_res)
        post_avg = round(settings.detector_post_avg / time_res)
        wait     = round(settings.detector_wait     / time_res)
        delta    =       settings.detector_delta

        knock_delay = settings.knock_delay
        knock_energy = settings.knock_energy

        debug_timeit = settings.debug_timeit

        scheduler = dn.Scheduler()
        input_node = clz.get_input_node(scheduler, samplerate, hop_length, win_length,
                                        pre_max, post_max, pre_avg, post_avg, wait, delta,
                                        knock_delay, knock_energy)

        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        input_ctxt = dn.timeit(input_node, "  input") if debug_timeit else nullcontext(input_node)

        with input_ctxt as input_node:
            stream_ctxt = dn.record(manager, input_node,
                                    samplerate=samplerate,
                                    buffer_shape=(buffer_length, nchannels),
                                    format=format,
                                    device=device,
                                    )

            with stream_ctxt as stream:
                yield clz(scheduler, stream)

    def start(self):
        return self.input_stream.start_stream()

    def stop(self):
        return self.input_stream.stop_stream()

    def is_active(self):
        return self.input_stream.is_active()

    def is_stopped(self):
        return self.input_stream.is_stopped()

    def close(self):
        return self.input_stream.close()

    def add_listener(self, node):
        return self.listeners_scheduler.add_node(node, (0,))

    def remove_listener(self, key):
        self.listeners_scheduler.remove_node(key)

    @contextlib.contextmanager
    def subdetector(self, start_time):
        clz = type(self)
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, start_time)
        subnode_key = self.add_listener(subnode)
        subdetector = clz(scheduler, self.input_stream)
        try:
            yield subdetector
        finally:
            self.remove_listener(subnode_key)

class MonoRenderer:
    def __init__(self, drawers_scheduler, display_thread):
        self.drawers_scheduler = drawers_scheduler
        self.display_thread = display_thread

    @classmethod
    @dn.datanode
    def get_display_node(clz, scheduler, resize_event, framerate, display_delay, rows, columns):
        width = 0
        height = 0

        def SIGWINCH_handler(sig, frame):
            resize_event.set()
        signal.signal(signal.SIGWINCH, SIGWINCH_handler)
        resize_event.set()

        index = 0
        with scheduler:
            yield
            while True:
                if resize_event.is_set():
                    resize_event.clear()
                    size = shutil.get_terminal_size()
                    width = size.columns if columns == -1 else min(columns, size.columns)
                    height = size.lines if rows == -1 else min(rows, size.lines)

                time = index / framerate + display_delay
                view = [[" "]*width for _ in range(height)]
                view = scheduler.send((time, view))
                yield "\r" + "\n".join(map("".join, view)) + "\r" + (f"\x1b[{height-1}A" if height > 1 else "")
                index += 1

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, start_time):
        with scheduler:
            time, view = yield
            while True:
                time_ = time - start_time
                view = scheduler.send((time_, view))
                time, view = yield view

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings):
        framerate = settings.display_framerate
        display_delay = settings.display_delay
        rows = settings.display_rows
        columns = settings.display_columns
        debug_timeit = settings.debug_timeit

        resize_event = threading.Event()

        @dn.datanode
        def show():
            try:
                while True:
                    current_time, view = yield
                    if view and not resize_event.is_set():
                        print(view, end="", flush=True)
            finally:
                print()

        scheduler = dn.Scheduler()
        display_node = clz.get_display_node(scheduler, resize_event, framerate, display_delay, rows, columns)
        display_ctxt = dn.timeit(display_node, "display") if debug_timeit else nullcontext(display_node)
        with display_ctxt as display_node:
            with dn.interval(display_node, show(), 1/framerate) as thread:
                yield clz(scheduler, thread)

    def start(self):
        return self.display_thread.start()

    def is_active(self):
        return self.display_thread.is_alive()

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    @contextlib.contextmanager
    def subrenderer(self, start_time, zindex=(0,)):
        clz = type(self)
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, start_time)
        subnode_key = self.add_drawer(subnode, zindex=zindex)
        subrenderer = clz(scheduler, self.display_thread)
        try:
            yield subrenderer
        finally:
            self.remove_drawer(subnode_key)


@cfg.configurable
class KerminalSettings:
    # input
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = 'f4'

    # output
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512
    output_channels: int = 1
    output_format: str = 'f4'

    # detector
    detector_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    detector_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100
    detector_pre_max: float = 0.03
    detector_post_max: float = 0.03
    detector_pre_avg: float = 0.03
    detector_post_avg: float = 0.03
    detector_wait: float = 0.03
    detector_delta: float = 5.48e-6

    # controls
    display_framerate: float = 160.0 # ~ 2 / detector_time_res
    display_delay: float = 0.0
    display_rows: int = 1
    display_columns: int = -1
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3
    sound_delay: float = 0.0

    # debug
    debug_timeit: bool = False

class Kerminal:
    def __init__(self, mixer, detector, renderer, ref_time, settings):
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.ref_time = ref_time
        self.settings = settings

    @property
    def time(self):
        return time.time() - self.ref_time

    @classmethod
    def run(clz, knock_program, settings=None):
        if isinstance(settings, str):
            with open(settings, 'r') as file:
                settings = KerminalSettings()
                cfg.config_read(file, main=settings)

        try:
            manager = pyaudio.PyAudio()

            # initialize audio/video streams
            with AudioMixer.create(manager, settings) as mixer,\
                 KnockDetector.create(manager, settings) as detector,\
                 MonoRenderer.create(settings) as renderer:

                # activate audio/video streams
                ref_time = time.time()
                mixer.start()
                detector.start()
                renderer.start()

                self = clz(mixer, detector, renderer, ref_time, settings=settings)

                # connect to program
                SIGINT_event = threading.Event()
                def SIGINT_handler(sig, frame):
                    SIGINT_event.set()
                signal.signal(signal.SIGINT, SIGINT_handler)

                main = threading.Thread(target=knock_program.connect, args=(self, SIGINT_event))

                try:
                    main.start()

                    while main.is_alive() and self.is_active():
                        signal.signal(signal.SIGINT, SIGINT_handler)
                        time.sleep(SIGINT_reassignment_time)

                finally:
                    SIGINT_event.set()
                    main.join()

        finally:
            manager.terminate()

    def is_active(self):
        return self.mixer.is_active() and self.detector.is_active() and self.renderer.is_active()

    def add_effect(self, node, time=None, zindex=(0,)):
        if time is not None:
            node = self._timed_effect(node, time)
        return self.mixer.add_effect(node, zindex)

    def remove_effect(self, key):
        self.mixer.remove_effect(key)

    @functools.lru_cache(maxsize=32)
    def load_sound(self, filepath):
        with audioread.audio_open(filepath) as file:
            samplerate = file.samplerate
        node = dn.load(filepath)
        if samplerate != self.settings.output_samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.settings.output_samplerate, samplerate)))
        with node as filenode:
            sound = list(filenode)
        return sound

    @dn.datanode
    def _timed_effect(self, node, start_time):
        node = dn.DataNode.wrap(node)

        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        with node:
            time, data = yield
            offset = round((start_time - time) * samplerate) if start_time is not None else 0

            while offset < 0:
                length = min(-offset, buffer_length)
                dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                node.send((time+offset/samplerate, dummy))
                offset += length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    data2 = node.send((time+offset/samplerate, data2))
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                time, data = yield data

            while True:
                time, data = yield node.send((time, data))

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None
        if channels is None:
            channels = self.settings.output_channels

        if start is not None or end is not None:
            node = dn.tslice(node, samplerate, start, end)
        node = dn.pipe(node, dn.rechannel(channels))
        if samplerate is not None and samplerate != self.settings.output_samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.settings.output_samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))

        node = dn.pipe(lambda a:a[1], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)

    def add_listener(self, node):
        return self.detector.add_listener(node)

    def remove_listener(self, key):
        self.detector.remove_listener(key)

    def on_hit(self, func, time=None, duration=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    @staticmethod
    def _hit_listener(func, start_time, duration):
        (time, strength, detected), _ = yield
        if start_time is None:
            start_time = time

        while time < start_time:
            (time, strength, detected), _ = yield

        while duration is None or time < start_time + duration:
            if detected:
                finished = func(strength)
                if finished:
                    return

            (time, strength, detected), _ = yield

    def add_drawer(self, node, zindex=(0,)):
        return self.renderer.add_drawer(node, zindex)

    def add_text(self, text_node, y=0, x=0, ymask=slice(None,None), xmask=slice(None,None), zindex=(0,)):
        return self.renderer.add_drawer(self._text_drawer(text_node, y, x, ymask, xmask), zindex)

    def add_pad(self, pad_node, ymask=slice(None,None), xmask=slice(None,None), zindex=(0,)):
        return self.renderer.add_drawer(self._pad_drawer(pad_node, ymask, xmask), zindex)

    def remove_drawer(self, key):
        self.renderer.remove_drawer(key)

    @dn.datanode
    @staticmethod
    def _text_drawer(text_node, y=0, x=0, ymask=slice(None,None), xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            time, view = yield
            while True:
                rows = len(view)
                columns = len(view[0]) if view else 0

                text = text_node.send((time, range(-y, rows-y)[ymask], range(-x, columns-x)[xmask]))
                view, y, x = tui.addtext(view, y, x, text, ymask=ymask, xmask=xmask)

                time, view = yield view

    @dn.datanode
    @staticmethod
    def _pad_drawer(pad_node, ymask=slice(None,None), xmask=slice(None,None)):
        pad_node = dn.DataNode.wrap(pad_node)
        with pad_node:
            time, view = yield
            while True:
                subview, y, x = tui.newpad(view, ymask=ymask, xmask=xmask)
                subview = pad_node.send((time, subview))
                view, yran, xran = tui.addpad(view, y, x, subview)

                time, view = yield view

    @contextlib.contextmanager
    def subkerminal(self, start_time=None):
        if start_time is None:
            start_time = self.time

        with self.mixer.submixer(start_time) as submixer,\
             self.detector.subdetector(start_time) as subdetector,\
             self.renderer.subrenderer(start_time) as subrenderer:

            ref_time = self.ref_time + start_time
            yield Kerminal(submixer, subdetector, subrenderer, ref_time, self.settings)
