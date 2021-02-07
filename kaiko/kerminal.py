import time
import itertools
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


@contextlib.contextmanager
def nullcontext(value):
    yield value

class KerminalMixer:
    def __init__(self, effects_scheduler, samplerate, buffer_length, nchannels):
        self.effects_scheduler = effects_scheduler
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.nchannels = nchannels

    @staticmethod
    @dn.datanode
    def get_node(scheduler, samplerate, buffer_length, nchannels, sound_delay):
        index = 0
        with scheduler:
            yield
            while True:
                time = index * buffer_length / samplerate + sound_delay
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                data = scheduler.send((data, time))
                yield data
                index += 1

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
        output_node = clz.get_node(scheduler, samplerate, buffer_length, nchannels, sound_delay)
        output_ctxt = dn.timeit(output_node, " output") if debug_timeit else nullcontext(output_node)

        with output_ctxt as output_node:
            stream_ctxt = dn.play(manager, output_node,
                                  samplerate=samplerate,
                                  buffer_shape=(buffer_length, nchannels),
                                  format=format,
                                  device=device,
                                  )

            with stream_ctxt as stream:
                yield stream, clz(scheduler, samplerate, buffer_length, nchannels)

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, ref_time):
        with scheduler:
            data, time = yield
            while True:
                time_ = time - ref_time
                data = scheduler.send((data, time_))
                data, time = yield data

    @classmethod
    @contextlib.contextmanager
    def submixer(clz, mixer, ref_time, zindex=(0,)):
        samplerate = mixer.samplerate
        buffer_length = mixer.buffer_length
        nchannels = mixer.nchannels

        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, ref_time)
        subnode_key = mixer.add_effect(subnode, zindex=zindex)
        try:
            yield clz(scheduler, samplerate, buffer_length, nchannels)
        finally:
            mixer.remove_effect(subnode_key)

    def remove_effect(self, key):
        return self.effects_scheduler.remove_node(key)

    def add_effect(self, node, time=None, zindex=(0,)):
        if time is not None:
            node = self.delay(node, time)
        return self.effects_scheduler.add_node(node, zindex=zindex)

    @dn.datanode
    def delay(self, node, start_time):
        node = dn.DataNode.wrap(node)

        samplerate = self.samplerate
        buffer_length = self.buffer_length
        nchannels = self.nchannels

        with node:
            data, time = yield
            offset = round((start_time - time) * samplerate) if start_time is not None else 0

            while offset < 0:
                length = min(-offset, buffer_length)
                dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                node.send((dummy, time+offset/samplerate))
                offset += length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    data2 = node.send((data2, time+offset/samplerate))
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                data, time = yield data

            while True:
                data, time = yield node.send((data, time))

    def resample(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None):
        if channels is None:
            channels = self.nchannels

        if start is not None or end is not None:
            node = dn.tslice(node, samplerate, start, end)
        node = dn.pipe(node, dn.rechannel(channels))
        if samplerate is not None and samplerate != self.samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))

        return node

    @functools.lru_cache(maxsize=32)
    def load_sound(self, filepath):
        with audioread.audio_open(filepath) as file:
            samplerate = file.samplerate
        node = dn.load(filepath)
        if samplerate != self.samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.samplerate, samplerate)))
        with node as filenode:
            sound = list(filenode)
        return sound

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None

        node = self.resample(node, samplerate, channels, volume, start, end)

        node = dn.pipe(lambda a:a[0], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)

class KerminalDetector:
    def __init__(self, listeners_scheduler):
        self.listeners_scheduler = listeners_scheduler

    @staticmethod
    @dn.datanode
    def get_node(scheduler, samplerate, hop_length, win_length,
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

                scheduler.send((None, time, strength, detected))
                data = yield

                index += 1

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
        input_node = clz.get_node(scheduler, samplerate, hop_length, win_length,
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
                yield stream, clz(scheduler)

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, ref_time):
        with scheduler:
            _, time, strength, detected = yield
            while True:
                time_ = time - ref_time
                scheduler.send((None, time_, strength, detected))
                _, time, strength, detected = yield

    @classmethod
    @contextlib.contextmanager
    def subdetector(clz, detector, ref_time):
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, ref_time)
        subnode_key = detector.add_listener(subnode)
        try:
            yield clz(scheduler)
        finally:
            detector.remove_listener(subnode_key)

    def add_listener(self, node):
        return self.listeners_scheduler.add_node(node, (0,))

    def remove_listener(self, key):
        self.listeners_scheduler.remove_node(key)

    def on_hit(self, func, time=None, duration=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    @staticmethod
    def _hit_listener(func, start_time, duration):
        _, time, strength, detected = yield
        if start_time is None:
            start_time = time

        while time < start_time:
            _, time, strength, detected = yield

        while duration is None or time < start_time + duration:
            if detected:
                finished = func(strength)
                if finished:
                    return

            _, time, strength, detected = yield

class KerminalRenderer:
    def __init__(self, drawers_scheduler):
        self.drawers_scheduler = drawers_scheduler

    @staticmethod
    @dn.datanode
    def get_node(scheduler, resize_event, framerate, display_delay, columns):
        width = 0

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

                time = index / framerate + display_delay
                view = tui.newwin1(width)
                view = scheduler.send((view, time, width))
                yield "\r" + "".join(view) + "\r"
                index += 1

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings):
        framerate = settings.display_framerate
        display_delay = settings.display_delay
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
        display_node = clz.get_node(scheduler, resize_event, framerate, display_delay, columns)
        display_ctxt = dn.timeit(display_node, "display") if debug_timeit else nullcontext(display_node)
        with display_ctxt as display_node:
            with dn.interval(display_node, show(), 1/framerate) as thread:
                yield thread, clz(scheduler)

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, ref_time):
        with scheduler:
            view, time, width = yield
            while True:
                time_ = time - ref_time
                view = scheduler.send((view, time_, width))
                view, time, width = yield view

    @classmethod
    @contextlib.contextmanager
    def subrenderer(clz, renderer, ref_time, zindex=(0,)):
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, ref_time)
        subnode_key = renderer.add_drawer(subnode, zindex=zindex)
        try:
            yield clz(scheduler)
        finally:
            renderer.remove_drawer(subnode_key)

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    def add_text(self, text_node, x=0, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._text_drawer(text_node, x, xmask), zindex)

    def add_pad(self, pad_node, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._pad_drawer(pad_node, xmask), zindex)

    @dn.datanode
    @staticmethod
    def _text_drawer(text_node, x=0, xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            view, time, width = yield
            while True:
                text = text_node.send((time, range(-x, width-x)[xmask]))
                view, x = tui.addtext1(view, width, x, text, xmask=xmask)

                view, time, width = yield view

    @dn.datanode
    @staticmethod
    def _pad_drawer(pad_node, xmask=slice(None,None)):
        pad_node = dn.DataNode.wrap(pad_node)
        with pad_node:
            view, time, width = yield
            while True:
                subview, x, subwidth = tui.newpad1(view, width, xmask=xmask)
                subview = pad_node.send(((time, subwidth), subview))
                view, xran = tui.addpad1(view, width, x, subview, subwidth)

                view, time, width = yield view

class KerminalController:
    def __init__(self, handlers_scheduler):
        self.handlers_scheduler = handlers_scheduler

    @staticmethod
    @dn.datanode
    def get_node(scheduler):
        with scheduler:
            while True:
                t, key = yield
                scheduler.send((None, t, key))

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings):
        scheduler = dn.Scheduler()
        node = clz.get_node(scheduler)
        with dn.input(node) as thread:
            yield thread, clz(scheduler)

    @classmethod
    @dn.datanode
    def get_subnode(clz, scheduler, ref_time):
        with scheduler:
            _, time, key = yield
            while True:
                time_ = time - ref_time
                scheduler.send((None, time_, key))
                _, time, key = yield

    @classmethod
    @contextlib.contextmanager
    def subcontroller(clz, controller, ref_time):
        scheduler = dn.Scheduler()
        subnode = clz.get_subnode(scheduler, ref_time)
        subnode_key = controller.add_handler(subnode)
        try:
            yield clz(scheduler)
        finally:
            controller.remove_handler(subnode_key)

    def add_handler(self, node):
        return self.handlers_scheduler.add_node(node, (0,))

    def remove_handler(self, key):
        self.handlers_scheduler.remove_node(key)


def until_interrupt(dt=0.005):
    SIGINT_event = threading.Event()
    def SIGINT_handler(sig, frame):
        SIGINT_event.set()
    signal.signal(signal.SIGINT, SIGINT_handler)

    while True:
        yield

        if SIGINT_event.wait(dt):
            break

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
    output_buffer_length: int = 512*4
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
    display_columns: int = -1
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3
    sound_delay: float = 0.0

    # debug
    debug_timeit: bool = False

class Kerminal:
    def __init__(self, mixer, detector, renderer, controller):
        self.ref_time = None
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller

    @classmethod
    def execute(clz, knockable, settings=None):
        if isinstance(settings, str):
            with open(settings, 'r') as file:
                settings = KerminalSettings()
                cfg.config_read(file, main=settings)

        try:
            manager = pyaudio.PyAudio()

            # initialize audio/text streams
            with KerminalMixer.create(manager, settings) as (audioout_stream, mixer),\
                 KerminalDetector.create(manager, settings) as (audioin_stream, detector),\
                 KerminalRenderer.create(settings) as (textout_thread, renderer),\
                 KerminalController.create(settings) as (textin_thread, controller):

                # initialize kerminal
                self = clz(mixer, detector, renderer, controller)

                # activate audio/video streams
                self.ref_time = time.time()
                audioout_stream.start_stream()
                audioin_stream.start_stream()
                textout_thread.start()
                textin_thread.start()

                # connect to knockable
                with knockable.connect(self) as main:
                    main.start()

                    for _ in until_interrupt():
                        if (not main.is_alive()
                            or not audioout_stream.is_active()
                            or not audioin_stream.is_active()
                            or not textout_thread.is_alive()
                            or not textin_thread.is_alive()):
                            break

        finally:
            manager.terminate()

    @classmethod
    @contextlib.contextmanager
    def subkerminal(clz, kerminal, ref_time=0.0):
        with KerminalMixer.submixer(kerminal.mixer, ref_time) as submixer,\
             KerminalDetector.subdetector(kerminal.detector, ref_time) as subdetector,\
             KerminalRenderer.subrenderer(kerminal.renderer, ref_time) as subrenderer,\
             KerminalController.subcontroller(kerminal.controller, ref_time) as subcontroller:

            subkerminal = clz(submixer, subdetector, subrenderer, subcontroller)
            subkerminal.ref_time = kerminal.ref_time + ref_time
            yield subkerminal


    @property
    def time(self):
        return time.time() - self.ref_time

    def tick(self, dt, t0=0.0, stop_event=None):
        if stop_event is None:
            stop_event = threading.Event()

        for i in itertools.count():
            if stop_event.wait(max(0.0, t0+i*dt - self.time)):
                break

            yield self.time
