import time
import bisect
import functools
import contextlib
import dataclasses
import queue
import threading
import numpy
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from . import terminals as term
from . import audios as aud
from . import clocks


class Monitor:
    def __init__(self, path, N=10):
        self.path = path
        self.N = N

        # state
        self.count = None
        self.eff = None
        self.avg = None
        self.best = None
        self.worst = None

        # statistics
        self.total_avg = None
        self.total_dev = None
        self.total_eff = None

    @dn.datanode
    def monitoring(self, node):
        if hasattr(time, "thread_time"):
            get_time = time.thread_time
        elif hasattr(time, "clock_gettime"):
            get_time = lambda: time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        else:
            get_time = time.perf_counter

        N = self.N
        start = prev = 0.0
        stop = numpy.inf
        self.count = 0
        total = 0.0
        total2 = 0.0
        spend_N = [0.0] * N
        recent_N = [0.0] * N
        best_N = [numpy.inf] * N
        worst_N = [-numpy.inf] * N

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w") as file:
            with node:
                try:
                    data = yield

                    start = stop = prev = time.perf_counter()

                    while True:

                        t0 = get_time()
                        data = node.send(data)
                        t = get_time() - t0
                        stop = time.perf_counter()
                        spend = stop - prev
                        prev = stop
                        print(f"{spend}\t{t}", file=file)

                        self.count += 1
                        total += t
                        total2 += t ** 2
                        spend_N.insert(0, spend)
                        spend_N.pop()
                        recent_N.insert(0, t)
                        recent_N.pop()
                        bisect.insort_left(best_N, t)
                        best_N.pop()
                        bisect.insort(worst_N, t)
                        worst_N.pop(0)
                        self.avg = sum(recent_N) / N
                        self.eff = sum(recent_N) / sum(spend_N)
                        self.best = sum(best_N) / N
                        self.worst = sum(worst_N) / N

                        data = yield data

                except StopIteration:
                    return

                finally:
                    stop = time.perf_counter()

                    if self.count > 0:
                        self.total_avg = total / self.count
                        self.total_dev = (
                            total2 / self.count - self.total_avg ** 2
                        ) ** 0.5
                        self.total_eff = total / (stop - start)

    def __str__(self):
        if self.count is None:
            return f"UNINITIALIZED"

        if self.total_avg is None:
            return f"count={self.count}"

        assert self.total_dev is not None
        assert self.total_eff is not None

        if self.best is None or self.best == float("inf"):
            return (
                f"count={self.count}, "
                f"avg={self.total_avg*1000:5.3f}±{self.total_dev*1000:5.3f}ms "
                f"({self.total_eff: >6.1%})"
            )

        assert self.worst is not None

        return (
            f"count={self.count}, "
            f"avg={self.total_avg*1000:5.3f}±{self.total_dev*1000:5.3f}ms"
            f" ({self.best*1000:5.3f}ms ~ {self.worst*1000:5.3f}ms) "
            f"({self.total_eff: >6.1%})"
        )


@dn.datanode
def _task_init_time(engine):
    yield
    engine.init_time = time.perf_counter()
    yield
    while True:
        yield

def _set_init_time(engine, task):
    if engine.init_time is not None:
        return task
    else:
        return dn.pipe(_task_init_time(engine), task)


class MixerSettings(cfg.Configurable):
    r"""
    Fields
    ------
    output_device : int
        The index of output device, or -1 for default device.
    output_samplerate : int
        The samplerate of output device.
    output_buffer_length : int
        The buffer length of output device. Note that too large will affect the
        reaction time, but too small will affect the efficiency.
    output_channels : int
        The number of channels of output device.
    output_format : str
        The data format of output device. The valid formats are 'f4', 'i4',
        'i2', 'i1', 'u1'.

    sound_delay : float
        The delay of clock of the mixer.
    """
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512 * 4
    output_channels: int = 1
    output_format: str = "f4"

    sound_delay: float = 0.0


class Mixer:
    def __init__(self, effects_pipeline, init_time, settings, monitor):
        self.effects_pipeline = effects_pipeline
        self.init_time = init_time
        self.settings = settings
        self.monitor = monitor

    @classmethod
    def create(cls, settings, manager, clock, init_time=None, monitor=None):
        pipeline = dn.DynamicPipeline()
        self = cls(pipeline, init_time, settings, monitor)
        return self._task(manager, clock, monitor), self

    def _task(self, manager, clock, monitor):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels
        format = self.settings.output_format
        device = self.settings.output_device
        sound_delay = self.settings.sound_delay

        tick_node = clock.tick_slice("mixer", sound_delay)
        output_node = self._mix_node(self.effects_pipeline, tick_node, self.settings)
        if monitor:
            output_node = monitor.monitoring(output_node)

        task = aud.play(
            manager,
            output_node,
            samplerate=samplerate,
            buffer_shape=(buffer_length, nchannels),
            format=format,
            device=device,
        )

        return _set_init_time(self, task)

    @dn.datanode
    def _mix_node(self, pipeline, tick_node, settings):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        timer = dn.pipe(
            dn.count(0, 1),
            lambda index: slice(
                index * buffer_length / samplerate,
                (index + 1) * buffer_length / samplerate,
            ),
        )

        with timer, tick_node, pipeline:
            yield
            init_time = self.init_time
            while True:
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                try:
                    time_slice = timer.send(None)
                    time_slice = slice(time_slice.start + init_time, time_slice.stop + init_time)
                    slices_map = tick_node.send(time_slice)
                    data = pipeline.send((data, slices_map))
                except StopIteration:
                    return
                yield data

    def add_effect(self, node, zindex=(0,)):
        return self.effects_pipeline.add_node(node, zindex=zindex)

    def remove_effect(self, key):
        return self.effects_pipeline.remove_node(key)

    @dn.datanode
    def tmask(self, node, time):
        node = dn.DataNode.wrap(node)

        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        with node:
            data, slices_map = yield
            if time is None:
                time = slices_map[0][1].start

            while True:
                data_offset = round(slices_map[0][0].start * samplerate)
                for data_slice, time_slice, ratio in slices_map:
                    # pause
                    if time_slice.start == time_slice.stop:
                        continue

                    # skip
                    if data_slice.start == data_slice.stop:
                        time_slice = slice(time_slice.stop, time_slice.stop)

                    offset = round((time - time_slice.start) * samplerate)

                    # underrun
                    while offset < 0:
                        length = min(-offset, buffer_length)
                        dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                        try:
                            node.send(dummy)
                        except StopIteration:
                            return
                        offset += length
                        time += length / samplerate

                    # overrun
                    data_start = round(data_slice.start * samplerate) - data_offset
                    data_stop = round(data_slice.stop * samplerate) - data_offset
                    if data_stop - data_start <= offset:
                        continue

                    try:
                        data[data_start + offset : data_stop] = node.send(
                            data[data_start + offset : data_stop]
                        )
                    except StopIteration:
                        return

                    time = time_slice.stop

                data, slices_map = yield data.copy()

    def resample(
        self, samplerate=None, channels=None, volume=0.0, start=None, end=None
    ):
        pipeline = []
        if start is not None or end is not None:
            pipeline.append(
                dn.tspan(samplerate or self.settings.output_samplerate, start, end)
            )
        if channels is not None and channels != self.settings.output_channels:
            pipeline.append(dn.rechannel(self.settings.output_channels, channels))
        if samplerate is not None and samplerate != self.settings.output_samplerate:
            pipeline.append(
                dn.resample(ratio=(self.settings.output_samplerate, samplerate))
            )
        if volume != 0:
            pipeline.append(lambda s: s * 10 ** (volume / 20))
        return dn.pipe(*pipeline)

    def play(
        self,
        node,
        samplerate=None,
        channels=None,
        volume=0.0,
        start=None,
        end=None,
        time=None,
        zindex=(0,),
    ):
        node = dn.pipe(node, self.resample(samplerate, channels, volume, start, end))
        node = self.tmask(dn.attach(node), time)
        return self.add_effect(node, zindex=zindex)

    def play_file(self, path, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        meta = aud.AudioMetadata.read(path)
        node = aud.load(path)
        node = dn.tslice(node, meta.samplerate, start, end)
        # initialize before attach; it will seek to the starting frame
        node.__enter__()

        return self.play(
            node,
            samplerate=meta.samplerate,
            channels=meta.channels,
            volume=volume,
            time=time,
            zindex=zindex,
        )


class AsyncAdditiveValue:
    def __init__(self, value):
        self.value = value
        self._queue = queue.Queue()

    def add(self, step):
        self._queue.put(step)

    def get(self):
        while not self._queue.empty():
            self.value += self._queue.get()
        return self.value


class DetectorSettings(cfg.Configurable):
    r"""
    Fields
    ------
    input_device : int
        The index of input device, or -1 for default device.
    input_samplerate : int
        The samplerate of input device.
    input_buffer_length : int
        The buffer length of input device. Note that too large will affect the
        reaction time, but too small will affect the efficiency.
    input_channels : int
        The number of channels of input device.
    input_format : str
        The data format of input device. The valid formats are 'f4', 'i4', 'i2',
        'i1', 'u1'.

    knock_delay : float
        The delay of clock of the detector.
    knock_energy : float
        The reference volume of the detector.
    """
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = "f4"

    @cfg.subconfig
    class detect(cfg.Configurable):
        time_res: float = 0.0116099773  # hop_length = 512 if samplerate == 44100
        freq_res: float = 21.5332031  # win_length = 512*4 if samplerate == 44100
        pre_max: float = 0.03
        post_max: float = 0.03
        pre_avg: float = 0.03
        post_avg: float = 0.03
        wait: float = 0.03
        delta: float = 5.48e-6  # ~ noise_power * 20

    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3  # ~ Dt / knock_max_energy


class Detector:
    def __init__(self, listeners_pipeline, knock_energy, init_time, settings, monitor):
        self.listeners_pipeline = listeners_pipeline
        self.knock_energy = knock_energy
        self.init_time = init_time
        self.settings = settings
        self.monitor = monitor

    @classmethod
    def create(cls, settings, manager, clock, init_time=None, monitor=None):
        pipeline = dn.DynamicPipeline()
        knock_energy = AsyncAdditiveValue(settings.knock_energy)
        self = cls(pipeline, knock_energy, init_time, settings, monitor)
        return self._task(manager, clock, monitor), self

    def _task(self, manager, clock, monitor):
        samplerate = self.settings.input_samplerate
        buffer_length = self.settings.input_buffer_length
        nchannels = self.settings.input_channels
        format = self.settings.input_format
        device = self.settings.input_device
        time_res = self.settings.detect.time_res
        hop_length = round(samplerate * time_res)
        knock_delay = self.settings.knock_delay

        tick_node = clock.tick("detector", knock_delay)

        input_node = self._detect_node(
            self.listeners_pipeline, tick_node, self.knock_energy, self.settings
        )
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        if monitor:
            input_node = monitor.monitoring(input_node)

        task = aud.record(
            manager,
            input_node,
            samplerate=samplerate,
            buffer_shape=(buffer_length, nchannels),
            format=format,
            device=device,
        )

        return _set_init_time(self, task)

    @dn.datanode
    def _detect_node(self, pipeline, tick_node, knock_energy, settings):
        samplerate = settings.input_samplerate

        time_res = settings.detect.time_res
        freq_res = settings.detect.freq_res
        hop_length = round(samplerate * time_res)
        win_length = round(samplerate / freq_res)

        pre_max = round(settings.detect.pre_max / time_res)
        post_max = round(settings.detect.post_max / time_res)
        pre_avg = round(settings.detect.pre_avg / time_res)
        post_avg = round(settings.detect.post_avg / time_res)
        wait = round(settings.detect.wait / time_res)
        delta = settings.detect.delta

        prepare = max(post_max, post_avg)

        timer = dn.count(-prepare * hop_length / samplerate, hop_length / samplerate)

        window = dn.get_half_Hann_window(win_length)
        onset = dn.pipe(
            dn.frame(win_length=win_length, hop_length=hop_length),
            dn.power_spectrum(
                win_length=win_length,
                samplerate=samplerate,
                windowing=window,
                weighting=True,
            ),
            dn.onset_strength(1),
        )

        picker = dn.pipe(
            lambda a: (a, a),
            dn.pair(
                dn.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta),
                dn.delay(0.0 for _ in range(-prepare, 0)),
            ),
        )

        with pipeline, timer, tick_node, onset, picker:
            data = yield
            init_time = self.init_time
            while True:
                try:
                    strength = onset.send(data)
                    detected, strength = picker.send(strength)
                    time = timer.send(None)
                    time += init_time
                    tick, ratio = tick_node.send(time)
                    normalized_strength = strength / knock_energy.get()
                    pipeline.send((None, tick, ratio, normalized_strength, detected))
                except StopIteration:
                    return
                data = yield

    def add_listener(self, node):
        return self.listeners_pipeline.add_node(node, (0,))

    def remove_listener(self, key):
        self.listeners_pipeline.remove_node(key)

    def on_hit(self, func, time=None, duration=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    @staticmethod
    def _hit_listener(func, start_time, duration):
        _, time, ratio, strength, detected = yield
        if start_time is None:
            start_time = time

        while time < start_time:
            _, time, ratio, strength, detected = yield

        while duration is None or time < start_time + duration:
            if detected and ratio != 0:
                finished = func(strength)
                if finished:
                    return

            _, time, ratio, strength, detected = yield


@functools.lru_cache(maxsize=32)
def to_range(start, stop, width):
    # range of slice without clamp
    if start is None:
        start = 0
    elif start < 0:
        start = width + start
    else:
        start = start

    if stop is None:
        stop = width
    elif stop < 0:
        stop = width + stop
    else:
        stop = stop

    return range(start, stop)


class RichBar:
    def __init__(self, terminal_settings):
        self.markups = []

    def add_markup(self, markup, mask=slice(None, None), shift=0):
        self.markups.append((markup, mask, shift))

    def draw(self, width, renderer):
        buffer = [" "] * width
        xran = range(width)

        for markup, mask, shift in self.markups:
            if mask.start is None:
                x = shift
            elif mask.start >= 0:
                x = shift + mask.start
            else:
                x = shift + mask.start + width

            renderer._render_bar(
                buffer, markup, x=x, width=width, xmask=xran[mask], attrs=()
            )

        return "".join(buffer).rstrip()


class RendererSettings(cfg.Configurable):
    r"""
    Fields
    ------
    display_framerate : float
        The framerate of the renderer.
    display_delay : float
        The delay of clock of the renderer.
    resize_delay : float
        The delay time to redraw display after resizing.
    """
    display_framerate: float = 160.0  # ~ 2 / detect.time_res
    display_delay: float = 0.0
    resize_delay: float = 0.5


class Renderer:
    def __init__(self, drawers_pipeline, init_time, settings, monitor):
        self.drawers_pipeline = drawers_pipeline
        self.init_time = init_time
        self.settings = settings
        self.monitor = monitor

    @classmethod
    def create(cls, settings, term_settings, clock, init_time=None, monitor=None):
        pipeline = dn.DynamicPipeline()
        self = cls(pipeline, init_time, settings, monitor)
        return self._task(term_settings, clock, monitor), self

    def _task(self, term_settings, clock, monitor):
        framerate = self.settings.display_framerate
        display_delay = self.settings.display_delay

        tick_node = clock.tick("renderer", display_delay)
        render_node = self._render_node(self.drawers_pipeline, tick_node, self.settings, term_settings)
        display_node = self._resize_node(
            render_node,
            self.settings,
            term_settings,
        )
        if monitor:
            display_node = monitor.monitoring(display_node)

        task = term.show(display_node, 1 / framerate, hide_cursor=True)

        return _set_init_time(self, task)

    @dn.datanode
    def _render_node(self, pipeline, tick_node, settings, term_settings):
        framerate = settings.display_framerate

        rich_renderer = mu.RichRenderer(term_settings.unicode_version)
        clear_line = rich_renderer.render(rich_renderer.clear_line().expand())
        clear_below = rich_renderer.render(rich_renderer.clear_below().expand())
        width = 0
        logs = []
        msgs = []
        curr_msgs = list(msgs)

        timer = dn.count(1 / framerate, 1 / framerate)

        with timer, tick_node, pipeline:
            shown, resized, size = yield
            init_time = self.init_time
            while True:
                width = size.columns
                view = RichBar(term_settings)
                try:
                    time = timer.send(None)
                    time += init_time
                    tick, ratio = tick_node.send(time)
                    view, msgs, logs = pipeline.send(((view, msgs, logs), tick, width))
                except StopIteration:
                    return
                view_str = view.draw(width, rich_renderer)

                logs_str = (
                    rich_renderer.render(mu.Group(tuple(logs))) + "\r" if logs else ""
                )

                # track changes of the message
                if not resized and not logs and curr_msgs == msgs:
                    res_text = f"{clear_line}{view_str}\r"
                else:
                    msg_text = (
                        rich_renderer.render_less(
                            mu.Group((mu.Text("\n"), *msgs)), size
                        )
                        if msgs
                        else ""
                    )
                    res_text = f"{clear_below}{logs_str}{view_str}\r{msg_text}"

                shown, resized, size = yield res_text
                if shown:
                    logs.clear()
                    curr_msgs = list(msgs)

    @staticmethod
    @dn.datanode
    def _resize_node(render_node, settings, term_settings):
        resize_delay = settings.resize_delay

        rich_renderer = mu.RichRenderer(term_settings.unicode_version)
        clear_line = rich_renderer.render(rich_renderer.clear_line().expand())
        clear_screen = rich_renderer.render(rich_renderer.clear_screen().expand())
        size_node = term.terminal_size()

        width = 0
        resizing_since = time.perf_counter()
        resized = False
        with render_node, size_node:
            shown = yield
            while True:
                try:
                    size = size_node.send(None)
                except StopIteration:
                    return

                if size.columns < width:
                    resizing_since = time.perf_counter()
                    resized = True
                if resized and time.perf_counter() < resizing_since + resize_delay:
                    yield f"{clear_line}resizing...\r"
                    width = size.columns
                    continue

                width = size.columns
                try:
                    res_text = render_node.send((shown, resized, size))
                except StopIteration:
                    return
                if resized:
                    res_text = f"{clear_screen}{res_text}"

                shown = yield res_text
                if shown:
                    resized = False

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_pipeline.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_pipeline.remove_node(key)

    def add_log(self, msg, zindex=(0,)):
        return self.add_drawer(self._log_drawer(msg), zindex)

    def add_message(self, msg, zindex=(0,)):
        return self.add_drawer(self._msg_drawer(msg), zindex)

    def add_texts(self, texts_node, xmask=slice(None, None), zindex=(0,)):
        return self.add_drawer(self._texts_drawer(texts_node, xmask), zindex)

    @staticmethod
    @dn.datanode
    def _log_drawer(msg):
        (view, msgs, logs), _, _ = yield
        logs.append(msg)
        yield (view, msgs, logs)

    @staticmethod
    @dn.datanode
    def _msg_drawer(msg):
        (view, msgs, logs), _, _ = yield
        msgs.append(msg)
        yield (view, msgs, logs)

    @staticmethod
    @dn.datanode
    def _texts_drawer(text_node, xmask=slice(None, None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            (view, msg, logs), time, width = yield
            while True:
                xran = to_range(xmask.start, xmask.stop, width)

                try:
                    texts = text_node.send((time, xran))
                except StopIteration:
                    return

                for shift, text in texts:
                    view.add_markup(text, xmask, shift=shift)

                (view, msg, logs), time, width = yield (view, msg, logs)


class ControllerSettings(cfg.Configurable):
    r"""
    Fields
    ------
    update_interval : float
        The update interval of controllers. This is not related to precision of
        time, it is timeout of stdin select. The user will only notice the
        difference in latency when closing the controller.
    """
    update_interval: float = 0.1


class Controller:
    def __init__(self, handlers_pipeline, init_time, settings):
        self.handlers_pipeline = handlers_pipeline
        self.init_time = init_time
        self.settings = settings

    @classmethod
    def create(cls, settings, term_settings, clock, init_time=None):
        pipeline = dn.DynamicPipeline()
        self = cls(pipeline, init_time, settings)
        return self._task(term_settings, clock), self

    def _task(self, term_settings, clock):
        update_interval = self.settings.update_interval

        tick_node = clock.tick("controller", 0.0)

        task = term.inkey(
            self._control_node(
                self.handlers_pipeline, tick_node, self.settings, term_settings
            ),
            dt=update_interval,
        )

        return _set_init_time(self, task)

    @dn.datanode
    def _control_node(self, pipeline, tick_node, settings, term_settings):
        keycodes = term_settings.keycodes

        timer = dn.time(0.0)

        with timer, tick_node, pipeline:
            _, keycode = yield
            init_time = self.init_time

            while True:
                if keycode is None:
                    keyname = None
                elif keycode in keycodes:
                    keyname = keycodes[keycode]
                elif keycode in term.printable_ascii_names:
                    keyname = term.printable_ascii_names[keycode]
                elif keycode[0] == "\x1b" and keycode[1:] in term.printable_ascii_names:
                    keyname = "Alt_" + term.printable_ascii_names[keycode[1:]]
                else:
                    keyname = repr(keycode)

                try:
                    time = timer.send(None)
                    time += init_time
                    tick, ratio = tick_node.send(time)
                    pipeline.send((None, tick, keyname, keycode))
                except StopIteration:
                    return

                _, keycode = yield

    def add_handler(self, node, keyname=None):
        return self.handlers_pipeline.add_node(self._filter_node(node, keyname), (0,))

    def remove_handler(self, key):
        self.handlers_pipeline.remove_node(key)

    @dn.datanode
    def _filter_node(self, node, name):
        node = dn.DataNode.wrap(node)
        with node:
            while True:
                _, time, keyname, keycode = yield
                if keycode is None:
                    continue
                if name is None or name == keyname:
                    try:
                        node.send((None, time, keyname, keycode))
                    except StopIteration:
                        return


class EngineLoader:
    def __init__(self, engine_factory, ratain_time=3.0):
        self.engine_factory = engine_factory
        self.ratain_time = ratain_time
        self.required = set()

        self.require_lock = threading.Lock()
        self.engine_task = None
        self.engine = None

    @contextlib.contextmanager
    def require(self):
        key = object()

        with self.require_lock:
            self.required.add(key)
            if self.engine is None:
                self.engine_task, self.engine = self.engine_factory()
            engine = self.engine

        try:
            yield engine
        finally:
            with self.require_lock:
                self.required.remove(key)

    @dn.datanode
    def task(self):
        while True:
            yield
            with self.require_lock:
                if self.engine_task is None:
                    continue

            with self.engine_task:
                expiration = None
                while True:
                    with self.require_lock:
                        current_time = time.perf_counter()
                        if expiration is None and not self.required:
                            expiration = current_time + self.ratain_time
                        if expiration is not None and self.required:
                            expiration = None
                        if expiration is not None and current_time >= expiration:
                            self.engine_task = None
                            self.engine = None
                            break

                    yield

                    try:
                        self.engine_task.send(None)
                    except StopIteration:
                        raise RuntimeError("engine stop unexpectedly")


class DevicesSettings(cfg.Configurable):
    mixer = cfg.subconfig(MixerSettings)
    detector = cfg.subconfig(DetectorSettings)
    renderer = cfg.subconfig(RendererSettings)
    controller = cfg.subconfig(ControllerSettings)
    terminal = cfg.subconfig(term.TerminalSettings)
