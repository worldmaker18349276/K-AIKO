import time
import bisect
import functools
import dataclasses
import queue
import numpy
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from . import terminals as term
from . import audios as aud
from . import loggers as log


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


class ClockAction:
    pass


@dataclasses.dataclass(frozen=True)
class ClockPause(ClockAction):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockResume(ClockAction):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockSkip(ClockAction):
    time: float
    delta: float


@dn.datanode
def clock(offset, slope, actions):
    slope0 = slope

    waited = None
    action = None

    time = yield
    while True:
        # find unhandled action
        if waited is None and not actions.empty():
            waited = actions.get()
        if waited is not None and waited.time <= time:
            action, waited = waited, None
        else:
            action = None

        # update state
        if action is None:
            time = yield offset + time * slope
        elif isinstance(action, ClockResume):
            offset, slope = offset + action.time * (slope - slope0), slope0
        elif isinstance(action, ClockPause):
            offset, slope = offset + action.time * slope, 0
        elif isinstance(action, ClockSkip):
            offset += action.delta
        else:
            raise TypeError


@dn.datanode
def clock_slice(offset, slope, actions):
    slope0 = slope

    slices_map = []
    waited = None
    action = None

    time_slice = yield
    while True:
        # find unhandle action
        if waited is None and not actions.empty():
            waited = actions.get()
        if waited is not None and waited.time <= time_slice.stop:
            action, waited = waited, None
        else:
            action = None

        # slice
        if action is not None:
            # if action.time < time_slice.start:
            #     print("clock underrun")
            time = action.time
        else:
            time = time_slice.stop
        if time_slice.start < time:
            slices_map.append(
                (
                    slice(time_slice.start, time),
                    slice(offset + time_slice.start * slope, offset + time * slope),
                )
            )
            time_slice = slice(time, time_slice.stop)

        # update state
        if action is None:
            time_slice = yield slices_map
            slices_map = []
        elif isinstance(action, ClockResume):
            offset, slope = offset + action.time * (slope - slope0), slope0
        elif isinstance(action, ClockPause):
            offset, slope = offset + action.time * slope, 0
        elif isinstance(action, ClockSkip):
            slices_map.append(
                (
                    slice(action.time, action.time),
                    slice(
                        offset + action.time * slope,
                        offset + action.time * slope + action.delta,
                    ),
                )
            )
            offset += action.delta
        else:
            raise TypeError


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
    def __init__(
        self,
        effects_scheduler,
        action_queue,
        samplerate,
        buffer_length,
        nchannels,
        monitor,
    ):
        self.effects_scheduler = effects_scheduler
        self.action_queue = action_queue
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.nchannels = nchannels
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, action_queue, settings, manager, init_time, monitor):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device

        output_node = Mixer._mix_node(scheduler, action_queue, settings, init_time)
        if monitor:
            output_node = monitor.monitoring(output_node)

        return aud.play(
            manager,
            output_node,
            samplerate=samplerate,
            buffer_shape=(buffer_length, nchannels),
            format=format,
            device=device,
        )

    @staticmethod
    @dn.datanode
    def _mix_node(scheduler, action_queue, settings, init_time):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        clock = Mixer._clock_node(
            action_queue, init_time, buffer_length, samplerate, settings
        )

        with clock, scheduler:
            yield
            while True:
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                try:
                    slices_map = clock.send(None)
                    data = scheduler.send((data, slices_map))
                except StopIteration:
                    return
                yield data

    @staticmethod
    def _clock_node(action_queue, init_time, buffer_length, samplerate, settings):
        return dn.pipe(
            dn.count(0, buffer_length),
            lambda time: slice(time, time + buffer_length),
            clock_slice(
                settings.sound_delay + init_time,
                1 / samplerate,
                action_queue,
            ),
        )

    @classmethod
    def create(cls, settings, manager, init_time=0.0, monitor=None):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        scheduler = dn.Scheduler()
        action_queue = queue.Queue()
        task = cls.get_task(
            scheduler, action_queue, settings, manager, init_time, monitor
        )
        return task, cls(
            scheduler, action_queue, samplerate, buffer_length, nchannels, monitor
        )

    def pause(self, when):
        self.action_queue.put(ClockPause(round(when * self.samplerate)))

    def resume(self, when):
        self.action_queue.put(ClockResume(round(when * self.samplerate)))

    def skip(self, when, delta):
        self.action_queue.put(
            ClockSkip(round(when * self.samplerate), round(delta * self.samplerate))
        )

    def add_effect(self, node, zindex=(0,)):
        return self.effects_scheduler.add_node(node, zindex=zindex)

    def remove_effect(self, key):
        return self.effects_scheduler.remove_node(key)

    @dn.datanode
    def tmask(self, node, time):
        node = dn.DataNode.wrap(node)

        samplerate = self.samplerate
        buffer_length = self.buffer_length
        nchannels = self.nchannels

        with node:
            data, slices_map = yield
            if time is None:
                time = slices_map[0][1].start

            while True:
                data_offset = slices_map[0][0].start
                for data_slice, time_slice in slices_map:
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
                            node.send((dummy, time))
                        except StopIteration:
                            return
                        offset += length
                        time += length / samplerate

                    # overrun
                    if data_slice.stop - data_slice.start <= offset:
                        continue

                    try:
                        slic = slice(
                            data_slice.start + offset - data_offset,
                            data_slice.stop - data_offset,
                        )
                        data[slic] = node.send(data[slic])
                    except StopIteration:
                        return

                    time = time_slice.stop

                data, slices_map = yield data.copy()

    def resample(
        self, samplerate=None, channels=None, volume=0.0, start=None, end=None
    ):
        pipeline = []
        if start is not None or end is not None:
            pipeline.append(dn.tspan(samplerate or self.samplerate, start, end))
        if channels is not None and channels != self.nchannels:
            pipeline.append(dn.rechannel(self.nchannels, channels))
        if samplerate is not None and samplerate != self.samplerate:
            pipeline.append(dn.resample(ratio=(self.samplerate, samplerate)))
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
        # node = dn.pipe(lambda a: a[0], dn.attach(node))
        # if time is not None:
        #     node = self.tmask(node, time)
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
    def __init__(self, listeners_scheduler, action_queue, monitor):
        self.listeners_scheduler = listeners_scheduler
        self.action_queue = action_queue
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, action_queue, settings, manager, init_time, monitor):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        time_res = settings.detect.time_res
        hop_length = round(samplerate * time_res)

        input_node = Detector._detect_node(scheduler, action_queue, init_time, settings)
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        if monitor:
            input_node = monitor.monitoring(input_node)

        return aud.record(
            manager,
            input_node,
            samplerate=samplerate,
            buffer_shape=(buffer_length, nchannels),
            format=format,
            device=device,
        )

    @staticmethod
    @dn.datanode
    def _detect_node(scheduler, action_queue, init_time, settings):
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

        clock = Detector._clock_node(
            action_queue, init_time, hop_length, samplerate, prepare, settings
        )

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

        with scheduler, clock, onset, picker:
            data = yield
            while True:
                try:
                    strength = onset.send(data)
                    detected, strength = picker.send(strength)
                    tick = clock.send(None)
                    normalized_strength = strength / settings.knock_energy
                    scheduler.send((None, tick, normalized_strength, detected))
                except StopIteration:
                    return
                data = yield

    @staticmethod
    def _clock_node(action_queue, init_time, hop_length, samplerate, prepare, settings):
        return dn.pipe(
            dn.count(-prepare * hop_length / samplerate, hop_length / samplerate),
            clock(
                settings.knock_delay + init_time,
                1,
                action_queue,
            ),
        )

    @classmethod
    def create(cls, settings, manager, init_time=0.0, monitor=None):
        scheduler = dn.Scheduler()
        action_queue = queue.Queue()
        task = cls.get_task(
            scheduler, action_queue, settings, manager, init_time, monitor
        )
        return task, cls(scheduler, action_queue, monitor)

    def pause(self, when):
        self.action_queue.put(ClockPause(when))

    def resume(self, when):
        self.action_queue.put(ClockResume(when))

    def skip(self, when, delta):
        self.action_queue.put(ClockSkip(when, delta))

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
        self.rich_renderer = mu.RichBarRenderer(terminal_settings.unicode_version)

    def add_markup(self, markup, mask=slice(None, None), shift=0):
        self.markups.append((markup, mask, shift))

    def draw(self, width):
        buffer = [" "] * width
        xran = range(width)

        for markup, mask, shift in self.markups:
            if mask.start is None:
                x = shift
            elif mask.start >= 0:
                x = shift + mask.start
            else:
                x = shift + mask.start + width

            self.rich_renderer._render(
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
    def __init__(self, drawers_scheduler, action_queue, monitor):
        self.drawers_scheduler = drawers_scheduler
        self.action_queue = action_queue
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, action_queue, settings, term_settings, init_time, monitor):
        framerate = settings.display_framerate

        display_node = Renderer._resize_node(
            Renderer._render_node(
                scheduler, action_queue, settings, term_settings, init_time
            ),
            settings,
            term_settings,
        )
        if monitor:
            display_node = monitor.monitoring(display_node)
        return term.show(display_node, 1 / framerate, hide_cursor=True)

    @staticmethod
    @dn.datanode
    def _render_node(scheduler, action_queue, settings, term_settings, init_time):
        framerate = settings.display_framerate

        rich_renderer = mu.RichTextRenderer(term_settings.unicode_version)
        clear_line = rich_renderer.render(rich_renderer.clear_line().expand())
        clear_below = rich_renderer.render(rich_renderer.clear_below().expand())
        width = 0
        logs = []
        msgs = []
        curr_msgs = list(msgs)

        clock = Renderer._clock_node(action_queue, init_time, framerate, settings)

        with clock, scheduler:
            shown, resized, size = yield
            while True:
                width = size.columns
                view = RichBar(term_settings)
                try:
                    tick = clock.send(None)
                    view, msgs, logs = scheduler.send(((view, msgs, logs), tick, width))
                except StopIteration:
                    return
                view_str = view.draw(width)

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
        rich_renderer = mu.RichTextRenderer(term_settings.unicode_version)
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
                if (
                    resized
                    and time.perf_counter() < resizing_since + settings.resize_delay
                ):
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

    @staticmethod
    def _clock_node(action_queue, init_time, framerate, settings):
        return dn.pipe(
            dn.count(1 / framerate, 1 / framerate),
            clock(
                settings.display_delay + init_time,
                1,
                action_queue,
            ),
        )

    @classmethod
    def create(cls, settings, term_settings, init_time=0.0, monitor=None):
        scheduler = dn.Scheduler()
        action_queue = queue.Queue()
        task = cls.get_task(
            scheduler, action_queue, settings, term_settings, init_time, monitor
        )
        return task, cls(scheduler, action_queue, monitor)

    def pause(self, when):
        self.action_queue.put(ClockPause(when))

    def resume(self, when):
        self.action_queue.put(ClockResume(when))

    def skip(self, when, delta):
        self.action_queue.put(ClockSkip(when, delta))

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    def add_log(self, msg, zindex=(0,)):
        return self.add_drawer(self._log_drawer(msg), zindex)

    def add_message(self, msg, zindex=(0,)):
        return self.add_drawer(self._msg_drawer(msg), zindex)

    def add_text(self, text_node, xmask=slice(None, None), zindex=(0,)):
        return self.add_drawer(self._text_drawer(text_node, xmask), zindex)

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
    def _text_drawer(text_node, xmask=slice(None, None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            (view, msg, logs), time, width = yield
            while True:
                xran = to_range(xmask.start, xmask.stop, width)

                try:
                    res = text_node.send((time, xran))
                except StopIteration:
                    return

                if res is not None:
                    xshift, text = res
                    view.add_markup(text, xmask, xshift)

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
    def __init__(self, handlers_scheduler, action_queue):
        self.handlers_scheduler = handlers_scheduler
        self.action_queue = action_queue

    @staticmethod
    def get_task(scheduler, action_queue, settings, term_settings, init_time):
        return term.inkey(
            Controller._control_node(
                scheduler, action_queue, settings, term_settings, init_time
            ),
            dt=settings.update_interval,
        )

    @staticmethod
    @dn.datanode
    def _control_node(scheduler, action_queue, settings, term_settings, init_time):
        keycodes = term_settings.keycodes

        clock = Controller._clock_node(
            action_queue, init_time, settings.update_interval
        )

        with clock, scheduler:
            while True:
                _, keycode = yield

                if keycode is None:
                    keyname = None
                elif keycode in keycodes:
                    keyname = keycodes[keycode]
                elif keycode.isprintable():
                    keyname = "PRINTABLE"
                else:
                    keyname = repr(keycode)

                try:
                    tick = clock.send(None)
                    scheduler.send((None, tick, keyname, keycode))
                except StopIteration:
                    return

    @staticmethod
    def _clock_node(action_queue, init_time, update_interval):
        return dn.pipe(
            dn.time(0.0),
            clock(init_time, 1, action_queue),
        )

    @classmethod
    def create(cls, settings, term_settings, init_time=0.0):
        scheduler = dn.Scheduler()
        action_queue = queue.Queue()
        task = cls.get_task(scheduler, action_queue, settings, term_settings, init_time)
        return task, cls(scheduler, action_queue)

    def pause(self, when):
        self.action_queue.put(ClockPause(when))

    def resume(self, when):
        self.action_queue.put(ClockResume(when))

    def skip(self, when, delta):
        self.action_queue.put(ClockSkip(when, delta))

    def add_handler(self, node, keyname=None):
        return self.handlers_scheduler.add_node(self._filter_node(node, keyname), (0,))

    def remove_handler(self, key):
        self.handlers_scheduler.remove_node(key)

    @dn.datanode
    def _filter_node(self, node, name):
        node = dn.DataNode.wrap(node)
        with node:
            while True:
                _, tick, keyname, keycode = yield
                if keycode is None:
                    continue
                if name is None or name == keyname:
                    try:
                        node.send((None, tick, keyname, keycode))
                    except StopIteration:
                        return


@dataclasses.dataclass
class Metronome:
    offset: float
    tempo: float

    def time(self, beat):
        r"""Convert beat to time (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float

        Returns
        -------
        time : float
        """
        return self.offset + beat * 60 / self.tempo

    def beat(self, time):
        r"""Convert time (in seconds) to beat.

        Parameters
        ----------
        time : float

        Returns
        -------
        beat : float
        """
        return (time - self.offset) * self.tempo / 60

    def dtime(self, beat, length):
        r"""Convert length to time difference (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float
        length : int or Fraction or float

        Returns
        -------
        dtime : float
        """
        return self.time(beat + length) - self.time(beat)


class DevicesSettings(cfg.Configurable):
    mixer = cfg.subconfig(MixerSettings)
    detector = cfg.subconfig(DetectorSettings)
    renderer = cfg.subconfig(RendererSettings)
    controller = cfg.subconfig(ControllerSettings)
    terminal = cfg.subconfig(term.TerminalSettings)
    logger = cfg.subconfig(log.LoggerSettings)
