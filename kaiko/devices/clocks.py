import dataclasses
import threading
import queue
from ..utils import datanodes as dn


class ClockOperation:
    pass


@dataclasses.dataclass(frozen=True)
class ClockSpeed(ClockOperation):
    time: float
    ratio: float


@dataclasses.dataclass(frozen=True)
class ClockSkip(ClockOperation):
    time: float
    offset: float


@dataclasses.dataclass(frozen=True)
class ClockDelay(ClockOperation):
    time: float
    delay: float


@dataclasses.dataclass(frozen=True)
class ClockStop(ClockOperation):
    time: float


class Clock:
    def __init__(self, offset, ratio):
        self.offset = offset
        self.ratio = ratio
        self.action_queues = {}
        self.lock = threading.Lock()

    @staticmethod
    @dn.datanode
    def _clock(action_queue, offset, ratio, delay=0.0):
        action = None

        time = yield
        last_time = time
        last_tick = offset + last_time * ratio
        last_ratio = ratio
        while True:
            # find unhandled action
            if action is None and not action_queue.empty():
                action = action_queue.get()

            # update last_time, last_tick, last_ratio
            curr_time = time if action is None else min(action.time - delay, time)
            if last_time < curr_time:
                time_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)
                last_tick = max(last_tick, time_slice.start)
                last_ratio = 0 if last_tick > time_slice.stop else ratio
                last_tick = max(last_tick, time_slice.stop)
                last_time = curr_time

            # update offset, ratio
            if action is not None and (action_time := action.time - delay) <= time:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockSpeed):
                    offset, ratio = offset + action_time * (ratio - action.ratio), action.ratio
                elif isinstance(action, ClockSkip):
                    offset += action.offset
                elif isinstance(action, ClockDelay):
                    offset += action.delay * ratio
                    delay += action.delay
                else:
                    raise TypeError

                action = None
                continue

            time = yield (last_tick, last_ratio)

    @staticmethod
    @dn.datanode
    def _clock_slice(action_queue, offset, ratio, delay=0.0):
        action = None

        time_slice = yield
        slices_map = []
        last_ratio = ratio
        last_time = time_slice.start
        last_tick = offset + last_time * ratio
        while True:
            # find unhandled action
            if action is None and not action_queue.empty():
                action = action_queue.get()

            # update last_time, last_tick, last_ratio
            curr_time = time_slice.stop if action is None else min(action.time - delay, time_slice.stop)
            if last_time < curr_time:
                tick_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)

                if last_tick < tick_slice.start:
                    slices_map.append(
                        (
                            slice(last_time, last_time),
                            slice(last_tick, tick_slice.start),
                            float("inf"),
                        )
                    )
                    last_tick = tick_slice.start

                if last_tick > tick_slice.start:
                    cut_time = (last_tick - offset) / ratio if ratio != 0 else curr_time
                    cut_time = min(max(cut_time, last_time), curr_time)
                    slices_map.append(
                        (slice(last_time, cut_time), slice(last_tick, last_tick), 0)
                    )
                    last_ratio = 0
                    last_time = cut_time
                    tick_slice = slice(last_tick, tick_slice.stop)

                if last_time < curr_time:
                    slices_map.append((slice(last_time, curr_time), tick_slice, ratio))
                    last_ratio = ratio
                    last_time = curr_time
                    last_tick = tick_slice.stop

            # update offset, ratio
            if action is not None and (action_time := action.time - delay) <= time_slice.stop:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockSpeed):
                    offset, ratio = offset + action_time * (ratio - action.ratio), action.ratio
                elif isinstance(action, ClockSkip):
                    offset += action.offset
                elif isinstance(action, ClockDelay):
                    offset += action.delay * ratio
                    delay += action.delay
                else:
                    raise TypeError

                action = None
                continue

            time_slice = yield slices_map
            slices_map = []

    def clock(self, name, delay=0.0):
        with self.lock:
            action_queue = queue.Queue()
            self.action_queues[name] = action_queue
            return self._clock(action_queue, self.offset, self.ratio, delay=delay)

    def clock_slice(self, name, delay=0.0):
        with self.lock:
            action_queue = queue.Queue()
            self.action_queues[name] = action_queue
            return self._clock_slice(action_queue, self.offset, self.ratio, delay=delay)

    def speed(self, time, ratio):
        with self.lock:
            for action_queue in self.action_queues.values():
                action_queue.put(ClockSpeed(time, ratio))
            self.offset, self.ratio = self.offset + time * (self.ratio - ratio), ratio

    def skip(self, time, offset):
        with self.lock:
            for action_queue in self.action_queues.values():
                action_queue.put(ClockSkip(time, offset))
            self.offset += offset

    def delay(self, name, time, delay):
        with self.lock:
            action_queue = self.action_queues[name]
            action_queue.put(ClockDelay(time, delay))

    def stop(self, time):
        with self.lock:
            for action_queue in self.action_queues.values():
                action_queue.put(ClockStop(time))


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


