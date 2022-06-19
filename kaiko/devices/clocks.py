import dataclasses
import queue
from ..utils import datanodes as dn


class ClockOperation:
    pass


@dataclasses.dataclass(frozen=True)
class ClockPause(ClockOperation):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockResume(ClockOperation):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockSkip(ClockOperation):
    time: float
    delta: float


@dataclasses.dataclass(frozen=True)
class ClockStop(ClockOperation):
    time: float


class Clock:
    def __init__(self):
        self.action_queue = queue.Queue()

    @dn.datanode
    def clock(self, offset, ratio):
        ratio0 = ratio

        action = None

        time = yield
        last_time = time
        last_tick = offset + last_time * ratio
        last_ratio = ratio
        while True:
            # find unhandled action
            if action is None and not self.action_queue.empty():
                action = self.action_queue.get()

            # update last_time, last_tick, last_ratio
            curr_time = time if action is None else min(action.time, time)
            if last_time < curr_time:
                time_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)
                last_tick = max(last_tick, time_slice.start)
                last_ratio = 0 if last_tick > time_slice.stop else ratio
                last_tick = max(last_tick, time_slice.stop)
                last_time = curr_time

            # update offset, ratio
            if action is not None and action.time <= time:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockResume):
                    offset, ratio = offset + action.time * (ratio - ratio0), ratio0
                elif isinstance(action, ClockPause):
                    offset, ratio = offset + action.time * ratio, 0
                elif isinstance(action, ClockSkip):
                    offset += action.delta
                else:
                    raise TypeError

                action = None
                continue

            time = yield (last_tick, last_ratio)

    @dn.datanode
    def clock_slice(self, offset, ratio):
        ratio0 = ratio

        action = None

        time_slice = yield
        slices_map = []
        last_ratio = ratio
        last_time = time_slice.start
        last_tick = offset + last_time * ratio
        while True:
            # find unhandled action
            if action is None and not self.action_queue.empty():
                action = self.action_queue.get()

            # update last_time, last_tick, last_ratio
            curr_time = time_slice.stop if action is None else min(action.time, time_slice.stop)
            if last_time < curr_time:
                tick_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)

                if last_tick < tick_slice.start:
                    slices_map.append(
                        (
                            slice(last_time, last_time),
                            slice(last_tick, tick_slice.start),
                            last_ratio,
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
            if action is not None and action.time <= time_slice.stop:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockResume):
                    offset, ratio = offset + action.time * (ratio - ratio0), ratio0
                elif isinstance(action, ClockPause):
                    offset, ratio = offset + action.time * ratio, 0
                elif isinstance(action, ClockSkip):
                    offset += action.delta
                else:
                    raise TypeError

                action = None
                continue

            time_slice = yield slices_map
            slices_map = []

    def resume(self, time):
        self.action_queue.put(ClockResume(time))

    def pause(self, time):
        self.action_queue.put(ClockPause(time))

    def skip(self, time, delta):
        self.action_queue.put(ClockSkip(time, delta))

    def stop(self, time):
        self.action_queue.put(ClockStop(time))


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


