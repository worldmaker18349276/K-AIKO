import time
import itertools
import contextlib
import signal
import numpy
import pyaudio
import cfg
import realtime_analysis as ra


@cfg.configurable
class KnockConsole:
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
    display_fps: float = 60.0
    display_delay: float = 0.03
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3

    def __init__(self, config=None):
        self.closed = False
        if config is not None:
            cfg.config_read(open(config, "r"), main=self)

    def close(self):
        self.closed = True

    def SIGINT_handler(self, sig, frame):
        self.close()

    @ra.DataNode.from_generator
    def get_output_node(self, knock_program):
        samplerate = self.output_samplerate
        buffer_length = self.output_buffer_length
        channels = self.output_channels

        mixer = ra.AudioMixer(samplerate, (buffer_length, channels))
        sound_handler = knock_program.get_sound_handler(mixer)

        with contextlib.closing(self), mixer, sound_handler:
            yield
            while True:
                sound_handler.send(mixer.time)
                yield mixer.send()

    @ra.DataNode.from_generator
    def get_input_node(self, knock_program):
        samplerate = self.input_samplerate
        buffer_length = self.input_buffer_length
        channels = self.input_channels

        time_res = self.detector_time_res
        hop_length = round(samplerate*time_res)
        freq_res = self.detector_freq_res
        win_length = round(samplerate/freq_res)
        pre_max = self.detector_pre_max
        post_max = self.detector_post_max
        pre_avg = self.detector_pre_avg
        post_avg = self.detector_post_avg
        wait = self.detector_wait
        delta = self.detector_delta

        pre_max = round(pre_max / time_res)
        post_max = round(post_max / time_res)
        pre_avg = round(pre_avg / time_res)
        post_avg = round(post_avg / time_res)
        wait = round(wait / time_res)
        delay = max(post_max, post_avg)

        knock_delay = self.knock_delay
        knock_energy = self.knock_energy

        knock_handler = knock_program.get_knock_handler()

        window = ra.get_half_Hann_window(win_length)
        detector = ra.pipe(ra.frame(win_length=win_length, hop_length=hop_length),
                           ra.power_spectrum(win_length=win_length,
                                             samplerate=samplerate,
                                             windowing=window,
                                             weighting=True),
                           ra.onset_strength(1),
                           (lambda a: (None, a, a)),
                           ra.pair(itertools.count(-delay), # generate index
                                   ra.delay([0.0]*delay), # delay signal
                                   ra.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta) # pick peak
                                   ),
                           (lambda a: (a[0]*time_res-knock_delay,
                                       a[1]/knock_energy,
                                       a[2])),
                           knock_handler)

        if buffer_length != hop_length:
            detector = ra.unchunk(detector, chunk_shape=(hop_length, channels))

        with contextlib.closing(self), detector:
            while True:
                detector.send((yield))

    @ra.DataNode.from_generator
    def get_view_node(self, knock_program):
        view_handler = knock_program.get_view_handler()
        display_delay = self.display_delay
        display_fps = self.display_fps
        dt = 1/display_fps

        try:
            with contextlib.closing(self), view_handler:
                t0 = time.time()
                t1 = dt

                yield
                while not self.closed:
                    t = time.time() - t0

                    if t < t1 - 0.001:
                        signal.signal(signal.SIGINT, self.SIGINT_handler)
                        time.sleep(t1 - t)
                        continue

                    view_handler.send(t - display_delay)

                    t = time.time() - t0
                    t1 += dt
                    while t > t1:
                        print("underrun")
                        t1 += dt

                    yield

        finally:
            print()

    def run(self, knock_program):
        input_params = dict(samplerate=self.input_samplerate,
                            buffer_shape=(self.input_buffer_length,
                                          self.input_channels),
                            format=self.input_format,
                            device=self.input_device
                            )
        output_params = dict(samplerate=self.output_samplerate,
                             buffer_shape=(self.output_buffer_length,
                                           self.output_channels),
                             format=self.output_format,
                             device=self.output_device
                             )

        try:
            manager = pyaudio.PyAudio()

            with contextlib.closing(self), knock_program:
                output_node = self.get_output_node(knock_program)
                input_node = self.get_input_node(knock_program)
                view_node = self.get_view_node(knock_program)

                with ra.record(manager, input_node, **input_params) as input_stream,\
                     ra.play(manager, output_node, **output_params) as output_stream,\
                     view_node, contextlib.suppress(StopIteration):

                    input_stream.start_stream()
                    output_stream.start_stream()

                    while True:
                        view_node.send()

        finally:
            manager.terminate()


class KnockProgram:
    # def get_sound_handler(self, mixer): time -> None
    # def get_knock_handler(self): (time, strength, detected) -> None
    # def get_view_handler(self): time -> None
    # def pause(self)
    # def resume(self)
    # def __enter__(self)
    # def __exit__(self, type=None, value=None, traceback=None)
    pass


def test_speaker(manager, samplerate=44100, buffer_length=1024, channels=1, format="f4", device=-1):
    buffer_shape = (buffer_length, channels)
    duration = 2.0+0.5*4*channels

    mixer = ra.AudioMixer(samplerate=samplerate, buffer_shape=buffer_shape)
    click = ra.pulse(samplerate=samplerate)
    for n in range(channels):
        for m in range(4):
            mixer.play([click], samplerate=samplerate, delay=1.0+0.5*(4*n+m))

    print("testing...")
    with ra.play(manager, mixer, samplerate=samplerate,
                                 buffer_shape=buffer_shape,
                                 format=format, device=device) as output_stream:
        output_stream.start_stream()
        time.sleep(duration)
    print("finish!")

def test_mic(manager, samplerate=44100, buffer_length=1024, channels=1, format="f4", device=-1):
    duration = 8.0

    spec_width = 5
    win_length = 512*4
    decay_time = 0.01
    Dt = buffer_length / samplerate
    spec = ra.pipe(ra.frame(win_length, buffer_length),
                   ra.power_spectrum(win_length, samplerate=samplerate),
                   ra.draw_spectrum(spec_width, win_length=win_length, samplerate=samplerate, decay=Dt/decay_time),
                   lambda s: print(" "+s+"\r", end="", flush=True))

    print("testing...")
    with ra.record(manager, spec, samplerate=samplerate,
                                  buffer_shape=(buffer_length, channels),
                                  format=format, device=device) as input_stream:
        input_stream.start_stream()
        time.sleep(duration)
    print()
    print("finish!")

def input_with_default(hint, default, type=None):
    default_str = str(default)
    value = input(hint + default_str + "\b"*len(default_str))
    if value:
        return type(value) if type is not None else value
    else:
        return default

def configure_audio(config_name=None):
    config = configparser.ConfigParser()
    config.read("default.kconfig")
    if isinstance(config_name, str):
        config.read(config_name)
    elif isinstance(config_name, (dict, configparser.ConfigParser)):
        config.read_dict(config_name)
    elif config_name is None:
        pass
    else:
        raise ValueError("invalid configuration", config_name)

    try:
        manager = pyaudio.PyAudio()

        print()

        print("portaudio version:")
        print("  " + pyaudio.get_portaudio_version_text())

        print("available devices:")
        apis_list = [manager.get_host_api_info_by_index(i)["name"] for i in range(manager.get_host_api_count())]
        for index in range(manager.get_device_count()):
            info = manager.get_device_info_by_index(index)

            name = info["name"]
            api = apis_list[info["hostApi"]]
            freq = info["defaultSampleRate"]/1000
            ch_in = info["maxInputChannels"]
            ch_out = info["maxOutputChannels"]

            print("  {}. {} by {} ({} kHz, in: {}, out: {})".format(index, name, api, freq, ch_in, ch_out))

        default_input_device_index = manager.get_default_input_device_info()["index"]
        default_output_device_index = manager.get_default_output_device_info()["index"]
        print("default input device: {}".format(default_input_device_index))
        print("default output device: {}".format(default_output_device_index))

        print()
        print("[output]")
        samplerate = input_with_default("samplerate = ", config.getint("output", "samplerate"), int)
        buffer_length = input_with_default("buffer_length = ", config.getint("output", "buffer_length"), int)
        channels = input_with_default("channels = ", config.getint("output", "channels"), int)
        format = input_with_default("format = ", config.get("output", "format"))
        device = input_with_default("device = ", config.getint("output", "device"), int)
        test_speaker(manager, samplerate=samplerate,
                              buffer_length=buffer_length,
                              channels=channels,
                              format=format, device=device)

        print()
        print("[input]")
        samplerate = input_with_default("samplerate = ", config.getint("input", "samplerate"), int)
        buffer_length = input_with_default("buffer_length = ", config.getint("input", "buffer_length"), int)
        channels = input_with_default("channels = ", config.getint("input", "channels"), int)
        format = input_with_default("format = ", config.get("input", "format"))
        device = input_with_default("device = ", config.getint("input", "device"), int)
        test_mic(manager, samplerate=samplerate,
                          buffer_length=buffer_length,
                          channels=channels,
                          format=format, device=device)

    finally:
        manager.terminate()

