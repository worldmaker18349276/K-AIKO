import sys
import traceback
import os
import zipfile
import contextlib
import psutil
import appdirs
from . import datanodes as dn
from . import cfg
from . import kerminal
from . import beatmenu


def print_logo():
    print("\n"
        "  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█\n"
        "  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █\n"
        "  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █\n"
        "  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█\n"
        "\n"
        "\n"
        "  🎧  Use headphones for the best experience 🎝 \n"
        "\n"
        , flush=True)

def print_pyaudio_info(manager):
    import pyaudio

    print()

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
def load_pyaudio(theme):
    print(f"{theme.info_icon} Loading PyAudio...")
    print()

    ctxt = kerminal.prepare_pyaudio()

    print(theme.verb[0], end="", flush=True)
    try:
        manager = ctxt.__enter__()
        print_pyaudio_info(manager)
    finally:
        print(theme.verb[1], flush=True)

    try:
        yield manager
    except:
        ctxt.__exit__(*sys.exc_info())
    else:
        ctxt.__exit__(None, None, None)

def main(theme=None):
    # load theme
    settings = beatmenu.KAIKOTheme()
    if theme is not None:
        cfg.config_read(open(theme, 'r'), main=settings)

    data_icon = settings.data_icon
    info_icon = settings.info_icon
    hint_icon = settings.hint_icon
    verb = settings.verb
    emph = settings.emph
    warn = settings.warn

    try:
        # print logo
        print_logo()

        # load PyAudio
        with load_pyaudio(settings) as manager:

            # load user data
            user = beatmenu.BeatMenuUser(settings)
            user.load()

            game = beatmenu.BeatMenuGame(user)

            # play given beatmap
            if len(sys.argv) > 1:
                filepath = sys.argv[1]
                game.play(filepath).execute(manager)
                return

            # load songs
            game.reload()

            def play_menu():
                songs = []
                for filepath in game.beatmaps:
                    songs.append((os.path.basename(filepath), game.play(filepath)))
                return beatmenu.menu_tree(songs)

            menu = beatmenu.menu_tree([
                ("play", play_menu),
                ("settings", None),
            ])

            # enter menu
            print(f"{hint_icon} Use {'up'.join(emph)}/{'down'.join(emph)}/"
                  f"{'enter'.join(emph)}/{'esc'.join(emph)} keys to select options.")
            print(flush=True)

            with menu:
                while True:
                    result = beatmenu.explore(menu)
                    if hasattr(result, 'execute'):
                        result.execute(manager)
                    elif result is None:
                        break

    except KeyboardInterrupt:
        pass

    except:
        # print error
        print(warn[0], end="")
        traceback.print_exc(file=sys.stdout)
        print(warn[1], end="")

if __name__ == '__main__':
    main()
