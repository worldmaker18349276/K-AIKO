from ..utils import config as cfg
from ..devices import engines
from ..tui import beatshell
from ..beats import beatmaps
from . import loggers
from . import files


class KAIKOSettings(cfg.Configurable):
    logger = cfg.subconfig(loggers.LoggerSettings)
    files = cfg.subconfig(files.FileManagerSettings)
    devices = cfg.subconfig(engines.DevicesSettings)
    shell = cfg.subconfig(beatshell.BeatShellSettings)
    gameplay = cfg.subconfig(beatmaps.GameplaySettings)
