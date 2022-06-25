from ..utils import config as cfg
from ..beats import beatmaps
from . import devices
from . import beatshell
from . import loggers
from . import files
from . import bgm


class KAIKOSettings(cfg.Configurable):
    logger = cfg.subconfig(loggers.LoggerSettings)
    files = cfg.subconfig(files.FileManagerSettings)
    devices = cfg.subconfig(devices.DevicesSettings)
    shell = cfg.subconfig(beatshell.BeatShellSettings)
    bgm = cfg.subconfig(bgm.BGMControllerSettings)
    gameplay = cfg.subconfig(beatmaps.GameplaySettings)
