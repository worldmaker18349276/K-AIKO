from ..utils import config as cfg
from ..devices import engines
from ..beats import beatmaps
from . import beatshell
from . import loggers
from . import files
from . import bgm


class KAIKOSettings(cfg.Configurable):
    logger = cfg.subconfig(loggers.LoggerSettings)
    files = cfg.subconfig(files.FileManagerSettings)
    devices = cfg.subconfig(engines.DevicesSettings)
    shell = cfg.subconfig(beatshell.BeatShellSettings)
    bgm = cfg.subconfig(bgm.BGMControllerSettings)
    gameplay = cfg.subconfig(beatmaps.GameplaySettings)
