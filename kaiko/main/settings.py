from ..utils import config as cfg
from ..devices import engines
from ..beats import beatshell
from ..beats import beatmaps


class KAIKOSettings(cfg.Configurable):
    devices = cfg.subconfig(engines.DevicesSettings)
    shell = cfg.subconfig(beatshell.BeatShellSettings)
    gameplay = cfg.subconfig(beatmaps.GameplaySettings)
