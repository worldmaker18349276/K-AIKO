import os
import dataclasses
import traceback
import getpass
import glob
import re
import shutil
from pathlib import Path
from ..utils import config as cfg
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from .loggers import Logger


class InvalidFileOperation(Exception):
    pass


class WildCardDescriptor:
    def __init__(self, provider):
        """Constructor of file descriptor.

        Parameters
        ----------
        provider : utils.providers.Provider
        """
        self.provider = provider

    def desc(self, path):
        return type(self).__doc__

    def info(self, path):
        return None

    def mk(self, path):
        raise InvalidFileOperation

    def rm(self, path):
        raise InvalidFileOperation

    def show(self, indent=0):
        return "[any] " + (type(self).__doc__ or "")


class FileDescriptor(WildCardDescriptor):
    def show(self, indent=0):
        return "[file] " + (type(self).__doc__ or "")


class DirDescriptor(WildCardDescriptor):
    def __init__(self, provider):
        super().__init__(provider)

        for name, child in type(self).__dict__.items():
            if isinstance(child, DirChildField):
                self.__dict__[name] = DirChild(
                    child.pattern,
                    child.is_required,
                    child.descriptor_type(provider),
                )

    def children(self):
        for child in self.__dict__.values():
            if isinstance(child, DirChild):
                yield child

    def show(self, indent=0):
        return "[dir] " + (type(cls).__doc__ or "") + "".join(
            "\n" + "  " * (indent+1) + child.pattern + ": " + child.descriptor.show(indent+1)
            for child in self.children()
        )


@dataclasses.dataclass(frozen=True)
class DirChild:
    pattern: str
    is_required: bool
    descriptor: WildCardDescriptor


@dataclasses.dataclass(frozen=True)
class DirChildField:
    pattern: str
    is_required: bool
    descriptor_type: type


def as_child(pattern, is_required=False):
    if is_required:
        if not re.fullmatch(r"([^/?*[]|\[\?]|\[\*]|\[\[])*", pattern):
            raise ValueError(f"invalid pattern: {pattern}")

        def child_decorator(descriptor_type):
            if not issubclass(descriptor_type, (FileDescriptor, DirDescriptor)):
                raise TypeError(f"invalid file descriptor: {descriptor_type}")
            return DirChildField(pattern, is_required, descriptor_type)

        return child_decorator

    else:
        if not re.fullmatch(r"[^/]*", pattern):
            raise ValueError(f"invalid pattern: {pattern}")

        def child_decorator(descriptor_type):
            if not issubclass(descriptor_type, WildCardDescriptor):
                raise TypeError(f"invalid file descriptor: {descriptor_type}")
            return DirChildField(pattern, is_required, descriptor_type)

        return child_decorator


def unescape_glob(pattern):
    if not re.fullmatch(r"([^/?*[]|\[\?]|\[\*]|\[\[])*", pattern):
        raise ValueError(f"invalid pattern: {pattern}")
    return pattern.replace("[?]", "?").replace("[*]", "*").replace("[[]", "[")


class FileManagerSettings(cfg.Configurable):
    @cfg.subconfig
    class display(cfg.Configurable):
        r"""
        Fields
        ------
        file_item : str
            The template for file item.
        file_unknown : str
            The template for unknown file.
        file_desc : str
            The template for file description.

        file_dir : str
            The template for directory.
        file_script : str
            The template for script file.
        file_beatmap : str
            The template for beatmap file.
        file_sound : str
            The template for audio file.
        file_link : str
            The template for link file.
        file_normal : str
            The template for normal file.
        file_other : str
            The template for other file.
        """
        file_item: str = "• [slot/]"
        file_unknown: str = "[weight=dim][slot/][/]"
        file_desc: str = "  [weight=dim][slot/][/]"

        file_dir: str = "[weight=bold][color=blue][slot/][/][/]/"
        file_script: str = "[weight=bold][color=green][slot/][/][/]"
        file_beatmap: str = "[weight=bold][color=magenta][slot/][/][/]"
        file_sound: str = "[color=magenta][slot/][/]"
        file_link: str = "[color=cyan][slot=src/][/] -> [slot=dst/]"
        file_normal: str = "[slot/]"
        file_other: str = "[slot/]"


@dataclasses.dataclass
class FileManager:
    username: str
    root: Path
    current: Path
    structure: DirDescriptor
    settings: FileManagerSettings

    @classmethod
    def create(cls, structure):
        username = getpass.getuser()
        root = Path("~/.local/share/K-AIKO").expanduser()
        settings = FileManagerSettings()
        return cls(username, root, Path("."), structure, settings)

    def set_settings(self, settings):
        self.settings = settings

    def check_is_prepared(self, logger):
        def go(path, tree):
            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)

                    if isinstance(child.descriptor, DirDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] There is a missing directory [emph]{subpath!s}[/].")
                            return False
                        if not subpath.is_dir():
                            raise ValueError(f"bad file structure: {subpath!s} should be a directory")
                        if not go(subpath, child.descriptor):
                            return False
                    elif isinstance(child.descriptor, FileDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] There is a missing file [emph]{subpath!s}[/].")
                            return False
                        if not subpath.is_file():
                            raise ValueError(f"bad file structure: {subpath!s} should be a file")
                    else:
                        raise TypeError(child.descriptor)
            return True

        if not self.root.exists():
            logger.print(f"[data/] The workspace [emph]{self.root!s}[/] is missing.")
            return False

        return go(self.root, self.structure)

    def prepare(self, logger):
        if self.check_is_prepared(logger):
            return

        # start up
        logger.print("[data/] Prepare your profile...")

        def go(path, tree):
            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)

                    if isinstance(child.descriptor, DirDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] Create directory [emph]{subpath!s}[/]...")
                            subpath.mkdir()
                        go(subpath, child.descriptor)
                    elif isinstance(child.descriptor, FileDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] Create file [emph]{subpath!s}[/]...")
                            subpath.touch()
                    else:
                        raise TypeError(child.descriptor)

        if not self.root.exists():
            self.root.mkdir()
            logger.print(
                f"[data/] Your data will be stored in {logger.emph(self.root.as_uri())}"
            )

        go(self.root, self.structure)

        logger.print(flush=True)

    def remove(self, logger):
        shutil.rmtree(str(self.root))

    def glob(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = path.resolve()
        except Exception:
            return (), None, None

        if not abspath.is_relative_to(self.root):
            return (), None, None

        route = [abspath, *abspath.parents]
        route = route[route.index(self.root)::-1][1:]

        index = ()
        descriptor = self.structure
        for current_path in route:
            if not isinstance(descriptor, DirDescriptor):
                return (), None, None

            for i, child in enumerate(descriptor.children()):
                if child.pattern == "**":
                    current_path = path

                if isinstance(child.descriptor, DirDescriptor):
                    if Path.exists(current_path) and not Path.is_dir(current_path):
                        continue
                elif isinstance(child.descriptor, FileDescriptor):
                    if Path.exists(current_path) and not Path.is_file(current_path):
                        continue
                elif isinstance(child.descriptor, WildCardDescriptor):
                    pass
                else:
                    raise TypeError

                if child.pattern == "**":
                    index = (*index, i)
                    return index, path, child.descriptor

                if current_path.match(child.pattern):
                    index = (*index, i)
                    descriptor = child.descriptor
                    break

            else:
                return (), None, None

        return index, path, descriptor

    def cd(self, path, logger):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = (self.root / self.current / path).resolve()
        except Exception:
            logger.print("[warn]Failed to resolve path[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

        if not abspath.is_relative_to(self.root):
            logger.print("[warn]Out of root directory[/]")
            return
        if not abspath.exists():
            logger.print("[warn]No such directory[/]")
            return
        if not abspath.is_dir():
            logger.print("[warn]Is not a directory[/]")
            return

        if not path.is_absolute():
            currpath = self.current / path
        elif path.is_relative_to(self.root):
            currpath = path.relative_to(self.root)
        else:
            currpath = abspath.relative_to(self.root)

        # don't resolve symlink
        self.current = Path(os.path.normpath(str(currpath)))

    def ls(self, logger):
        file_item = logger.rich.parse(self.settings.display.file_item, slotted=True)
        file_unknown = logger.rich.parse(self.settings.display.file_unknown, slotted=True)
        file_desc = logger.rich.parse(self.settings.display.file_desc, slotted=True)
        file_dir = logger.rich.parse(self.settings.display.file_dir, slotted=True)
        file_script = logger.rich.parse(self.settings.display.file_script, slotted=True)
        file_beatmap = logger.rich.parse(self.settings.display.file_beatmap, slotted=True)
        file_sound = logger.rich.parse(self.settings.display.file_sound, slotted=True)
        file_link = logger.rich.parse(self.settings.display.file_link, slotted=True)
        file_normal = logger.rich.parse(self.settings.display.file_normal, slotted=True)
        file_other = logger.rich.parse(self.settings.display.file_other, slotted=True)

        res = []
        for child in (self.root / self.current).resolve().iterdir():
            if child.is_symlink():
                name = logger.escape(str(child.readlink()), type="all")
            else:
                name = logger.escape(child.name, type="all")
            name = logger.rich.parse(name)

            if child.is_dir():
                name = file_dir(name)

            elif child.is_file():
                if child.suffix in [".py", ".kaiko-profile"]:
                    name = file_script(name)
                elif child.suffix in [".ka", ".kaiko", ".osu"]:
                    name = file_beatmap(name)
                elif child.suffix in [".wav", ".mp3", ".mp4", ".m4a", ".ogg"]:
                    name = file_sound(name)
                else:
                    name = file_normal(name)

            else:
                name = file_other(name)

            if child.is_symlink():
                linkname = logger.escape(child.name, type="all")
                linkname = logger.rich.parse(linkname)
                name = file_link(src=linkname, dst=name)

            ind, path, descriptor = self.glob(self.root / self.current / child.name)
            desc = descriptor.desc(path) if descriptor is not None else None
            desc = logger.rich.parse(desc, root_tag=True) if desc is not None else None

            ordering_key = (descriptor is None, ind, child.is_symlink(), not child.is_dir(), child.suffix, child.stem)

            if descriptor is None:
                name = file_unknown(name)
            desc = file_desc(desc) if desc is not None else mu.Text("")
            name = file_item(name)

            width = logger.rich.widthof(name)
            res.append((ordering_key, width, name, desc))

        res = sorted(res, key=lambda e: e[0])
        max_width = max((width for _, width, _, _ in res), default=0)

        for _, width, name, desc in res:
            padding = " "*(max_width - width) if width != -1 else " "
            logger.print(name, end=padding)
            logger.print(desc, end="\n")

    def mk(self, path, logger):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = (self.root / self.current / path).resolve()
        except Exception:
            logger.print("[warn]Failed to resolve path[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

        if not abspath.is_relative_to(self.root):
            logger.print("[warn]Out of root directory[/]")
            return
        if abspath.exists():
            logger.print("[warn]File already exists[/]")
            return

        ind, abspath, descriptor = self.glob(self.root / self.current / path)
        if descriptor is None:
            logger.print("[warn]Unknown file[/]")
            return

        try:
            descriptor.mk(abspath)
        except Exception:
            logger.print("[warn]Failed to make file[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def get(self, path, logger):
        try:
            abspath = (self.root / self.current / path).resolve()
        except Exception:
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        if not abspath.is_relative_to(self.root):
            logger.print("[warn]Out of root directory[/]")
            return
        if not abspath.exists():
            logger.print("[warn]No such file[/]")
            return
        if not abspath.is_file():
            logger.print("[warn]Is not a file[/]")
            return

        return abspath


class FilesCommand:
    def __init__(self, provider, profile_is_changed=False):
        self.provider = provider
        self.profile_is_changed = profile_is_changed

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    @cmd.function_command
    def cd(self, path):
        self.file_manager.cd(path, self.logger)

    @cmd.function_command
    def ls(self):
        self.file_manager.ls(self.logger)

    @cmd.function_command
    def cat(self, path):
        file_manager = self.file_manager
        logger = self.logger

        abspath = file_manager.get(path, logger)

        try:
            content = abspath.read_text()
        except UnicodeDecodeError:
            logger.print("[warn]Cannot read binary file.[/]")
            return

        code = logger.format_code(
            content, title=str(file_manager.current / path)
        )
        logger.print(code)

    @cmd.function_command
    def mk(self, path):
        file_manager = self.file_manager
        logger = self.logger

        abspath = file_manager.mk(path, logger)

    @cd.arg_parser("path")
    def _cd_path_parser(self):
        return cmd.PathParser(self.file_manager.root / self.file_manager.current, type="dir")

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        return cmd.PathParser(self.file_manager.root / self.file_manager.current, type="file")

    @mk.arg_parser("path")
    def _mk_path_parser(self):
        return cmd.PathParser(self.file_manager.root / self.file_manager.current, type="file", should_exist=False)

    @cmd.function_command
    def clean(self, bottom: bool = False):
        """[rich]Clean screen.

        usage: [cmd]clean[/] [[[kw]--bottom[/] [arg]{BOTTOM}[/]]]
                                   ╲
                          bool, move to bottom or
                           not; default is False.
        """
        self.logger.clear(bottom)

    @cmd.function_command
    @dn.datanode
    def bye(self):
        """[rich]Close K-AIKO.

        usage: [cmd]bye[/]
        """
        if self.profile_is_changed:
            yes = yield from self.logger.ask(
                "Exit without saving current configuration?"
            ).join()
            if not yes:
                return
        self.logger.print("Bye~")
        raise KeyboardInterrupt

    @cmd.function_command
    @dn.datanode
    def bye_forever(self):
        """[rich]Clean up all your data and close K-AIKO.

        usage: [cmd]bye_forever[/]
        """
        logger = self.logger

        logger.print("This command will clean up all your data.")

        yes = yield from logger.ask("Do you really want to do that?", False).join()
        if yes:
            self.file_manager.remove(logger)
            logger.print("Good luck~")
            raise KeyboardInterrupt


def CdCommand(provider):
    def make_command(name):
        return cmd.function_command(
            lambda self: self.provider.get(FileManager).cd(name, self.provider.get(Logger))
        )

    attrs = {}
    attrs["provider"] = provider
    attrs[".."] = make_command("..")
    file_manager = provider.get(FileManager)
    for child in (file_manager.root / file_manager.current).resolve().iterdir():
        if child.is_dir():
            attrs[child.name + "/"] = make_command(child.name)

    return type("CdCommand", (object,), attrs)()
