import os
import dataclasses
import traceback
import getpass
import re
from typing import Optional
import shutil
from pathlib import Path
from ..utils import config as cfg
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from .loggers import Logger


class InvalidFileOperation(Exception):
    pass


# if not subpath.exists():
#     logger.print(f"[data/] There is a missing directory [emph]{subpath!s}[/].")
#     res = False
# if not subpath.is_dir() or subpath.is_symlink():
#     logger.print(f"[data/] Bad file structure: [emph]{subpath!s}[/] should be a directory.")
#     res = False


class MissingFileException(Exception):
    def __init__(self, path):
        self.path = path


class WrongFileTypeException(Exception):
    def __init__(self, path, excepted):
        self.path = path
        self.excepted = excepted


@dataclasses.dataclass(frozen=True)
class RecognizedPath:
    abs: Path
    slashend: bool

    def exists(self):
        return self.abs.exists()

    def is_dir(self):
        return self.abs.is_dir()

    def is_file(self):
        return self.abs.is_file()

    def is_symlink(self):
        return self.abs.is_symlink()

    def validate(self):
        return

    def __str__(self):
        abspath = str(self.abs)
        if self.slashend:
            abspath = os.path.join(abspath, "")
        return abspath

    def desc(self, provider):
        return type(self).__doc__

    def info(self, provider):
        return None

    def mk(self, provider):
        raise InvalidFileOperation

    def rm(self, provider):
        raise InvalidFileOperation

    def mv(self, dst, provider):
        raise InvalidFileOperation

    def cp(self, dst, provider):
        raise InvalidFileOperation


class UnrecognizedPath(RecognizedPath):
    def mk(self, provider):
        if self.slashend:
            self.abs.mkdir(exist_ok=False)
        else:
            self.abs.touch(exist_ok=False)

    def rm(self, provider):
        if self.abs.is_dir() and not self.abs.is_symlink():
            self.abs.rmdir()
        else:
            self.abs.unlink()

    def mv(self, dst, provider):
        self.abs.rename(dst.abs)

    def cp(self, dst, provider):
        shutil.copy(self.abs, dst.abs)


class RecognizedWildCardPath(RecognizedPath):
    pass


class RecognizedFilePath(RecognizedPath):
    def validate(self):
        if not self.exists():
            raise MissingFileException(self.abs)
        if self.is_symlink() or not self.is_file():
            raise WrongFileTypeException(self.abs, "a file")


class RecognizedDirPath(RecognizedPath):
    def validate(self):
        if not self.exists():
            raise MissingFileException(self.abs)
        if self.is_symlink() or not self.is_dir():
            raise WrongFileTypeException(self.abs, "a directory")

    @classmethod
    def get_fields(cls):
        for field in cls.__dict__.values():
            if isinstance(field, (DirWildcardField, DirPatternField, DirChildField)):
                yield field

    def get_children(self):
        for name, field in type(self).__dict__.items():
            if isinstance(field, DirChildField):
                yield getattr(self, name)

    def recognize(self, path):
        slashend = os.path.split(str(path))[1] in ("", ".", "..")
        path = Path(os.path.abspath(str(path)))

        route = [(path, slashend)]
        route.extend((parent, True) for parent in path.parents)
        i = next((i for i, (path, _) in enumerate(route) if path == self.abs), None)
        if i is None:
            return UnrecognizedPath(path, slashend), None
        route = route[i::-1][1:]

        curr_index = ()
        curr_path_type = type(self)
        for next_path, next_slashend in route:
            if not issubclass(curr_path_type, RecognizedDirPath):
                return UnrecognizedPath(path, slashend), None

            for index, field in enumerate(curr_path_type.get_fields()):
                if isinstance(field, DirWildcardField):
                    if not field.match(path, slashend):
                        continue

                    curr_index = (*curr_index, index)
                    curr_path_type = field.path_type
                    return curr_path_type(path, slashend), curr_index

                elif isinstance(field, (DirPatternField, DirChildField)):
                    if not field.match(next_path, next_slashend):
                        continue

                    curr_index = (*curr_index, index)
                    curr_path_type = field.path_type
                    break

                else:
                    assert False

            else:
                return UnrecognizedPath(path, slashend), None

        return curr_path_type(path, slashend), curr_index


@dataclasses.dataclass(frozen=True)
class DirPatternField:
    pattern: str
    path_type: type

    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.path_type
        else:
            return self.find(instance.abs)

    def find(self, parent):
        for path in parent.iterdir():
            slashend = False
            if self.match(path, slashend):
                path = self.path_type(path, slashend)
                try:
                    path.validate()
                except WrongFileTypeException:
                    continue
                except MissingFileException:
                    assert False
                yield path

    def match(self, path, slashend):
        if issubclass(self.path_type, RecognizedFilePath) and slashend:
            return False
        if not path.match(self.pattern):
            return False
        return True


@dataclasses.dataclass(frozen=True)
class DirWildcardField:
    path_type: type

    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.path_type
        else:
            return self.find(instance.abs)

    def find(self, parent):
        for path in parent.iterdir():
            slashend = False
            if self.match(path, slashend):
                path = self.path_type(path, slashend)
                try:
                    path.validate()
                except WrongFileTypeException:
                    continue
                except MissingFileException:
                    assert False
                yield path

    def match(self, path, slashend):
        if issubclass(self.path_type, RecognizedFilePath) and slashend:
            return False
        return True


@dataclasses.dataclass(frozen=True)
class DirChildField:
    name: str
    path_type: type

    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.path_type
        else:
            return self.path_type(instance.abs / self.name, self.name.endswith("/"))

    def match(self, path, slashend):
        if issubclass(self.path_type, RecognizedFilePath) and slashend:
            return False
        if path.name != self.name:
            return False
        return True


def as_pattern(pattern, path_type=None):
    if path_type is None:
        return lambda path_type: as_pattern(pattern, path_type)

    if not re.fullmatch(r"[^/]+", pattern) or pattern in (".", ".."):
        raise ValueError(f"invalid pattern: {pattern}")

    if not issubclass(path_type, RecognizedPath):
        raise TypeError(f"invalid path type: {path_type}")

    if pattern == "**":
        return DirWildcardField(path_type)
    else:
        return DirPatternField(pattern, path_type)


def as_child(name, path_type=None):
    if path_type is None:
        return lambda path_type: as_child(name, path_type)

    if not re.fullmatch(r"[^/]+", name) or name in (".", ".."):
        raise ValueError(f"invalid name: {name}")

    if not issubclass(path_type, RecognizedPath):
        raise TypeError(f"invalid path type: {path_type}")

    return DirChildField(name, path_type)


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


class FileManager:
    ROOT_ENVVAR = "KAIKO"
    ROOT_PATH = "~/.local/share/K-AIKO"

    def __init__(self, root_path_type, provider):
        self.username = getpass.getuser()
        root = Path(self.ROOT_PATH).expanduser()
        self.root = root_path_type(root, True)
        self.current = self.root
        self.settings = FileManagerSettings()
        self.provider = provider

    def set_settings(self, settings):
        self.settings = settings

    @property
    def logger(self):
        return self.provider.get(Logger)

    def fix(self):
        logger = self.logger
        provider = self.provider

        def go(path):
            REDUNDANT_EXT = ".redundant"

            for subpath in path.get_children():
                try:
                    subpath.validate()

                except WrongFileTypeException:
                    logger.print(f"[warn]Wrong file type [emph]{subpath!s}[/][/]")
                    subpath_ = subpath.abs.parent / (subpath.abs.name + REDUNDANT_EXT)
                    logger.print(f"[data/] Rename to [emph]{subpath_!s}[/]...")
                    subpath.abs.rename(subpath_)
                    logger.print(f"[data/] Create file [emph]{subpath!s}[/]...")
                    subpath.mk(provider)

                except MissingFileException:
                    logger.print(f"[warn]Missing file [emph]{subpath!s}[/][/]")
                    logger.print(f"[data/] Create file [emph]{subpath!s}[/]...")
                    subpath.mk(provider)

                if isinstance(subpath, RecognizedDirPath):
                    go(subpath)

        try:
            self.root.validate()
        except WrongFileTypeException:
            raise RuntimeError(f"Workspace name {self.root!s} is already taken.")
        except MissingFileException:
            logger.print(f"[warn]The KAIKO workspace is missing[/]")
            logger.print(f"[data/] Create KAIKO workspace...")
            self.root.mk(provider)
            logger.print(
                f"[data/] Your data will be stored in {logger.as_uri(self.root.abs)}"
            )
        return go(self.root)

        logger.print(flush=True)

    def remove(self):
        shutil.rmtree(str(self.root))

    def recognize(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        return self.root.recognize(path)[0]

    def as_relative_path(self, path):
        relpath = str(path.abs.relative_to(self.root.abs))
        if relpath == ".":
            relpath = ""
        relpath = os.path.join(f"${self.ROOT_ENVVAR}", relpath)
        if path.slashend:
            relpath = os.path.join(relpath, "")
        return relpath

    def ls(self):
        logger = self.logger

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

        path = self.current.abs.resolve()

        res = []
        for child in path.iterdir():
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

            recpath, ind = self.root.recognize(self.current.abs / child.name)
            desc = recpath.desc(self.provider)
            desc = logger.rich.parse(desc, root_tag=True) if desc is not None else None

            ordering_key = (
                isinstance(recpath, UnrecognizedPath),
                ind,
                child.is_symlink(),
                not child.is_dir(),
                child.suffix,
                child.stem,
            )

            if isinstance(recpath, UnrecognizedPath):
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

    def validate(
        self,
        path,
        should_in_range=True,
        should_exist=True,
        file_type="file",
    ):
        logger = self.logger

        if should_in_range and not path.abs.resolve().is_relative_to(self.root.abs):
            logger.print(f"[warn]Out of root directory: {logger.as_uri(path.abs.resolve())}[/]")
            return False

        if should_exist and not path.exists():
            logger.print(f"[warn]No such file: {logger.as_uri(path.abs)}[/]")
            return False

        if not should_exist and path.exists():
            logger.print(f"[warn]File already exists: {logger.as_uri(path.abs)}[/]")
            return False

        if file_type == "file" and path.slashend:
            logger.print(f"[warn]Not a file: {logger.as_uri(path.abs)}[/]")
            return False

        if file_type == "file" and path.exists() and not path.is_file():
            logger.print(f"[warn]Not a file: {logger.as_uri(path.abs)}[/]")
            return False

        if file_type == "dir" and path.exists() and not path.is_dir():
            logger.print(f"[warn]Not a directory: {logger.as_uri(path.abs)}[/]")
            return False

        return True

    def cd(self, path):
        logger = self.logger

        if not self.validate(path, file_type="dir"):
            return
        path = dataclasses.replace(path, slashend=True)
        self.current = path

    def mk(self, path):
        logger = self.logger

        if not self.validate(path, should_exist=False, file_type="all"):
            return

        try:
            path.mk(self.provider)
        except Exception:
            logger.print(f"[warn]Failed to make file: {logger.as_uri(path.abs)}[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def rm(self, path):
        logger = self.logger

        if not self.validate(path, file_type="all"):
            return

        try:
            path.rm(self.provider)
        except Exception:
            logger.print(f"[warn]Failed to remove file: {logger.as_uri(path.abs)}[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def mv(self, path, dst):
        logger = self.logger

        if not self.validate(path, file_type="all"):
            return

        if not self.validate(dst, should_exist=False, file_type="all"):
            return

        if type(path) != type(dst):
            logger.print(
                f"[warn]Different file type: {logger.as_uri(path.abs)} -> {logger.as_uri(dst.abs)}[/]"
            )
            return

        try:
            path.mv(dst, self.provider)
        except Exception:
            logger.print(
                f"[warn]Failed to move file: {logger.as_uri(path.abs)} -> {logger.as_uri(dst.abs)}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def cp(self, src, path):
        logger = self.logger

        if not self.validate(path, should_exist=False, file_type="all"):
            return

        if not self.validate(src, should_in_range=False, file_type="all"):
            return

        try:
            src.cp(path, self.provider)
        except Exception:
            logger.print(
                f"[warn]Failed to copy file: {logger.as_uri(src.abs)} -> {logger.as_uri(path.abs)}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def make_parser(self, desc=None, filter=lambda _: True):
        return PathParser(self.root, self.current.abs, desc=desc, filter=filter)


class FilesCommand:
    def __init__(self, provider):
        self.provider = provider

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    @cmd.function_command
    def ls(self):
        self.file_manager.ls()

    @cmd.function_command
    def cd(self, path):
        self.file_manager.cd(path)

    @cmd.function_command
    def cat(self, path):
        file_manager = self.file_manager
        logger = self.logger

        try:
            content = path.abs.read_text()
        except UnicodeDecodeError:
            logger.print("[warn]Cannot read binary file.[/]")
            return

        relpath = file_manager.as_relative_path(path)
        code = logger.format_code(content, title=str(relpath))
        logger.print(code)

    @cmd.function_command
    def mk(self, path):
        self.file_manager.mk(path)

    @cmd.function_command
    def rm(self, path):
        self.file_manager.rm(path)

    @cmd.function_command
    def mv(self, path, dst):
        self.file_manager.mv(path, dst)

    @cmd.function_command
    def cp(self, src, path):
        self.file_manager.cp(src, path)

    @cd.arg_parser("path")
    def _cd_path_parser(self):
        return self.file_manager.make_parser(filter=os.path.isdir)

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        return self.file_manager.make_parser(filter=os.path.isfile)

    @mv.arg_parser("dst")
    @cp.arg_parser("path")
    @mk.arg_parser("path")
    def _any_path_parser(self, *_, **__):
        return self.file_manager.make_parser()

    @rm.arg_parser("path")
    @mv.arg_parser("path")
    def _lexists_path_parser(self, *_, **__):
        return self.file_manager.make_parser(filter=os.path.lexists)

    @cp.arg_parser("src")
    def _exists_path_parser(self, *_, **__):
        return self.file_manager.make_parser(filter=os.path.exists)

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
        from .profiles import ProfileManager
        profile_is_changed = self.provider.get(ProfileManager).is_changed()

        if profile_is_changed:
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


class PathParser(cmd.ArgumentParser):
    r"""Parse a file path."""

    def __init__(
        self,
        root,
        prefix=".",
        desc=None,
        filter=lambda _: True,
    ):
        r"""Contructor.

        Parameters
        ----------
        root : RecognizedDirPath
            The root of structure.
        prefix : str or Path, optional
            The prefix of path.
        desc : str, optional
            The description of this argument.
        filter : function, optional
            The filter function for the valid path, the argument is absolute
            path in the str type.
        """
        self.root = root
        self.prefix = prefix
        self._desc = desc
        self.filter = filter

    def desc(self):
        return self._desc if self._desc is not None else "It should be a path"

    def parse(self, token):
        path = token
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        path = os.path.join(self.prefix, path or ".")
        path, _ = self.root.recognize(path)

        if not self.filter(str(path)):
            desc = self.desc()
            raise cmd.CommandParseError(
                "Not a valid file type" + ("\n" + desc if desc is not None else "")
            )

        return path

    def suggest(self, token):
        suggestions = []

        def expand(path):
            path = os.path.expanduser(path)
            path = os.path.expandvars(path)
            path = os.path.join(self.prefix, path or ".")
            return path

        # check path
        currpath = expand(token)
        try:
            is_dir = os.path.isdir(currpath) and currpath.endswith("/")
            is_file = os.path.isfile(currpath)
        except ValueError:
            return suggestions

        if is_file and self.filter(currpath):
            suggestions.append((token or ".") + "\000")

        if is_dir and self.filter(currpath):
            suggestions.append(os.path.join(token or ".", "") + "\000")

        # separate parent and partial child name
        parent, child = os.path.split(token)
        parentpath = expand(parent)
        if not os.path.isdir(parentpath):
            return suggestions

        names = cmd.fit(child, os.listdir(parentpath))
        for name in names:
            # only suggest hidden files when starting with .
            if not child.startswith(".") and name.startswith("."):
                continue

            sugg = os.path.join(parent, name)
            suggpath = expand(sugg)

            if os.path.isdir(suggpath):
                sugg = os.path.join(sugg, "")
                suggestions.append(sugg)

            elif os.path.isfile(suggpath) and self.filter(suggpath):
                suggestions.append(sugg + "\000")

        return suggestions


def CdCommand(provider):
    def make_cd_command(name):
        @cmd.function_command
        def cd_command(self):
            file_manager = self.provider.get(FileManager)
            path = file_manager.recognize(file_manager.current.abs / name)
            return file_manager.cd(path)
        return cd_command

    attrs = {}
    attrs["provider"] = provider
    attrs[".."] = make_cd_command("..")
    file_manager = provider.get(FileManager)
    curr_path = file_manager.current.abs.resolve()
    for child in curr_path.iterdir():
        if child.is_dir():
            attrs[child.name + "/"] = make_cd_command(child.name)

    return type("CdCommand", (object,), attrs)()
