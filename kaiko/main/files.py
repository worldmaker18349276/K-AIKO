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


class PathDescriptor:
    def __init__(self, provider):
        """Constructor of path descriptor.

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

    def mv(self, path, dst):
        raise InvalidFileOperation

    def cp(self, src, path):
        raise InvalidFileOperation

    def show(self, indent=0):
        return "[any] " + (type(self).__doc__ or "")


class WildCardDescriptor(PathDescriptor):
    def show(self, indent=0):
        return "[any] " + (type(self).__doc__ or "")


class FileDescriptor(PathDescriptor):
    def show(self, indent=0):
        return "[file] " + (type(self).__doc__ or "")


class DirDescriptor(PathDescriptor):
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

    def recognize(self, path, root="."):
        root = Path(os.path.abspath(str(root)))
        slashend = os.path.split(str(path))[1] in ("", ".", "..")
        path = Path(os.path.abspath(str(path)))

        route = [(path, slashend)]
        route.extend((parent, True) for parent in path.parents)
        i = next((i for i, (path, _) in enumerate(route) if path == root), None)
        if i is None:
            return RecognizedPath(path, slashend, None), None
        route = route[i::-1][1:]

        index = ()
        descriptor = self
        for curr_path, curr_is_dir in route:
            if not isinstance(descriptor, DirDescriptor):
                return RecognizedPath(path, slashend, None), None

            for i, child in enumerate(descriptor.children()):
                if child.pattern == "**":
                    if isinstance(child.descriptor, FileDescriptor) and slashend:
                        continue
                    index = (*index, i)
                    return RecognizedPath(path, slashend, child.descriptor), index

                else:
                    if isinstance(child.descriptor, FileDescriptor) and curr_is_dir:
                        continue

                    if curr_path.match(child.pattern):
                        index = (*index, i)
                        descriptor = child.descriptor
                        break

            else:
                return RecognizedPath(path, slashend, None), None

        return RecognizedPath(path, slashend, descriptor), index


@dataclasses.dataclass(frozen=True)
class RecognizedPath:
    abs: Path
    slashend: bool
    descriptor: Optional[PathDescriptor]

    def desc(self):
        return self.descriptor.desc(self.abs) if self.descriptor is not None else None

    def info(self):
        return self.descriptor.info(self.abs) if self.descriptor is not None else None

    def exists(self):
        return self.abs.exists()

    def is_dir(self):
        return self.abs.is_dir()

    def is_file(self):
        return self.abs.is_file()

    def __str__(self):
        abspath = str(self.abs)
        if self.slashend:
            abspath = os.path.join(abspath, "")
        return abspath


@dataclasses.dataclass(frozen=True)
class DirChild:
    pattern: str
    is_required: bool
    descriptor: PathDescriptor


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
            if not issubclass(descriptor_type, PathDescriptor):
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
    current: RecognizedPath
    structure: DirDescriptor
    settings: FileManagerSettings

    ROOT_ENVVAR = "KAIKO"
    ROOT_PATH = "~/.local/share/K-AIKO"

    @classmethod
    def create(cls, structure):
        username = getpass.getuser()
        root = Path(cls.ROOT_PATH).expanduser().resolve()
        settings = FileManagerSettings()
        current, _ = structure.recognize(root, root=root)
        return cls(username, root, current, structure, settings)

    def set_settings(self, settings):
        self.settings = settings

    def check_is_prepared(self, logger):
        def go(path, tree):
            res = True
            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)

                    if isinstance(child.descriptor, DirDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] There is a missing directory [emph]{subpath!s}[/].")
                            res = False
                        if not subpath.is_dir() or subpath.is_symlink():
                            logger.print(f"[data/] Bad file structure: [emph]{subpath!s}[/] should be a directory.")
                            res = False
                        if not go(subpath, child.descriptor):
                            res = False
                    elif isinstance(child.descriptor, FileDescriptor):
                        if not subpath.exists():
                            logger.print(f"[data/] There is a missing file [emph]{subpath!s}[/].")
                            res = False
                        if not subpath.is_file() or subpath.is_symlink():
                            logger.print(f"[data/] Bad file structure: [emph]{subpath!s}[/] should be a file.")
                            res = False
                    else:
                        raise TypeError(child.descriptor)
            return res

        if not self.root.exists():
            logger.print(f"[data/] The workspace [emph]{self.root!s}[/] is missing.")
            return False

        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError(f"Workspace name {self.root!s} is already taken.")

        return go(self.root, self.structure)

    def prepare(self, logger):
        logger.print("[data/] Prepare your profile...")

        def go(path, tree):
            REDUNDANT_EXT = ".redundant"

            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)

                    if isinstance(child.descriptor, DirDescriptor):
                        if subpath.exists() and (not subpath.is_dir() or subpath.is_symlink()):
                            logger.print(f"[data/] Rename directory [emph]{subpath!s}[/]...")
                            subpath.rename(subpath.parent / (subpath.name + REDUNDANT_EXT))
                        if not subpath.exists():
                            logger.print(f"[data/] Create directory [emph]{subpath!s}[/]...")
                            subpath.mkdir()
                        go(subpath, child.descriptor)
                    elif isinstance(child.descriptor, FileDescriptor):
                        if subpath.exists() and (not subpath.is_file() or subpath.is_symlink()):
                            logger.print(f"[data/] Rename file [emph]{subpath!s}[/]...")
                            subpath.rename(subpath.parent / (subpath.name + REDUNDANT_EXT))
                        if not subpath.exists():
                            logger.print(f"[data/] Create file [emph]{subpath!s}[/]...")
                            subpath.touch()
                    else:
                        raise TypeError(child.descriptor)

        if not self.root.exists():
            self.root.mkdir()
            logger.print(
                f"[data/] Your data will be stored in {logger.as_uri(self.root)}"
            )

        go(self.root, self.structure)

        logger.print(flush=True)

    def remove(self, logger):
        shutil.rmtree(str(self.root))

    def recognize(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        return self.structure.recognize(path, root=self.root)[0]

    def as_relative_path(self, recpath):
        relpath = str(recpath.abs.relative_to(self.root))
        if relpath == ".":
            relpath = ""
        relpath = os.path.join(f"${self.ROOT_ENVVAR}", relpath)
        if recpath.slashend:
            relpath = os.path.join(relpath, "")
        return relpath

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

            recpath, ind = self.structure.recognize(self.current.abs / child.name, root=self.root)
            desc = recpath.desc()
            desc = logger.rich.parse(desc, root_tag=True) if desc is not None else None

            ordering_key = (
                recpath.descriptor is None,
                ind,
                child.is_symlink(),
                not child.is_dir(),
                child.suffix,
                child.stem,
            )

            if recpath.descriptor is None:
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
        logger,
        path,
        should_in_range=True,
        should_exist=True,
        file_type="file",
    ):
        if should_in_range and not path.abs.resolve().is_relative_to(self.root):
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

    def cd(self, logger, path):
        if not self.validate(logger, path, file_type="dir"):
            return
        path = dataclasses.replace(path, slashend=True)
        self.current = path

    def mk(self, logger, path):
        if not self.validate(logger, path, should_exist=False, file_type="all"):
            return

        try:
            if path.descriptor is not None:
                path.descriptor.mk(path.abs)
            elif path.slashend:
                path.abs.mkdir(exist_ok=False)
            else:
                path.abs.touch(exist_ok=False)
        except Exception:
            logger.print(f"[warn]Failed to make file: {logger.as_uri(path.abs)}[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def rm(self, logger, path):
        if not self.validate(logger, path, file_type="all"):
            return

        try:
            if path.descriptor is not None:
                path.descriptor.rm(path.abs)
            elif path.abs.is_dir() and not path.abs.is_symlink():
                path.abs.rmdir()
            else:
                path.abs.unlink()
        except Exception:
            logger.print(f"[warn]Failed to remove file: {logger.as_uri(path.abs)}[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def mv(self, logger, path, dst):
        if not self.validate(logger, path, file_type="all"):
            return

        if not self.validate(logger, dst, should_exist=False, file_type="all"):
            return

        if path.descriptor != dst.descriptor:
            logger.print(
                f"[warn]Different file type: {logger.as_uri(path.abs)} -> {logger.as_uri(dst.abs)}[/]"
            )
            return

        try:
            if path.descriptor is not None:
                path.descriptor.mv(path.abs, dst.abs)
            else:
                path.abs.rename(dst.abs)
        except Exception:
            logger.print(
                f"[warn]Failed to move file: {logger.as_uri(path.abs)} -> {logger.as_uri(dst.abs)}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def cp(self, logger, src, path):
        if not self.validate(logger, path, should_exist=False, file_type="all"):
            return

        if not self.validate(logger, src, should_in_range=False, file_type="all"):
            return

        try:
            if path.descriptor is not None:
                path.descriptor.cp(src.abs, path.abs)
            else:
                shutil.copy(src.abs, path.abs)
        except Exception:
            logger.print(
                f"[warn]Failed to copy file: {logger.as_uri(src.abs)} -> {logger.as_uri(path.abs)}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

    def make_parser(self, desc=None, filter=lambda _: True):
        return PathParser(
            self.root,
            self.current.abs,
            self.structure,
            desc=desc,
            filter=filter,
        )


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
    def ls(self):
        self.file_manager.ls(self.logger)

    @cmd.function_command
    def cd(self, path):
        self.file_manager.cd(self.logger, path)

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
        self.file_manager.mk(self.logger, path)

    @cmd.function_command
    def rm(self, path):
        self.file_manager.rm(self.logger, path)

    @cmd.function_command
    def mv(self, path, dst):
        self.file_manager.mv(self.logger, path, dst)

    @cmd.function_command
    def cp(self, src, path):
        self.file_manager.cp(self.logger, src, path)

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


class PathParser(cmd.ArgumentParser):
    r"""Parse a file path."""

    def __init__(
        self,
        root=".",
        prefix=".",
        structure=None,
        desc=None,
        filter=lambda _: True,
    ):
        r"""Contructor.

        Parameters
        ----------
        root : str, optional
            The root of structure.
        prefix : str, optional
            The prefix of path.
        structure : PathDescriptor, optional
            The structure of path to parse.
        desc : str, optional
            The description of this argument.
        filter : function, optional
            The filter function for the valid path, the argument is absolute
            path in the str type.
        """
        self.root = root
        self.prefix = prefix
        self.structure = structure
        self._desc = desc
        self.filter = filter

    def desc(self):
        return self._desc if self._desc is not None else "It should be a path"

    def parse(self, token):
        path = token
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        path = os.path.join(self.prefix, path or ".")
        path, _ = self.structure.recognize(path, root=self.root)

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
            logger = self.provider.get(Logger)
            path = file_manager.recognize(file_manager.current.abs / name)
            return file_manager.cd(logger, path)
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
