import os
import dataclasses
import getpass
import re
import shutil
from inspect import cleandoc
from pathlib import Path
from ..utils import config as cfg
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from .loggers import Logger


@dataclasses.dataclass(frozen=True)
class RecognizedPath:
    abs: Path
    slashend: bool

    def normalize(self):
        return self

    def try_relative_to(self, path):
        try:
            relpath = str(self.abs.relative_to(path.abs))
        except ValueError:
            relpath = str(self.abs)
        if self.slashend:
            relpath = os.path.join(relpath, "")
        return relpath

    def __str__(self):
        abspath = str(self.abs)
        if self.slashend:
            abspath = os.path.join(abspath, "")
        return abspath

    def info(self, provider):
        doc = type(self).__doc__
        if doc is None:
            return None
        return doc.split("\n", 1)[0]

    def info_detailed(self, provider):
        doc = type(self).__doc__
        if doc is None:
            return None

        docs = doc.split("\n\n", 1)
        if len(docs) == 1:
            return None

        return cleandoc(docs[1])

    def mk(self, provider):
        file_manager = provider.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        if self.slashend:
            self.abs.mkdir(exist_ok=False)
        else:
            self.abs.touch(exist_ok=False)

    def rm(self, provider):
        file_manager = provider.get(FileManager)
        file_manager.validate_path(self, should_exist=True, file_type="all")
        if self.abs.is_dir() and not self.abs.is_symlink():
            self.abs.rmdir()
        else:
            self.abs.unlink()

    def mv(self, dst, provider):
        if type(self) != type(dst):
            raise ValueError(f"Different file type: {self!s} -> {dst!s}")
        file_manager = provider.get(FileManager)
        file_manager.validate_path(self, should_exist=True, file_type="all")
        file_manager.validate_path(dst, should_exist=False, file_type="all")
        self.abs.rename(dst.abs)

    def cp(self, src, provider):
        file_manager = provider.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        file_manager.validate_path(src, should_exist=True, should_in_range=False, file_type="all")
        shutil.copy(src.abs, self.abs)


class UnrecognizedPath(RecognizedPath):
    pass


class RecognizedFilePath(RecognizedPath):
    def normalize(self):
        return dataclasses.replace(self, slashend=False) if self.slashend else self


class RecognizedDirPath(RecognizedPath):
    def normalize(self):
        return dataclasses.replace(self, slashend=True) if not self.slashend else self

    @classmethod
    def get_fields(cls):
        for field in cls.__dict__.values():
            if isinstance(field, (DirPatternField, DirChildField)):
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
            return UnrecognizedPath(path, slashend)
        route = route[i::-1][1:]

        curr_path_type = type(self)
        for next_path, next_slashend in route:
            if not issubclass(curr_path_type, RecognizedDirPath):
                return UnrecognizedPath(path, slashend)

            for index, field in enumerate(curr_path_type.get_fields()):
                if not field.match(next_path, next_slashend):
                    continue

                curr_path_type = field.path_type
                break

            else:
                return UnrecognizedPath(path, slashend)

        return curr_path_type(path, slashend)

    def iterdir(self):
        fields = list(self.get_fields())
        for path in self.abs.iterdir():
            slashend = path.is_dir()
            for field in fields:
                if not field.match(path, slashend):
                    continue

                if issubclass(field.path_type, RecognizedFilePath) and not path.is_file():
                    continue
                if issubclass(field.path_type, RecognizedDirPath) and not path.is_dir():
                    continue

                yield field.path_type(path, slashend)
                break

            else:
                yield UnrecognizedPath(path, slashend)


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
            slashend = path.is_dir()
            if self.match(path, slashend):
                if issubclass(self.path_type, RecognizedFilePath) and not path.is_file():
                    continue
                if issubclass(self.path_type, RecognizedDirPath) and not path.is_dir():
                    continue

                yield self.path_type(path, slashend)

    def match(self, path, slashend):
        if issubclass(self.path_type, RecognizedFilePath) and slashend:
            return False
        if not path.match(self.pattern):
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

    return DirPatternField(pattern, path_type)


def as_child(name, path_type=None):
    if path_type is None:
        return lambda path_type: as_child(name, path_type)

    if not re.fullmatch(r"[^/]+", name) or name in (".", ".."):
        raise ValueError(f"invalid name: {name}")

    if not issubclass(path_type, RecognizedPath):
        raise TypeError(f"invalid path type: {path_type}")

    return DirChildField(name, path_type)


def rename_path(parent, stem, suffix):
    name = stem + suffix
    if "/" in name or not name.isprintable():
        raise ValueError(f"Invalid file name: {name}")

    path = parent / name
    n = 1
    while path.exists():
        n += 1
        name = f"{stem} ({n}){suffix}"
        path = parent / name

    return path


class InvalidFileOperation(Exception):
    pass


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
        file_info : str
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
        file_info: str = "  [weight=dim][slot/][/]"

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

    def init_env(self):
        os.environ[self.ROOT_ENVVAR] = str(self.root.abs)

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
                subpath_mu = self.as_relative_path(subpath, markup=True)
                if not subpath.abs.exists():
                    logger.print(f"[warn]Missing file {subpath_mu}[/]")
                    logger.print(f"[data/] Create file [emph]{logger.escape(str(subpath))}[/]...")
                    subpath.mk(provider)

                elif (
                    isinstance(subpath, RecognizedFilePath) and not subpath.abs.is_file()
                    or isinstance(subpath, RecognizedDirPath) and not subpath.abs.is_dir()
                ):
                    logger.print(f"[warn]Wrong file type {subpath_mu}[/]")
                    subpath_ = subpath.abs.parent / (subpath.abs.name + REDUNDANT_EXT)
                    subpath_mu_ = self.as_relative_path(subpath_, markup=True)
                    logger.print(f"[data/] Rename to {subpath_mu_}...")
                    subpath.abs.rename(subpath_)
                    logger.print(f"[data/] Create file {subpath_mu}...")
                    subpath.mk(provider)

                if isinstance(subpath, RecognizedDirPath):
                    go(subpath)

        if not self.root.abs.exists():
            logger.print(f"[warn]The KAIKO workspace is missing[/]")
            logger.print(f"[data/] Create KAIKO workspace...")
            self.root.mk(provider)
            logger.print(
                f"[data/] Your data will be stored in {logger.as_uri(self.root.abs)}"
            )

        elif not self.root.abs.is_dir():
            raise RuntimeError(f"Workspace name {self.root!s} is already taken.")

        return go(self.root)

        logger.print(flush=True)

    def remove(self):
        shutil.rmtree(str(self.root))

    def recognize(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        return self.root.recognize(path)

    def iterdir(self):
        current_path = self.current
        if isinstance(current_path, RecognizedDirPath):
            return current_path.iterdir()
        elif current_path.abs.is_dir():
            return (UnrecognizedPath(subpath, subpath.is_dir()) for subpath in current_path.abs.iterdir())
        else:
            return ()

    def get_field_types(self):
        current_path = self.current
        if isinstance(current_path, RecognizedDirPath):
            return [field.path_type for field in current_path.get_fields()]
        else:
            return []

    def as_relative_path(self, path, parent=None, markup=False):
        if parent is not None:
            relpath = path.try_relative_to(parent)
            if markup:
                relpath = f"[emph]{self.logger.escape(relpath)}[/]"
            return relpath

        if not path.abs.is_relative_to(self.root.abs):
            relpath = str(path)
            if markup:
                relpath = f"[emph]{self.logger.escape(relpath)}[/]"
            return relpath

        relpath = str(path.abs.relative_to(self.root.abs))
        if relpath == ".":
            relpath = ""
        relpath = os.path.join(f"${self.ROOT_ENVVAR}", relpath)

        if path.slashend:
            relpath = os.path.join(relpath, "")

        if markup:
            relpath = f"[emph]{self.logger.escape(relpath)}[/]"

        return relpath

    def validate_path(self, path, should_exist=None, should_in_range=True, file_type="file"):
        # should_exist: Optional[bool]

        if should_in_range and not path.abs.resolve().is_relative_to(self.root.abs):
            raise InvalidFileOperation(f"Out of root directory: {str(path.abs.resolve())}")

        path_str = self.as_relative_path(path)

        if path.slashend and path.abs.is_file():
            raise InvalidFileOperation(f"The given path ends with a slash, but found a file: {path_str}")

        if path.slashend and file_type == "file":
            raise InvalidFileOperation(f"The given path ends with a slash, but a file is required: {path_str}")

        if should_exist is True and not path.abs.exists():
            raise InvalidFileOperation(f"No such file: {path_str}")

        if should_exist is False and path.abs.exists():
            raise InvalidFileOperation(f"File already exists: {path_str}")

        if file_type == "file" and path.abs.exists() and not path.abs.is_file():
            raise InvalidFileOperation(f"Not a file: {path_str}")

        if file_type == "dir" and path.abs.exists() and not path.abs.is_dir():
            raise InvalidFileOperation(f"Not a directory: {path_str}")

    def cd(self, path):
        try:
            self.validate_path(path, should_exist=True, file_type="dir")

        except InvalidFileOperation as e:
            logger = self.logger
            path_mu = self.as_relative_path(path, markup=True)
            logger.print(f"[warn]Failed to change current directory to {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        self.current = path.normalize()

    def mk(self, path):
        try:
            path.mk(self.provider)

        except InvalidFileOperation as e:
            logger = self.logger
            path_mu = self.as_relative_path(path, markup=True)
            logger.print(f"[warn]Failed to make file: {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def rm(self, path):
        try:
            path.rm(self.provider)

        except InvalidFileOperation as e:
            logger = self.logger
            path_mu = self.as_relative_path(path, markup=True)
            logger.print(f"[warn]Failed to remove file: {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def mv(self, path, dst):
        try:
            path.mv(dst, self.provider)

        except InvalidFileOperation as e:
            logger = self.logger
            path_mu = self.as_relative_path(path, markup=True)
            dst_mu = self.as_relative_path(dst, markup=True)
            logger.print(
                f"[warn]Failed to move file: {path_mu} -> {dst_mu}[/]"
            )
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def cp(self, src, path):
        try:
            path.cp(src, self.provider)

        except InvalidFileOperation as e:
            logger = self.logger
            src_mu = self.as_relative_path(src, markup=True)
            path_mu = self.as_relative_path(path, markup=True)
            logger.print(
                f"[warn]Failed to copy file: {src_mu} -> {path_mu}[/]"
            )
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def make_parser(self, desc=None, filter=lambda _: True):
        return PathParser(
            self.root,
            self.current.abs,
            desc=desc,
            filter=filter,
            suggs=[f"${self.ROOT_ENVVAR}/"],
            provider=self.provider,
        )


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
        logger = self.logger
        file_manager = self.file_manager
        display_settings = file_manager.settings.display
        current_path = file_manager.current

        file_item = logger.rich.parse(display_settings.file_item, expand=False, slotted=True)
        file_unknown = logger.rich.parse(display_settings.file_unknown, expand=False, slotted=True)
        file_info = logger.rich.parse(display_settings.file_info, expand=False, slotted=True)
        file_dir = logger.rich.parse(display_settings.file_dir, expand=False, slotted=True)
        file_script = logger.rich.parse(display_settings.file_script, expand=False, slotted=True)
        file_beatmap = logger.rich.parse(display_settings.file_beatmap, expand=False, slotted=True)
        file_sound = logger.rich.parse(display_settings.file_sound, expand=False, slotted=True)
        file_link = logger.rich.parse(display_settings.file_link, expand=False, slotted=True)
        file_normal = logger.rich.parse(display_settings.file_normal, expand=False, slotted=True)
        file_other = logger.rich.parse(display_settings.file_other, expand=False, slotted=True)

        field_types = file_manager.get_field_types()

        res = []
        for path in file_manager.iterdir():
            if path.abs.is_symlink():
                name = logger.escape(str(path.abs.readlink()), type="all")
            else:
                name = logger.escape(path.abs.name, type="all")
            name = logger.rich.parse(name, expand=False)

            if path.abs.is_dir():
                name = file_dir(name)

            elif path.abs.is_file():
                if path.abs.suffix in [".py", ".kaiko-profile"]:
                    name = file_script(name)
                elif path.abs.suffix in [".ka", ".kaiko", ".osu"]:
                    name = file_beatmap(name)
                elif path.abs.suffix in [".wav", ".mp3", ".mp4", ".m4a", ".ogg"]:
                    name = file_sound(name)
                else:
                    name = file_normal(name)

            else:
                name = file_other(name)

            if path.abs.is_symlink():
                linkname = logger.escape(path.abs.name, type="all")
                linkname = logger.rich.parse(linkname, expand=False)
                name = file_link(src=linkname, dst=name)

            ind = field_types.index(type(path)) if type(path) in field_types else len(field_types)
            info = path.info(self.provider)
            info = logger.rich.parse(info, root_tag=True, expand=False) if info is not None else None

            ordering_key = (
                isinstance(path, UnrecognizedPath),
                ind,
                path.abs.is_symlink(),
                not path.abs.is_dir(),
                path.abs.suffix,
                path.abs.stem,
            )

            if isinstance(path, UnrecognizedPath):
                name = file_unknown(name)
            info = file_info(info) if info is not None else mu.Text("")
            name = file_item(name)

            width = logger.rich.widthof(name.expand())
            res.append((ordering_key, width, name, info))

        res = sorted(res, key=lambda e: e[0])
        max_width = max((width for _, width, _, _ in res), default=0)

        with logger.stack():
            for _, width, name, info in res:
                padding = " "*(max_width - width) if width != -1 else " "
                logger.print(name, end=padding)
                logger.print(info, end="\n")

    @cmd.function_command
    def cd(self, path):
        self.file_manager.cd(path)

    @cmd.function_command
    def cat(self, path):
        file_manager = self.file_manager
        logger = self.logger

        try:
            file_manager.validate_path(path, should_exist=True, file_type="file")
        except InvalidFileOperation as e:
            path_mu = self.as_relative_path(path, markup=True)
            logger.print(f"[warn]Failed to change directory to {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        try:
            content = path.abs.read_text()
        except UnicodeDecodeError:
            logger.print("[warn]Cannot read binary file.[/]")
            return

        relpath = file_manager.as_relative_path(path)
        code = logger.format_code(content, title=relpath)
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
        return self.file_manager.make_parser(
            desc="It should be an existing directory",
            filter=lambda path: os.path.isdir(str(path)),
        )

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        return self.file_manager.make_parser(
            desc="It should be an existing file",
            filter=lambda path: os.path.isfile(str(path)),
        )

    @mv.arg_parser("dst")
    @cp.arg_parser("path")
    @mk.arg_parser("path")
    def _any_path_parser(self, *_, **__):
        return self.file_manager.make_parser()

    @rm.arg_parser("path")
    @mv.arg_parser("path")
    def _lexists_path_parser(self, *_, **__):
        return self.file_manager.make_parser(
            desc="It should be an existing path",
            filter=lambda path: os.path.lexists(str(path)),
        )

    @cp.arg_parser("src")
    def _exists_path_parser(self, *_, **__):
        return self.file_manager.make_parser(
            desc="It should be an existing path",
            filter=lambda path: os.path.exists(str(path)),
        )

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
        suggs=None,
        provider=None,
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
            The filter function for the result.
        suggs : list of str, optional
            The absolute path that will be suggested.
        provider : Provider, optional
        """
        self.root = root
        self.prefix = prefix
        self._desc = desc
        self.filter = filter
        self.suggs = suggs
        self.provider = provider

    def desc(self):
        return self._desc if self._desc is not None else "It should be a path"

    def parse(self, token):
        path = token
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        path = os.path.join(self.prefix, path or ".")
        path = self.root.recognize(path)

        if not self.filter(path):
            desc = self.desc()
            raise cmd.CommandParseError(
                "Not a valid file type" + ("\n" + desc if desc is not None else "")
            )

        return path

    def suggest(self, token):
        suggestions = cmd.fit(token, self.suggs)
        if token in suggestions:
            suggestions.remove(token)

        def add(list, elem):
            if elem not in list:
                list.append(elem)

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

        if is_file and self.filter(self.root.recognize(currpath)):
            add(suggestions, (token or ".") + "\000")

        if is_dir and self.filter(self.root.recognize(currpath)):
            add(suggestions, os.path.join(token or ".", "") + "\000")

        # deal with variables
        if re.fullmatch(r"\$\w+", token) and os.path.exists(currpath):
            if os.path.isdir(currpath):
                add(suggestions, os.path.join(token, ""))

            elif os.path.isfile(currpath) and self.filter(self.root.recognize(currpath)):
                add(suggestions, token + "\000")

            return suggestions

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
                add(suggestions, sugg)

            elif os.path.isfile(suggpath) and self.filter(self.root.recognize(suggpath)):
                add(suggestions, sugg + "\000")

        return suggestions

    def info(self, token):
        if self.provider is None:
            return None
        path = self.parse(token)
        info = path.info_detailed(self.provider)
        if info is None:
            info = path.info(self.provider)
        return info


def DirectCdCommand(provider):
    file_commands = FilesCommand(provider)
    def make_cd_command(path, doc):
        command = lambda _: file_commands.cd(path)
        command.__doc__ = doc
        return cmd.function_command(command)

    file_manager = provider.get(FileManager)

    attrs = {}

    path = file_manager.recognize(file_manager.current.abs / "..")
    doc = path.info_detailed(provider)
    attrs[".."] = make_cd_command(path, doc)

    for path in file_manager.iterdir():
        if path.slashend:
            doc = path.info_detailed(provider)
            attrs[path.abs.name + "/"] = make_cd_command(path, doc)

    return type("DirectCdCommand", (object,), attrs)()
