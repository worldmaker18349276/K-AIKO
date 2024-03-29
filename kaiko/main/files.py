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
from ..utils import providers
from .loggers import Logger


@dataclasses.dataclass(frozen=True)
class CognizedPath:
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

    def info(self):
        doc = type(self).__doc__
        if doc is None:
            return None
        return doc.split("\n", 1)[0]

    def info_detailed(self):
        doc = type(self).__doc__
        if doc is None:
            return None

        docs = doc.split("\n\n", 1)
        if len(docs) == 1:
            return None

        return cleandoc(docs[1])

    def fix(self):
        file_manager = providers.get(FileManager)
        logger = providers.get(Logger)

        file_manager.validate_path(self, file_type="all")

        REDUNDANT_EXT = ".redundant"

        path_mu = file_manager.as_relative_path(self)
        path_mu = logger.format_path(path_mu)
        if not self.abs.exists():
            logger.print(f"[warn]Missing file {path_mu}[/]")

        elif (
            isinstance(self, RecognizedFilePath)
            and not self.abs.is_file()
            or isinstance(self, RecognizedDirPath)
            and not self.abs.is_dir()
        ):
            logger.print(f"[warn]Wrong file type {path_mu}[/]")
            path_ = self.abs.parent / (self.abs.name + REDUNDANT_EXT)
            path_mu_ = file_manager.as_relative_path(UnrecognizedPath(path_, False))
            path_mu_ = logger.format_path(path_mu_)
            logger.print(f"[data/] Rename {path_mu} to {path_mu_}...")
            self.abs.rename(path_)

        else:
            return

        logger.print(f"[data/] Create file {path_mu}...")

        if isinstance(self, RecognizedDirPath):
            self.abs.mkdir(exist_ok=False)
        else:
            self.abs.touch(exist_ok=False)

    def mk(self):
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        if self.slashend:
            self.abs.mkdir(exist_ok=False)
        else:
            self.abs.touch(exist_ok=False)

    def rm(self):
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=True, file_type="all")
        if self.abs.is_dir() and not self.abs.is_symlink():
            self.abs.rmdir()
        else:
            self.abs.unlink()

    def mv(self, dst):
        if type(self) != type(dst):
            raise InvalidFileOperation(
                f"Different file type: {type(self).__name__} -> {type(dst).__name__}"
            )
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=True, file_type="all")
        file_manager.validate_path(dst, should_exist=False, file_type="all")
        self.abs.rename(dst.abs)

    def cp(self, src):
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        file_manager.validate_path(
            src, should_exist=True, should_in_range=False, file_type="all"
        )
        shutil.copy(src.abs, self.abs)


class UnmovablePath(CognizedPath):
    def rm(self):
        raise InvalidFileOperation(
            "Deleting important directories or files may crash the program"
        )

    def mv(self, dst):
        raise InvalidFileOperation(
            "Moving important directories or files may crash the program"
        )


class UnrecognizedPath(CognizedPath):
    pass


class RecognizedPath(CognizedPath):
    pass


class RecognizedFilePath(RecognizedPath):
    def normalize(self):
        return dataclasses.replace(self, slashend=False) if self.slashend else self

    def mk(self):
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        self.abs.touch(exist_ok=False)


class RecognizedDirPath(RecognizedPath):
    def normalize(self):
        return dataclasses.replace(self, slashend=True) if not self.slashend else self

    def mk(self):
        file_manager = providers.get(FileManager)
        file_manager.validate_path(self, should_exist=False, file_type="all")
        self.abs.mkdir(exist_ok=False)

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

            for field in curr_path_type.get_fields():
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

                if (
                    issubclass(field.path_type, RecognizedFilePath)
                    and not path.is_file()
                ):
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
                if (
                    issubclass(self.path_type, RecognizedFilePath)
                    and not path.is_file()
                ):
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
            slashend = isinstance(self.path_type, RecognizedDirPath)
            return self.path_type(instance.abs / self.name, slashend)

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

    if not issubclass(path_type, CognizedPath):
        raise TypeError(f"invalid path type: {path_type}")

    return DirPatternField(pattern, path_type)


def as_child(name, path_type=None):
    if path_type is None:
        return lambda path_type: as_child(name, path_type)

    if not re.fullmatch(r"[^/]+", name) or name in (".", ".."):
        raise ValueError(f"invalid name: {name}")

    if not issubclass(path_type, CognizedPath):
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

    def __init__(self, root_path_type):
        self.username = getpass.getuser()
        root = Path(self.ROOT_PATH).expanduser()
        self.root = root_path_type(root, True)
        self.current = self.root
        self.settings = FileManagerSettings()

    def init_env(self):
        os.environ[self.ROOT_ENVVAR] = str(self.root.abs)

    def set_settings(self, settings):
        self.settings = settings

    def fix(self):
        logger = providers.get(Logger)

        if not self.root.abs.exists():
            logger.print(f"[warn]The KAIKO workspace is missing[/]")
            logger.print(f"[data/] Create KAIKO workspace...")
            self.root.mk()
            logger.print(
                f"[data/] Your data will be stored in {logger.as_uri(self.root.abs)}"
            )

        elif not self.root.abs.is_dir():
            raise RuntimeError(f"Workspace name {self.root!s} is already taken.")

        def go(path):
            for subpath in path.get_children():
                subpath.fix()
                if isinstance(subpath, RecognizedDirPath):
                    go(subpath)

        go(self.root)

        logger.print(end="", flush=True)

    def remove(self):
        shutil.rmtree(str(self.root))

    def recognize(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        return self.root.recognize(path)

    def iterdir(self):
        current_path = self.current
        if isinstance(current_path, RecognizedDirPath):
            return current_path.iterdir()
        elif isinstance(current_path, UnrecognizedPath) and current_path.abs.is_dir():
            return (
                UnrecognizedPath(subpath, subpath.is_dir())
                for subpath in current_path.abs.iterdir()
            )
        else:
            return (_ for _ in [])

    def get_field_types(self):
        current_path = self.current
        if isinstance(current_path, RecognizedDirPath):
            return [field.path_type for field in current_path.get_fields()]
        else:
            return []

    def as_relative_path(self, path, parent=None):
        if parent is not None:
            return path.try_relative_to(parent)

        if not path.abs.is_relative_to(self.root.abs):
            return str(path)

        relpath = str(path.abs.relative_to(self.root.abs))
        if relpath == ".":
            relpath = ""
        relpath = os.path.join(f"${self.ROOT_ENVVAR}", relpath)

        if path.slashend:
            relpath = os.path.join(relpath, "")

        return relpath

    def validate_path(
        self, path, should_exist=None, should_in_range=True, file_type="file"
    ):
        # should_exist: Optional[bool]

        if should_in_range and not path.abs.resolve().is_relative_to(self.root.abs):
            raise InvalidFileOperation(
                f"Out of root directory: {str(path.abs.resolve())}"
            )

        path_str = self.as_relative_path(path)

        if path.slashend and path.abs.is_file():
            raise InvalidFileOperation(
                f"The given path ends with a slash, but found a file: {path_str}"
            )

        if path.slashend and file_type == "file":
            raise InvalidFileOperation(
                f"The given path ends with a slash, but a file is required: {path_str}"
            )

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
            logger = providers.get(Logger)
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            logger.print(f"[warn]Failed to change current directory to {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        self.current = path.normalize()

    def mk(self, path):
        try:
            path.mk()

        except InvalidFileOperation as e:
            logger = providers.get(Logger)
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            logger.print(f"[warn]Failed to make file: {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def rm(self, path):
        try:
            path.rm()

        except InvalidFileOperation as e:
            logger = providers.get(Logger)
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            logger.print(f"[warn]Failed to remove file: {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def mv(self, path, dst):
        try:
            path.mv(dst)

        except InvalidFileOperation as e:
            logger = providers.get(Logger)
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            dst_mu = self.as_relative_path(dst)
            dst_mu = logger.format_path(dst_mu)
            logger.print(f"[warn]Failed to move file: {path_mu} -> {dst_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def cp(self, src, path):
        try:
            path.cp(src)

        except InvalidFileOperation as e:
            logger = providers.get(Logger)
            src_mu = self.as_relative_path(src)
            src_mu = logger.format_path(src_mu)
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            logger.print(f"[warn]Failed to copy file: {src_mu} -> {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

    def make_parser(self, desc=None, filter=lambda _: True):
        return PathParser(
            self.root,
            self.current.abs,
            desc=desc,
            filter=filter,
            suggs=[f"${self.ROOT_ENVVAR}/"],
        )


class FilesCommand:
    @cmd.function_command
    def ls(self):
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)
        display_settings = file_manager.settings.display
        current_path = file_manager.current

        file_item = logger.rich.parse(
            display_settings.file_item, expand=False, slotted=True
        )
        file_unknown = logger.rich.parse(
            display_settings.file_unknown, expand=False, slotted=True
        )
        file_info = logger.rich.parse(
            display_settings.file_info, expand=False, slotted=True
        )
        file_dir = logger.rich.parse(
            display_settings.file_dir, expand=False, slotted=True
        )
        file_script = logger.rich.parse(
            display_settings.file_script, expand=False, slotted=True
        )
        file_beatmap = logger.rich.parse(
            display_settings.file_beatmap, expand=False, slotted=True
        )
        file_sound = logger.rich.parse(
            display_settings.file_sound, expand=False, slotted=True
        )
        file_link = logger.rich.parse(
            display_settings.file_link, expand=False, slotted=True
        )
        file_normal = logger.rich.parse(
            display_settings.file_normal, expand=False, slotted=True
        )
        file_other = logger.rich.parse(
            display_settings.file_other, expand=False, slotted=True
        )

        res = []
        for path in file_manager.iterdir():
            # format path name
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

            if isinstance(path, UnrecognizedPath):
                name = file_unknown(name)

            name = file_item(name)

            # format path info
            info = path.info()
            info = (
                logger.rich.parse(info, root_tag=True, expand=False)
                if info is not None
                else None
            )
            info = file_info(info) if info is not None else mu.Text("")

            width = logger.rich.widthof(name.expand())
            res.append((path, width, name, info))

        # sort paths
        field_types = file_manager.get_field_types()

        def key(path):
            ind = (
                field_types.index(type(path))
                if type(path) in field_types
                else len(field_types)
            )
            return (
                isinstance(path, UnrecognizedPath),
                ind,
                path.abs.is_symlink(),
                not path.abs.is_dir(),
                path.abs.suffix,
                path.abs.stem,
            )

        res = sorted(res, key=lambda e: key(e[0]))

        max_width = max((width for _, width, _, _ in res), default=0)
        with logger.print_stack() as print:
            for _, width, name, info in res:
                padding = " " * (max_width - width) if width != -1 else " "
                print(name, end=padding)
                print(info, end="\n")

    @cmd.function_command
    def cd(self, path):
        file_manager = providers.get(FileManager)
        file_manager.cd(path)

    @cmd.function_command
    def cat(self, path):
        file_manager = providers.get(FileManager)
        logger = providers.get(Logger)

        try:
            file_manager.validate_path(path, should_exist=True, file_type="file")
        except InvalidFileOperation as e:
            path_mu = self.as_relative_path(path)
            path_mu = logger.format_path(path_mu)
            logger.print(f"[warn]Failed to access file: {path_mu}[/]")
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        try:
            content = path.abs.read_text()
        except UnicodeDecodeError:
            logger.print("[warn]Failed to read file as text.[/]")
            return

        relpath = file_manager.as_relative_path(path)
        code = logger.format_code(content, title=relpath)
        logger.print(code)

    @cmd.function_command
    def mk(self, path):
        file_manager = providers.get(FileManager)
        file_manager.mk(path)

    @cmd.function_command
    def rm(self, path):
        file_manager = providers.get(FileManager)
        file_manager.rm(path)

    @cmd.function_command
    def mv(self, path, dst):
        file_manager = providers.get(FileManager)
        file_manager.mv(path, dst)

    @cmd.function_command
    def cp(self, src, path):
        file_manager = providers.get(FileManager)
        file_manager.cp(src, path)

    @cd.arg_parser("path")
    def _cd_path_parser(self):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser(
            desc="It should be an existing directory",
            filter=lambda path: os.path.isdir(str(path)),
        )

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser(
            desc="It should be an existing file",
            filter=lambda path: os.path.isfile(str(path)),
        )

    @mv.arg_parser("dst")
    @cp.arg_parser("path")
    @mk.arg_parser("path")
    def _any_path_parser(self, *_, **__):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser()

    @rm.arg_parser("path")
    @mv.arg_parser("path")
    def _lexists_path_parser(self, *_, **__):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser(
            desc="It should be an existing path",
            filter=lambda path: os.path.lexists(str(path)),
        )

    @cp.arg_parser("src")
    def _exists_path_parser(self, *_, **__):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser(
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
        logger = providers.get(Logger)
        logger.clear(bottom)

    @cmd.function_command
    @dn.datanode
    def bye(self):
        """[rich]Close K-AIKO.

        usage: [cmd]bye[/]
        """
        from .profiles import ProfileManager

        logger = providers.get(Logger)

        profile_is_changed = providers.get(ProfileManager).is_changed()

        if profile_is_changed:
            yes = yield from logger.ask(
                "Exit without saving current configuration?"
            ).join()
            if not yes:
                return
        logger.print("Bye~")
        raise KeyboardInterrupt

    @cmd.function_command
    @dn.datanode
    def bye_forever(self):
        """[rich]Clean up all your data and close K-AIKO.

        usage: [cmd]bye_forever[/]
        """
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        logger.print("This command will clean up all your data.")

        yes = yield from logger.ask("Do you really want to do that?", False).join()
        if yes:
            file_manager.remove(logger)
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
        """
        self.root = root
        self.prefix = prefix
        self._desc = desc
        self.filter = filter
        self.suggs = suggs

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

            elif os.path.isfile(currpath) and self.filter(
                self.root.recognize(currpath)
            ):
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

            elif os.path.isfile(suggpath) and self.filter(
                self.root.recognize(suggpath)
            ):
                add(suggestions, sugg + "\000")

        return suggestions

    def info(self, token):
        path = self.parse(token)
        info = path.info_detailed()
        if info is None:
            info = path.info()
        return info


def DirectCdCommand():
    file_commands = FilesCommand()

    def make_cd_command(path, doc):
        command = lambda _: file_commands.cd(path)
        command.__doc__ = doc
        return cmd.function_command(command)

    file_manager = providers.get(FileManager)
    logger = providers.get(Logger)

    attrs = {}

    path = file_manager.recognize(file_manager.current.abs / "..")
    relpath = "../"
    relpath_mu = logger.format_path(relpath)
    doc = f"[rich]change directory to {relpath_mu}\n" + (path.info() or "")
    attrs[".."] = make_cd_command(path, doc)

    for path in file_manager.iterdir():
        if path.slashend:
            relpath = path.abs.name + "/"
            relpath_mu = logger.format_path(relpath)
            doc = f"[rich]change directory to {relpath_mu}\n" + (path.info() or "")
            attrs[relpath] = make_cd_command(path, doc)

    return type("DirectCdCommand", (object,), attrs)()
