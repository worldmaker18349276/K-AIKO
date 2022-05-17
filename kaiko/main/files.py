import os
import dataclasses
import traceback
import getpass
import glob
import re
import shutil
from pathlib import Path
from ..utils import datanodes as dn
from ..utils import commands as cmd


class InvalidFileOperation(Exception):
    pass


class WildCardDescriptor:
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


@dataclasses.dataclass
class FileManager:
    username: str
    root: Path
    current: Path
    structure: DirDescriptor

    @classmethod
    def create(cls, structure):
        username = getpass.getuser()
        root = Path("~/.local/share/K-AIKO").expanduser()
        return cls(username, root, Path("."), structure)

    def is_prepared(self):
        def go(path, tree):
            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)
                    if not subpath.exists():
                        return False

                    if isinstance(child.descriptor, DirDescriptor):
                        if not subpath.is_dir():
                            raise ValueError(f"bad file structure: {subpath!s} should be a directory")
                        if not go(subpath, child.descriptor):
                            return False
                    elif isinstance(child.descriptor, FileDescriptor):
                        if not subpath.is_file():
                            raise ValueError(f"bad file structure: {subpath!s} should be a file")
                    else:
                        raise TypeError(child.descriptor)

        if not self.root.exists():
            return False

        return go(self.root, self.structure)

    def prepare(self, logger):
        if self.is_prepared():
            return

        # start up
        logger.print("[data/] Prepare your profile...")

        def go(path, tree):
            for child in tree.children():
                if child.is_required:
                    subpath = path / unescape_glob(child.pattern)

                    if isinstance(child.descriptor, DirDescriptor):
                        if not subpath.exists():
                            subpath.mkdir()
                        go(subpath, child.descriptor)
                    elif isinstance(child.descriptor, FileDescriptor):
                        if not subpath.exists():
                            subpath.touch()
                    else:
                        raise TypeError(child.descriptor)

        if not self.root.exists():
            self.root.mkdir()

        go(self.root, self.structure)

        logger.print(
            f"[data/] Your data will be stored in {logger.emph(self.root.as_uri())}"
        )
        logger.print(flush=True)

    def remove(self, logger):
        shutil.rmtree(str(self.root))

    def get_desc(self, path, ret_ind=False):
        if not ret_ind:
            return self.get_desc(path, ret_ind=True)[1]

        index, expanded_path, descriptor = self.glob(path)
        if descriptor is None:
            return (), None

        return index, descriptor.desc(expanded_path)

    def glob(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = path.resolve(strict=True)
        except Exception:
            return (), None, None

        if not abspath.is_relative_to(self.root):
            return (), None, None

        if not abspath.exists():
            return (), None, None

        route = [abspath, *abspath.parents]
        route = route[route.index(self.root)::-1][1:]

        desc_func = None
        index = ()
        tree = self.structure
        for current_path in route:
            if not isinstance(tree, DirDescriptor):
                return (), None, None

            for i, child in enumerate(tree.children()):
                if child.pattern == "**":
                    current_path = path

                if isinstance(child.descriptor, DirDescriptor):
                    if not Path.is_dir(current_path):
                        continue
                elif isinstance(child.descriptor, FileDescriptor):
                    if not Path.is_file(current_path):
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
                    tree = child.descriptor
                    break

            else:
                return (), None, None

        return index, path, tree

    def cd(self, path, logger):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = (self.root / self.current / path).resolve(strict=True)
        except Exception:
            logger.print("[warn]Filed to resolve path[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return

        if not abspath.exists():
            logger.print("[warn]No such directory[/]")
            return
        if not abspath.is_relative_to(self.root):
            logger.print("[warn]Out of root directory[/]")
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
        res = []
        for child in (self.root / self.current).resolve().iterdir():
            if child.is_symlink():
                name = logger.escape(str(child.readlink()), type="all")
            else:
                name = logger.escape(child.name, type="all")

            if child.is_dir():
                name = f"[file_dir]{name}[/]"

            elif child.is_file():
                if child.suffix in [".py", ".kaiko-profile"]:
                    name = f"[file_script]{name}[/]"
                elif child.suffix in [".ka", ".kaiko", ".osu"]:
                    name = f"[file_beatmap]{name}[/]"
                elif child.suffix in [".wav", ".mp3", ".mp4", ".m4a", ".ogg"]:
                    name = f"[file_sound]{name}[/]"
                else:
                    name = f"[file_normal]{name}[/]"

            else:
                name = f"[file_other]{name}[/]"

            if child.is_symlink():
                linkname = logger.escape(child.name, type="all")
                name = f"[file_link]{linkname}[/]{name}"

            ind, path, descriptor = self.glob(self.root / self.current / child.name)
            desc = descriptor.desc(path) if descriptor is not None else None

            ordering_key = (descriptor is None, ind, child.is_symlink(), not child.is_dir(), child.suffix, child.stem)

            if descriptor is None:
                name = f"[file_unknown]{name}[/]"
            desc = f"[file_desc]{desc}[/]" if desc is not None else ""
            name = f"[file_item]{name}[/]"

            name = logger.rich.parse(name)
            width = logger.rich.widthof(name)
            desc = logger.rich.parse(desc)

            res.append((ordering_key, width, name, desc))

        res = sorted(res, key=lambda e: e[0])
        max_width = max((width for _, width, _, _ in res), default=0)

        for _, width, name, desc in res:
            padding = " "*(max_width - width) if width != -1 else " "
            logger.print(name, end=padding)
            logger.print(desc, end="\n")

    def get(self, path, logger):
        try:
            abspath = (self.root / self.current / path).resolve(strict=True)
        except Exception:
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        if not abspath.exists():
            logger.print("[warn]No such file[/]")
            return
        if not abspath.is_relative_to(self.root):
            logger.print("[warn]Out of root directory[/]")
            return
        if not abspath.is_file():
            logger.print("[warn]Is not a file[/]")
            return

        return abspath


class FilesCommand:
    def __init__(self, menu):
        self.menu = menu

    @cmd.function_command
    def cd(self, path):
        self.menu.file_manager.cd(path, self.menu.logger)

    @cmd.function_command
    def ls(self):
        self.menu.file_manager.ls(self.menu.logger)

    @cmd.function_command
    def cat(self, path):
        abspath = self.menu.file_manager.get(path, self.menu.logger)

        try:
            content = abspath.read_text()
        except UnicodeDecodeError:
            self.menu.logger.print("[warn]Cannot read binary file.[/]")
            return

        code = self.menu.logger.format_code(
            content, title=str(self.menu.file_manager.current / path)
        )
        self.menu.logger.print(code)

    @cd.arg_parser("path")
    def _cd_path_parser(self):
        return cmd.PathParser(self.menu.file_manager.root / self.menu.file_manager.current, type="dir")

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        return cmd.PathParser(self.menu.file_manager.root / self.menu.file_manager.current, type="file")

    @cmd.function_command
    def clean(self, bottom: bool = False):
        """[rich]Clean screen.

        usage: [cmd]clean[/] [[[kw]--bottom[/] [arg]{BOTTOM}[/]]]
                                   ╲
                          bool, move to bottom or
                           not; default is False.
        """
        self.menu.logger.clear(bottom)

    @cmd.function_command
    @dn.datanode
    def bye(self):
        """[rich]Close K-AIKO.

        usage: [cmd]bye[/]
        """
        if self.menu.profiles_manager.is_changed():
            yes = yield from self.menu.logger.ask(
                "Exit without saving current configuration?"
            ).join()
            if not yes:
                return
        self.menu.logger.print("Bye~")
        raise KeyboardInterrupt

    @cmd.function_command
    @dn.datanode
    def bye_forever(self):
        """[rich]Clean up all your data and close K-AIKO.

        usage: [cmd]bye_forever[/]
        """
        logger = self.menu.logger

        logger.print("This command will clean up all your data.")

        yes = yield from logger.ask("Do you really want to do that?", False).join()
        if yes:
            self.menu.file_manager.remove(logger)
            logger.print("Good luck~")
            raise KeyboardInterrupt


def CdCommand(menu):
    def make_command(name):
        return cmd.function_command(lambda self: self.menu.file_manager.cd(name, self.menu.logger))

    attrs = {}
    attrs["menu"] = menu
    attrs[".."] = make_command("..")
    for child in (menu.file_manager.root / menu.file_manager.current).resolve().iterdir():
        if child.is_dir():
            attrs[child.name + "/"] = make_command(child.name)

    return type("CdCommand", (object,), attrs)()
