import os
import dataclasses
import traceback
import getpass
import glob
import shutil
from pathlib import Path
from ..utils import datanodes as dn
from ..utils import commands as cmd


@dataclasses.dataclass
class FileManager:
    username: str
    root: Path
    current: Path
    structure: dict
    initializer: dict

    @classmethod
    def create(cls, structure, initializer):
        username = getpass.getuser()
        root = Path("~/.local/share/K-AIKO").expanduser()
        return cls(username, root, Path("."), structure, initializer)

    def is_prepared(self):
        if not self.root.exists():
            return False

        def go(path, tree):
            for key, value in tree.items():
                if key == ".":
                    continue

                subpath = path / key

                if not subpath.exists():
                    return False

                if isinstance(value, dict):
                    if not subpath.is_dir():
                        return False
                    if not go(subpath, value):
                        return False
                else:
                    if not subpath.is_file():
                        return False

        return go(self.root, self.initializer)

    def prepare(self, logger):
        if self.is_prepared():
            return

        # start up
        logger.print("[data/] Prepare your profile...")

        if not self.root.exists():
            self.root.mkdir()
            init = self.initializer.get(".", None)
            if init:
                init(self.root)

        def go(path, tree):
            for key, value in tree.items():
                if key == ".":
                    continue

                subpath = path / key

                if not subpath.exists():
                    if isinstance(value, dict):
                        subpath.mkdir()
                        init = value.get(".", None)
                        if init:
                            init(subpath)
                        go(subpath, value)
                    else:
                        subpath.touch()
                        if value:
                            value(subpath)

        go(self.root, self.initializer)

        logger.print(
            f"[data/] Your data will be stored in {logger.emph(self.root.as_uri())}"
        )
        logger.print(flush=True)

    def remove(self, logger):
        shutil.rmtree(str(self.root))

    def get_desc(self, path, ret_ind=False):
        if not ret_ind:
            return self.get_desc(path, ret_ind=True)[1]

        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = path.resolve(strict=True)
        except Exception:
            return (), None

        if not abspath.is_relative_to(self.root):
            return (), None

        if not abspath.exists():
            return (), None

        route = [abspath, *abspath.parents]
        route = route[route.index(self.root)::-1]

        desc_func = None
        index = ()
        tree = {glob.escape(str(self.root)): self.structure}
        for current_path in route:
            if not isinstance(tree, dict):
                return (), None

            for i, (pattern, subtree) in enumerate(tree.items()):
                if pattern == ".":
                    continue

                elif pattern == "**":
                    desc_func = subtree
                    curr_index = i
                    break

                elif isinstance(subtree, dict):
                    if not Path.is_dir(current_path):
                        continue

                    if not current_path.match(pattern):
                        continue

                    desc_func = subtree["."]
                    curr_index = i
                    break

                else:
                    if not Path.is_file(current_path):
                        continue

                    if not current_path.match(pattern):
                        continue

                    desc_func = subtree
                    curr_index = i
                    break

            else:
                desc_func = None
                curr_index = None
                continue

            tree = subtree
            index = (*index, curr_index)

        desc = (
            desc_func
            if desc_func is None or isinstance(desc_func, str)
            else desc_func(path.relative_to(self.root))
        )
        return index, desc

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

            ind, desc = self.get_desc(self.root / self.current / child.name, ret_ind=True)
            ordering_key = (desc is None, ind, child.is_symlink(), not child.is_dir(), child.suffix, child.stem)

            if desc is None:
                name = f"[file_unknown]{name}[/]"
                desc = ""
            else:
                desc = f"[file_desc]{desc}[/]"
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
