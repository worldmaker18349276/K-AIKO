import os
import dataclasses
import traceback
import getpass
import glob
import shutil
from pathlib import Path


@dataclasses.dataclass
class FileManager:
    username: str
    root: Path
    current: Path

    structure = {
        ".": "(The workspace of KAIKO)",
        "Beatmaps": {
            ".": "(The place to hold your beatmaps)",
            "*": {
                ".": "(Beatmapset of a song)",
                "*.kaiko": "(Beatmap file in kaiko format)",
                "*.ka": "(Beatmap file in kaiko format)",
                "*.osu": "(Beatmap file in osu format)",
                "**": "(Inner file of this beatmapset)",
            },
            "*.osz": "(Compressed beatmapset file)",
        },
        "Profiles": {
            ".": "(The place to manage your profiles)",
            "*.kaiko-profile": "(Your custom profile)",
            ".default-profile": "(The file of default profile name)",
        },
        "Resources": {
            ".": "(The place to store some resources of KAIKO)",
            "**": "(Resource file)",
        },
        "Devices": {
            ".": "(The place to manage your devices)",
        },
        "Cache": {
            ".": "(The place to cache some data for better exprience)",
            ".beatshell-history": "(The command history)",
            "**": "(Cache data)",
        },
    }

    @classmethod
    def create(cls):
        username = getpass.getuser()
        root = Path("~/.local/share/K-AIKO").expanduser()
        return cls(username, root, Path("."))

    @property
    def cache_dir(self):
        return self.root / "Cache"

    @property
    def profiles_dir(self):
        return self.root / "Profiles"

    @property
    def beatmaps_dir(self):
        return self.root / "Beatmaps"

    @property
    def resources_dir(self):
        return self.root / "Resources"

    @property
    def devices_dir(self):
        return self.root / "Devices"

    def is_prepared(self):
        if not self.root.exists():
            return False

        if not self.cache_dir.exists():
            return False

        if not self.profiles_dir.exists():
            return False

        if not self.beatmaps_dir.exists():
            return False

        if not self.resources_dir.exists():
            return False

        if not self.devices_dir.exists():
            return False

        return True

    def prepare(self, logger):
        if self.is_prepared():
            return

        # start up
        logger.print("[data/] Prepare your profile...")
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.beatmaps_dir.mkdir(parents=True, exist_ok=True)
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        self.devices_dir.mkdir(parents=True, exist_ok=True)

        logger.print(
            f"[data/] Your data will be stored in {logger.emph(self.root.as_uri())}"
        )
        logger.print(flush=True)

    def remove(self, logger):
        logger.print(
            f"[data/] Remove profiles directory {logger.emph(self.profiles_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.profiles_dir))
        logger.print(
            f"[data/] Remove beatmaps directory {logger.emph(self.beatmaps_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.beatmaps_dir))
        logger.print(
            f"[data/] Remove resources directory {logger.emph(self.resources_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.resources_dir))
        logger.print(
            f"[data/] Remove devices directory {logger.emph(self.resources_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.devices_dir))
        logger.print(
            f"[data/] Remove root directory {logger.emph(self.root.as_uri())}..."
        )
        shutil.rmtree(str(self.root))

    def get_desc(self, path):
        path = Path(os.path.expandvars(os.path.expanduser(path)))
        try:
            abspath = path.resolve(strict=True)
        except Exception:
            return None

        if not abspath.is_relative_to(self.root):
            return None

        if not abspath.exists():
            return None

        route = [abspath, *abspath.parents]
        route = route[route.index(self.root)::-1]

        desc = None
        tree = {glob.escape(str(self.root)): self.structure}
        for current_path in route:
            current_relpath = current_path.relative_to(self.root)

            if not isinstance(tree, dict):
                return None

            for pattern, subtree in tree.items():
                if pattern == ".":
                    continue

                elif pattern == "**":
                    # get description
                    desc_func = subtree
                    desc = (
                        desc_func
                        if desc_func is None or isinstance(desc_func, str)
                        else desc_func(path)
                    )
                    return desc

                else:
                    if not current_path.match(pattern):
                        continue

                    # check file type
                    type_func = (
                        Path.is_dir
                        if isinstance(subtree, dict)
                        else Path.is_file
                    )
                    if not type_func(current_path):
                        continue

                    # get description
                    desc_func = (
                        subtree["."]
                        if isinstance(subtree, dict)
                        else subtree
                    )
                    desc = (
                        desc_func
                        if desc_func is None or isinstance(desc_func, str)
                        else desc_func(current_relpath)
                    )
                    if desc is None:
                        continue

                    tree = subtree
                    break

            else:
                return None

        else:
            return desc

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

            desc = self.get_desc(self.root / self.current / child.name)
            if desc is None:
                name = f"[file_unknown]{name}[/]"
            else:
                name = f"{name}[file_desc]{desc}[/]"
            name = f"[file_item]{name}[/]"
            logger.print(name)

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
