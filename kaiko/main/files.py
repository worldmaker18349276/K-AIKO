import os
import dataclasses
import traceback
import getpass
import shutil
from pathlib import Path


@dataclasses.dataclass
class FileManager:
    username: str
    root: Path
    current: Path

    structure = {
        "Beatmaps": {
            "*": { "**": True },
            "*.osz": True,
        },
        "Profiles": {
            "*.kaiko-profile": True,
            ".default-profile": True,
        },
        "Resources": { "**": True },
        "Cache": { "**": True },
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
            f"[data/] Remove root directory {logger.emph(self.root.as_uri())}..."
        )
        shutil.rmtree(str(self.root))

    def is_known(self, path):
        try:
            abspath = path.resolve(strict=True)
        except Exception:
            return False

        if not abspath.is_relative_to(self.root):
            return False

        if not abspath.exists():
            return False

        def go(parent=self.root, children=self.structure):
            if abspath.is_dir() and path.match(str(parent)):
                return True
            for name, subtree in children.items():
                if isinstance(subtree, dict):
                    if go(parent / name, subtree):
                        return True
                elif name == "**":
                    for parentpath in [path, *path.parents]:
                        if parentpath.match(str(parent / "*")):
                            return subtree == True or subtree(path)
                        if parentpath == self.root:
                            break
                else:
                    if abspath.is_file() and path.match(str(parent / name)):
                        return subtree == True or subtree(path)
            return False

        return go()

    def cd(self, path, logger):
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

        # don't resolve symlink
        self.current = Path(os.path.normpath(str(self.current / path)))

    def ls(self, logger):
        for child in (self.root / self.current).resolve().iterdir():
            name = logger.escape(child.name)

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

            if not self.is_known(self.root / self.current / child.name):
                name = f"[file_unknown]{name}[/]"
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
