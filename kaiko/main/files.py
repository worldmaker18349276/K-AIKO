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

    @classmethod
    def create(cls):
        username = getpass.getuser()
        root = Path("~/.local/share/K-AIKO").expanduser()
        return cls(username, root, Path("."))

    @property
    def cache_dir(self):
        return self.root / "cache"

    @property
    def profiles_dir(self):
        return self.root / "profiles"

    @property
    def beatmaps_dir(self):
        return self.root / "beatmaps"

    @property
    def resources_dir(self):
        return self.root / "resources"

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

    def cd(self, path, logger):
        try:
            abspath = (self.root / self.current / path).resolve(strict=True)
        except Exception:
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

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
        abspath = self.root / self.current
        for abschild in abspath.iterdir():
            child = logger.escape(str(abschild.relative_to(abspath)))

            if child.startswith("."):
                child = f"[file_hidden]{child}[/]"

            elif abschild.is_symlink():
                child = f"[file_link]{child}[/]"

            elif abschild.is_dir():
                child = f"[file_dir]{child}[/]"

            elif abschild.is_file():
                if abschild.suffix in [".py", ".kaiko-profile"]:
                    child = f"[file_script]{child}[/]"
                elif abschild.suffix in [".ka", ".kaiko", ".osu"]:
                    child = f"[file_beatmap]{child}[/]"
                elif abschild.suffix in [".wav", ".mp3", ".mp4", ".m4a", ".ogg"]:
                    child = f"[file_sound]{child}[/]"
                else:
                    child = f"[file_normal]{child}[/]"

            else:
                child = f"[file_other]{child}[/]"

            logger.print(child)

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
