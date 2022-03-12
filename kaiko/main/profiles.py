import os
import traceback
import tempfile
import subprocess
from pathlib import Path
from ..utils import config as cfg
from ..utils import parsec as pc
from ..utils import commands as cmd
from ..utils import datanodes as dn
from ..devices import engines
from ..beats import beatshell
from ..beats import beatmaps


class KAIKOSettings(cfg.Configurable):
    devices = cfg.subconfig(engines.DevicesSettings)
    shell = cfg.subconfig(beatshell.BeatShellSettings)
    gameplay = cfg.subconfig(beatmaps.GameplaySettings)


def exists(program):
    if os.name == "nt":
        rc = subprocess.call(
            ["where", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        rc = subprocess.call(
            ["which", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return rc == 0


@dn.datanode
def edit(text, editor, suffix=""):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".tmp" + suffix) as file:
        file.write(text)
        file.flush()

        yield from dn.subprocess_task([editor, file.name]).join()

        return open(file.name, mode="r").read()


class ProfileManager:
    """Profile manager for Configurable object.

    Attributes
    ----------
    logger : loggers.Logger
    path : Path
        The path of profiles directory.
    profiles : list of str
        A list of names of profiles.
    default_name : str or None
        The name of default profile.
    current_name : str
        The name of current profile.
    current : config.Configurable
        The current configuration.
    """

    default_meta = ".default-profile"
    extension = ".py"
    settings_name = "settings"
    config_type = KAIKOSettings

    def __init__(self, path, logger):
        if isinstance(path, str):
            path = Path(path)

        self.logger = logger
        self.path = path
        self.profiles = []
        self.default_name = None
        self.current_name = None
        self.current = None
        self._current_mtime = None
        self._profiles_mtime = None
        self.on_change_handlers = []

    def on_change(self, on_change_handler):
        self.on_change_handlers.append(on_change_handler)

    # config manipulation

    def set(self, fields, value):
        return cfg.set(self.current, fields, value)

    def unset(self, fields):
        return cfg.unset(self.current, fields)

    def get(self, fields):
        return cfg.get(self.current, fields)

    def has(self, fields):
        return cfg.has(self.current, fields)

    def get_default(self, fields):
        return cfg.get_default(self.current, fields)

    def has_default(self, fields):
        return cfg.has_default(self.current, fields)

    def get_title(self):
        return self.current_name + self.extension

    def format(self):
        return cfg.format(self.config_type, self.current, name=self.settings_name)

    # profiles management

    def is_uptodate(self):
        if not self.path.exists():
            return False
        return self._profiles_mtime == os.stat(str(self.path)).st_mtime

    def is_changed(self):
        current_path = self.path / (self.current_name + self.extension)
        if not current_path.exists():
            return True
        return self._current_mtime != os.stat(str(current_path)).st_mtime

    def set_as_changed(self):
        self._current_mtime = None
        for on_change_handler in self.on_change_handlers:
            on_change_handler(self.current)

    def update(self):
        """Update the list of profiles.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        logger.print("[data/] Update profiles...")

        if not self.path.exists():
            logger.print(
                f"[warn]The profile directory doesn't exist: {logger.emph(self.path.as_uri())}[/]"
            )
            return False

        if not self.path.is_dir():
            logger.print(
                f"[warn]Wrong file type for profile directory: {logger.emph(self.path.as_uri())}[/]"
            )
            return False

        profiles_mtime = os.stat(str(self.path)).st_mtime

        # update default_name
        default_meta_path = self.path / self.default_meta
        if default_meta_path.exists():
            if not default_meta_path.is_file():
                logger.print(
                    f"[warn]Wrong file type for default profile: {logger.emph(default_meta_path.as_uri())}[/]"
                )
                return False
            self.default_name = default_meta_path.read_text().rstrip("\n")

        # update profiles
        self.profiles = [
            subpath.stem
            for subpath in self.path.iterdir()
            if subpath.suffix == self.extension
        ]
        self._profiles_mtime = profiles_mtime
        return True

    def set_default(self):
        """Set the current profile as default.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        logger.print(
            f"[data/] Set {logger.emph(self.current_name)} as the default profile..."
        )

        if not self.path.exists():
            logger.print(
                f"[warn]No such profile directory: {logger.emph(self.path.as_uri())}[/]"
            )
            return False

        default_meta_path = self.path / self.default_meta
        if default_meta_path.exists() and not default_meta_path.is_file():
            logger.print(
                f"[warn]Wrong file type for default profile: {logger.emph(default_meta_path.as_uri())}[/]"
            )
            return False

        default_meta_path.write_text(self.current_name)
        self.default_name = self.current_name

        return True

    def save(self):
        """Save the current configuration.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        current_path = self.path / (self.current_name + self.extension)
        logger.print(
            f"[data/] Save configuration to {logger.emph(current_path.as_uri())}..."
        )

        if not self.path.exists():
            logger.print(
                f"[warn]The profile directory doesn't exist: {logger.emph(self.path.as_uri())}[/]"
            )
            return False

        if current_path.exists() and not current_path.is_file():
            logger.print(
                f"[warn]Wrong file type for profile: {logger.emph(current_path.as_uri())}[/]"
            )
            return False

        try:
            cfg.write(
                self.config_type, self.current, current_path, name=self.settings_name
            )
        except Exception:
            logger.print("[warn]Fail to format configuration[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return False

        self._current_mtime = os.stat(str(current_path)).st_mtime
        self.update()
        return True

    def load(self):
        """Load the current profile.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        current_path = self.path / (self.current_name + self.extension)
        logger.print(
            f"[data/] Load configuration from {logger.emph(current_path.as_uri())}..."
        )

        if not current_path.exists():
            logger.print(
                f"[warn]The profile doesn't exist: {logger.emph(current_path.as_uri())}[/]"
            )
            return False

        if not current_path.is_file():
            logger.print(
                f"[warn]Wrong file type for profile: {logger.emph(current_path.as_uri())}[/]"
            )
            return False

        current_mtime = os.stat(str(current_path)).st_mtime
        try:
            self.current = cfg.read(
                self.config_type, current_path, name=self.settings_name
            )
        except Exception:
            logger.print("[warn]Fail to parse configuration[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return False

        self.set_as_changed()
        self._current_mtime = current_mtime
        return True

    def use(self, name=None):
        """change the current profile.

        Parameters
        ----------
        name : str, optional
            The name of profile, or None for default.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        if name is None:
            if self.default_name is None:
                logger.print("[warn]No default profile[/]")
                return False

            name = self.default_name

        if name not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(name)}[/]")
            return False

        old_name = self.current_name
        self.current_name = name
        succ = self.load()
        if not succ:
            self.current_name = old_name
            return False
        return True

    def new(self, name=None, clone=None):
        """make a new profile.

        Parameters
        ----------
        name : str, optional
            The name of profile.
        clone : str, optional
            The name of profile to clone.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        logger.print("Make new profile...")

        if clone is not None and clone not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(clone)}[/]")
            return False

        if isinstance(name, str) and (not name.isprintable() or "/" in name):
            logger.print(f"[warn]Invalid profile name: {logger.emph(name)}[/]")
            return False

        if name in self.profiles:
            logger.print(
                f"[warn]This profile name {logger.emph(name)} already exists.[/]"
            )
            return False

        if name is None:
            name = "new profile"
            n = 1
            while name in self.profiles:
                n += 1
                name = f"new profile ({str(n)})"
            logger.print(f"Create profile with name {logger.emph(name)}.")

        if clone is None:
            self.current_name = name
            self.current = self.config_type()
            self.set_as_changed()

        else:
            old_name = self.current_name
            self.current_name = clone
            succ = self.load()
            if not succ:
                self.current_name = old_name
                return False
            self.current_name = name

        return True

    def delete(self, name):
        """Delete a profile.

        Parameters
        ----------
        name : str
            The name of profile to delete.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        target_path = self.path / (name + self.extension)
        logger.print(f"[data/] Delete profile {logger.emph(target_path.as_uri())}...")

        if name not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(name)}[/]")
            return False

        if target_path.exists():
            if not target_path.is_file():
                logger.print(
                    f"[warn]Wrong file type for profile: {logger.emph(target_path.as_uri())}[/]"
                )
                return False
            target_path.unlink()

        self.profiles.remove(name)
        return True

    def rename(self, name):
        """Rename the current profile.

        Parameters
        ----------
        name : str
            The new name of profile.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        if self.current_name == name:
            return True

        current_path = self.path / (self.current_name + self.extension)
        target_path = self.path / (name + self.extension)
        current_name = logger.emph(current_path.as_uri())
        target_name = logger.emph(target_path.as_uri())
        logger.print(f"[data/] Rename profile {current_name} to {target_name}...")

        if not name.isprintable() or "/" in name:
            logger.print(f"[warn]Invalid profile name: {logger.emph(name)}[/]")
            return False

        if name in self.profiles:
            logger.print(
                f"[warn]This profile name {logger.emph(name)} already exists.[/]"
            )
            return False

        if self.current_name in self.profiles:
            if current_path.exists():
                current_path.rename(target_path)

            self.profiles.remove(self.current_name)
            self.profiles.append(name)

        if self.current_name == self.default_name:
            self.current_name = name
            self.set_default()
        else:
            self.current_name = name

        return True


class FieldParser(cmd.TreeParser):
    def __init__(self, config_type):
        self.config_type = config_type

        def make_tree(fields):
            tree = {}
            for name, value in fields.items():
                if isinstance(value, dict):
                    tree[name + "."] = make_tree(value)
                else:
                    tree[name] = lambda token: tuple(token.split("."))
            return tree

        super().__init__(make_tree(self.config_type.__configurable_fields__))

    def info(self, token):
        field = self.parse(token)
        return self.config_type.get_field_doc(field)


class ProfilesCommand:
    def __init__(self, profiles, logger):
        self.profiles = profiles
        self.logger = logger

    # configuration

    @cmd.function_command
    def show(self):
        """[rich]Show the current configuration.

        usage: [cmd]profiles[/] [cmd]show[/]
        """
        text = self.profiles.format()
        is_changed = self.profiles.is_changed()
        title = self.profiles.get_title()
        self.logger.print(
            self.logger.format_code(text, title=title, is_changed=is_changed)
        )

    @cmd.function_command
    def show_diff(self):
        """[rich]Show the difference of configuration.

        usage: [cmd]profiles[/] [cmd]show_diff[/]
        """
        current_path = self.profiles.path / (
            self.profiles.current_name + self.profiles.extension
        )
        if not current_path.exists() or not current_path.is_file():
            old = ""
        else:
            old = open(current_path, "r").read()

        text = self.profiles.format()
        is_changed = self.profiles.is_changed()
        title = self.profiles.get_title()
        self.logger.print(
            self.logger.format_code_diff(old, text, title=title, is_changed=is_changed)
        )

    @cmd.function_command
    def has(self, field):
        """[rich]Check whether this field is set in the configuration.

        usage: [cmd]profiles[/] [cmd]has[/] [arg]{field}[/]
                              ╱
                       The field name.
        """
        return self.profiles.has(field)

    @cmd.function_command
    def get(self, field):
        """[rich]Get the value of this field in the configuration.

        usage: [cmd]profiles[/] [cmd]get[/] [arg]{field}[/]
                              ╱
                       The field name.
        """
        if not self.profiles.has(field) and not self.profiles.has_default(field):
            self.logger.print(f"[warn]No value for field {'.'.join(field)}[/]")
            return
        return self.profiles.get(field)

    @cmd.function_command
    def set(self, field, value):
        """[rich]Set this field in the configuration.

        usage: [cmd]profiles[/] [cmd]set[/] [arg]{field}[/] [arg]{value}[/]
                              ╱         ╲
                     The field name.   The value.
        """
        self.profiles.set(field, value)
        self.profiles.set_as_changed()

    @cmd.function_command
    def unset(self, field):
        """[rich]Unset this field in the configuration.

        usage: [cmd]profiles[/] [cmd]unset[/] [arg]{field}[/]
                                ╱
                         The field name.
        """
        self.profiles.unset(field)
        self.profiles.set_as_changed()

    @cmd.function_command
    @dn.datanode
    def edit(self):
        """[rich]Edit the configuration by external editor.

        usage: [cmd]profiles[/] [cmd]edit[/]
        """
        title = self.profiles.get_title()

        editor = self.profiles.current.devices.terminal.editor

        text = cfg.format(
            self.profiles.config_type,
            self.profiles.current,
            self.profiles.settings_name,
        )

        yield

        # open editor
        if not exists(editor):
            self.logger.print(f"[warn]Unknown editor: {self.logger.escape(editor)}[/]")
            return

        edited_text = yield from edit(text, editor, ".py").join()

        self.logger.print(
            self.logger.format_code_diff(
                text, edited_text, title=title, is_changed=True
            )
        )

        # parse result
        try:
            res = cfg.parse(
                self.profiles.config_type, edited_text, self.profiles.settings_name
            )

        except pc.ParseError as error:
            line, col = pc.ParseError.locate(error.text, error.index)
            code = error.text[: error.index] + "◊" + error.text[error.index :]

            self.logger.print(
                f"[warn]parse fail at ln {line}, col {col} (marked with ◊)[/]"
            )
            self.logger.print(self.logger.format_code(code))
            if error.__cause__ is not None:
                self.logger.print(
                    f"[warn]{self.logger.escape(str(error.__cause__))}[/]"
                )

        except:
            self.logger.print(f"[warn]An unexpected error occurred[/]")
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)

        else:
            self.profiles.current = res
            self.profiles.set_as_changed()

    @get.arg_parser("field")
    @has.arg_parser("field")
    @unset.arg_parser("field")
    @set.arg_parser("field")
    def _field_parser(self):
        return FieldParser(self.profiles.config_type)

    @set.arg_parser("value")
    def _set_value_parser(self, field):
        annotation = self.profiles.config_type.get_field_type(field)
        default = self.profiles.get(field)
        return cmd.LiteralParser(annotation, default)

    # profiles

    @cmd.function_command
    def list(self):
        """[rich]Show all profiles.

        usage: [cmd]profiles[/] [cmd]list[/]
        """
        logger = self.logger

        if not self.profiles.is_uptodate():
            self.profiles.update()

        for profile in self.profiles.profiles:
            note = ""
            if profile == self.profiles.default_name:
                note += " (default)"
            if profile == self.profiles.current_name:
                note += " (current)"
            logger.print(logger.emph(profile + self.profiles.extension) + note)

    @cmd.function_command
    def reload(self):
        """[rich]Reload the configuration.

        usage: [cmd]profiles[/] [cmd]reload[/]
        """
        logger = self.logger

        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.load()

    @cmd.function_command
    def save(self):
        """[rich]Save the configuration.

        usage: [cmd]profiles[/] [cmd]save[/]
        """
        logger = self.logger

        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.save()

    @cmd.function_command
    def set_default(self):
        """[rich]Set the current profile as default.

        usage: [cmd]profiles[/] [cmd]set_default[/]
        """
        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.set_default()

    @cmd.function_command
    def use(self, profile):
        """[rich]Change the current profile.

        usage: [cmd]profiles[/] [cmd]use[/] [arg]{profile}[/]
                                ╱
                       The profile name.
        """

        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.use(profile)

    @cmd.function_command
    def rename(self, profile):
        """[rich]Rename the current profile.

        usage: [cmd]profiles[/] [cmd]rename[/] [arg]{profile}[/]
                                  ╱
                        The profile name.
        """
        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.rename(profile)

    @cmd.function_command
    def new(self, profile, clone=None):
        """[rich]Make a new profile.

        usage: [cmd]profiles[/] [cmd]new[/] [arg]{profile}[/] [[[kw]--clone[/] [arg]{PROFILE}[/]]]
                                ╱                    ╲
                       The profile name.      The profile to be cloned.
        """
        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.new(profile, clone)

    @cmd.function_command
    def delete(self, profile):
        """[rich]Delete a profile.

        usage: [cmd]profiles[/] [cmd]delete[/] [arg]{profile}[/]
                                  ╱
                         The profile name.
        """
        if not self.profiles.is_uptodate():
            self.profiles.update()

        self.profiles.delete(profile)

    @rename.arg_parser("profile")
    @new.arg_parser("profile")
    def _new_profile_parser(self):
        return cmd.RawParser()

    @new.arg_parser("clone")
    @use.arg_parser("profile")
    @delete.arg_parser("profile")
    def _old_profile_parser(self, *_, **__):
        return cmd.OptionParser(
            self.profiles.profiles,
            desc="It should be the name of the profile that exists in the configuration.",
        )
