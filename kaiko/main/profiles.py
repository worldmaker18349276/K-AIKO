import os
import traceback
import tempfile
import subprocess
from pathlib import Path
from ..utils import config as cfg
from ..utils import parsec as pc
from ..utils import commands as cmd
from ..utils import datanodes as dn
from .loggers import Logger
from .files import FileDescriptor, DirDescriptor, as_child


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


class ProfilesDirDescriptor(DirDescriptor):
    "(The place to manage your profiles)"

    @as_child("*.kaiko-profile")
    class Profile(FileDescriptor):
        def desc(self, path):
            profile_manager = self.provider.get(ProfileManager)
            note = "(Untracked custom profile)"
            for profile in profile_manager.profiles:
                name = profile + profile_manager.extension
                if name == path.name:
                    note = "(Your custom profile)"
                    if profile == profile_manager.default_name:
                        note += " (default)"
                    if profile == profile_manager.current_name:
                        note += " (current)"
                    break
            return note

        def mk(self, path):
            profile_manager = self.provider.get(ProfileManager)
            logger = self.provider.get(Logger)

            if not profile_manager.is_uptodate():
                profile_manager.update(logger)
            profile_manager.make_empty(logger, name=path.stem)
            profile_manager.update(logger)

        def rm(self, path):
            profile_manager = self.provider.get(ProfileManager)
            logger = self.provider.get(Logger)

            if not profile_manager.is_uptodate():
                profile_manager.update(logger)
            profile_manager.delete(logger, name=path.stem)
            profile_manager.update(logger)

        def mv(self, path, dst):
            profile_manager = self.provider.get(ProfileManager)
            logger = self.provider.get(Logger)

            if not profile_manager.is_uptodate():
                profile_manager.update(logger)
            profile_manager.rename(logger, name=path.stem, newname=dst.stem)
            profile_manager.update(logger)

    @as_child(".default-profile")
    class Default(FileDescriptor):
        "(The file of default profile name)"


class ProfileManager:
    """Profile manager for Configurable object.

    Attributes
    ----------
    config_type : cfg.Configurable
        The configuration type to manage.
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
    extension = ".kaiko-profile"
    settings_name = "settings"

    def __init__(self, config_type, profiles_dir):
        if isinstance(profiles_dir, str):
            profiles_dir = Path(profiles_dir)

        self.config_type = config_type
        self.profiles_dir = profiles_dir
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
        if not self.profiles_dir.exists():
            return False
        return self._profiles_mtime == os.stat(str(self.profiles_dir)).st_mtime

    def is_changed(self):
        current_path = self.profiles_dir / (self.current_name + self.extension)
        if not current_path.exists():
            return True
        return self._current_mtime != os.stat(str(current_path)).st_mtime

    def set_as_changed(self):
        self._current_mtime = None
        for on_change_handler in self.on_change_handlers:
            on_change_handler(self.current)

    def update(self, logger):
        """Update the list of profiles.

        Parameters
        ----------
        logger : loggers.Logger

        Returns
        -------
        succ : bool
        """
        logger.print("[data/] Update profiles...")

        if not self.profiles_dir.exists():
            logger.print(
                f"[warn]The profile directory doesn't exist: {logger.emph(self.profiles_dir.as_uri())}[/]"
            )
            return False

        if not self.profiles_dir.is_dir():
            logger.print(
                f"[warn]Wrong file type for profile directory: {logger.emph(self.profiles_dir.as_uri())}[/]"
            )
            return False

        profiles_mtime = os.stat(str(self.profiles_dir)).st_mtime

        # update default_name
        default_meta_path = self.profiles_dir / self.default_meta
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
            for subpath in self.profiles_dir.iterdir()
            if subpath.suffix == self.extension
        ]
        self._profiles_mtime = profiles_mtime
        return True

    def set_default(self, logger, name=None):
        """Set the current profile as default.

        Parameters
        ----------
        logger : loggers.Logger
        name : str, optional

        Returns
        -------
        succ : bool
        """
        name = name if name is not None else self.current_name
        logger.print(
            f"[data/] Set {logger.emph(name, type='all')} as the default profile..."
        )

        if not self.profiles_dir.exists():
            logger.print(
                f"[warn]No such profile directory: {logger.emph(self.profiles_dir.as_uri())}[/]"
            )
            return False

        if name not in self.profiles:
            logger.print(
                f"[warn]This profile {logger.emph(name, type='all')} doesn't exist.[/]"
            )
            return False

        default_meta_path = self.profiles_dir / self.default_meta
        if default_meta_path.exists() and not default_meta_path.is_file():
            logger.print(
                f"[warn]Wrong file type for default profile: {logger.emph(default_meta_path.as_uri())}[/]"
            )
            return False

        default_meta_path.write_text(name)
        self.default_name = name

        return True

    def save(self, logger):
        """Save the current configuration.

        Parameters
        ----------
        logger : loggers.Logger

        Returns
        -------
        succ : bool
        """
        current_path = self.profiles_dir / (self.current_name + self.extension)
        logger.print(
            f"[data/] Save configuration to {logger.emph(current_path.as_uri())}..."
        )

        if not self.profiles_dir.exists():
            logger.print(
                f"[warn]The profile directory doesn't exist: {logger.emph(self.profiles_dir.as_uri())}[/]"
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
        self.update(logger)
        return True

    def load(self, logger):
        """Load the current profile.

        Parameters
        ----------
        logger : loggers.Logger

        Returns
        -------
        succ : bool
        """
        current_path = self.profiles_dir / (self.current_name + self.extension)
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

    def use(self, logger, name=None):
        """change the current profile.

        Parameters
        ----------
        logger : loggers.Logger
        name : str, optional
            The name of profile, or None for default.

        Returns
        -------
        succ : bool
        """
        if name is None:
            if self.default_name is None:
                logger.print("[warn]No default profile[/]")
                return False

            name = self.default_name

        if name not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(name, type='all')}[/]")
            return False

        old_name = self.current_name
        self.current_name = name
        succ = self.load(logger)
        if not succ:
            self.current_name = old_name
            return False
        return True

    def new(self, logger, name=None, clone=None):
        """make a new profile.

        Parameters
        ----------
        logger : loggers.Logger
        name : str, optional
            The name of profile.
        clone : str, optional
            The name of profile to clone.

        Returns
        -------
        succ : bool
        """
        logger.print("Make new profile...")

        if clone is not None and clone not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(clone, type='all')}[/]")
            return False

        if isinstance(name, str) and (not name.isprintable() or "/" in name):
            logger.print(f"[warn]Invalid profile name: {logger.emph(name, type='all')}[/]")
            return False

        if name in self.profiles:
            logger.print(
                f"[warn]This profile name {logger.emph(name, type='all')} already exists.[/]"
            )
            return False

        if name is None:
            name = "new profile"
            n = 1
            while name in self.profiles:
                n += 1
                name = f"new profile ({str(n)})"
            logger.print(f"Create profile with name {logger.emph(name, type='all')}.")

        if clone is None:
            self.current_name = name
            self.current = self.config_type()
            self.set_as_changed()

        else:
            old_name = self.current_name
            self.current_name = clone
            succ = self.load(logger)
            if not succ:
                self.current_name = old_name
                return False
            self.current_name = name

        return True

    def make_empty(self, logger, name):
        """Make an empty profile.

        Parameters
        ----------
        logger : loggers.Logger
        name : str
            The name of profile.

        Returns
        -------
        succ : bool
        """
        logger.print("Make new profile...")

        if isinstance(name, str) and (not name.isprintable() or "/" in name):
            logger.print(f"[warn]Invalid profile name: {logger.emph(name, type='all')}[/]")
            return False

        if name in self.profiles:
            logger.print(
                f"[warn]This profile name {logger.emph(name, type='all')} already exists.[/]"
            )
            return False

        config = self.config_type()

        path = self.profiles_dir / (name + self.extension)
        logger.print(f"[data/] Save configuration to {logger.emph(path.as_uri())}...")

        if not self.profiles_dir.exists():
            logger.print(
                f"[warn]The profile directory doesn't exist: {logger.emph(self.profiles_dir.as_uri())}[/]"
            )
            return False

        if path.exists():
            logger.print(
                f"[warn]File already exists: {logger.emph(path.as_uri())}[/]"
            )
            return False

        try:
            cfg.write(
                self.config_type, config, path, name=self.settings_name
            )
        except Exception:
            logger.print("[warn]Fail to format configuration[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return False

        self._current_mtime = os.stat(str(path)).st_mtime
        self.update(logger)
        return True

    def delete(self, logger, name):
        """Delete a profile.

        Parameters
        ----------
        logger : loggers.Logger
        name : str
            The name of profile to delete.

        Returns
        -------
        succ : bool
        """
        target_path = self.profiles_dir / (name + self.extension)
        logger.print(f"[data/] Delete profile {logger.emph(target_path.as_uri())}...")

        if name not in self.profiles:
            logger.print(f"[warn]No such profile: {logger.emph(name, type='all')}[/]")
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

    def rename(self, logger, name, newname):
        """Rename a profile.

        Parameters
        ----------
        logger : loggers.Logger
        name : str
            The old name of profile.
        newname : str
            The new name of profile.

        Returns
        -------
        succ : bool
        """
        current_path = self.profiles_dir / (self.current_name + self.extension)
        src_path = self.profiles_dir / (name + self.extension)
        dst_path = self.profiles_dir / (newname + self.extension)
        src_name = logger.emph(src_path.as_uri())
        dst_name = logger.emph(dst_path.as_uri())
        logger.print(f"[data/] Rename profile {src_name} to {dst_name}...")

        if name not in self.profiles:
            logger.print(
                f"[warn]This profile {logger.emph(name, type='all')} doesn't exist.[/]"
            )
            return False

        if not newname.isprintable() or "/" in newname:
            logger.print(f"[warn]Invalid profile name: {logger.emph(newname, type='all')}[/]")
            return False

        if newname in self.profiles:
            logger.print(
                f"[warn]This profile name {logger.emph(newname, type='all')} already exists.[/]"
            )
            return False

        if src_path.exists():
            src_path.rename(dst_path)

        if self.current_name == name:
            self.current_name = newname

        if self.default_name == name:
            self.set_default(name=newname)

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
    def __init__(self, provider):
        self.provider = provider

    @property
    def profile_manager(self):
        return self.provider.get(ProfileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    # configuration

    @cmd.function_command
    def show(self, diff: bool = False):
        """[rich]Show the current configuration.

        usage: [cmd]show[/] [[[kw]--diff[/] [arg]{DIFF}[/]]]
                               ╲
                        bool, highlight
                        changes or not.
        """
        text = self.profile_manager.format()
        is_changed = self.profile_manager.is_changed()
        title = self.profile_manager.get_title()

        if diff:
            current_path = self.profile_manager.profiles_dir / (
                self.profile_manager.current_name + self.profile_manager.extension
            )
            if not current_path.exists() or not current_path.is_file():
                old = ""
            else:
                old = open(current_path, "r").read()

            res = self.logger.format_code_diff(
                old, text, title=title, is_changed=is_changed
            )

        else:
            res = self.logger.format_code(text, title=title, is_changed=is_changed)

        self.logger.print(res)

    @cmd.function_command
    def has(self, field):
        """[rich]Check whether this field is set in the configuration.

        usage: [cmd]has[/] [arg]{field}[/]
                     ╱
              The field name.
        """
        return self.profile_manager.has(field)

    @cmd.function_command
    def get(self, field):
        """[rich]Get the value of this field in the configuration.

        usage: [cmd]get[/] [arg]{field}[/]
                     ╱
              The field name.
        """
        if not self.profile_manager.has(field) and not self.profile_manager.has_default(field):
            self.logger.print(f"[warn]No value for field {'.'.join(field)}[/]")
            return
        return self.profile_manager.get(field)

    @cmd.function_command
    def set(self, field, value):
        """[rich]Set this field in the configuration.

        usage: [cmd]set[/] [arg]{field}[/] [arg]{value}[/]
                     ╱         ╲
            The field name.   The value.
        """
        self.profile_manager.set(field, value)
        self.profile_manager.set_as_changed()

    @cmd.function_command
    def unset(self, field):
        """[rich]Unset this field in the configuration.

        usage: [cmd]unset[/] [arg]{field}[/]
                       ╱
                The field name.
        """
        self.profile_manager.unset(field)
        self.profile_manager.set_as_changed()

    @cmd.function_command
    @dn.datanode
    def edit(self):
        """[rich]Edit the configuration by external editor.

        usage: [cmd]edit[/]
        """
        title = self.profile_manager.get_title()

        editor = self.profile_manager.current.devices.terminal.editor

        text = cfg.format(
            self.profile_manager.config_type,
            self.profile_manager.current,
            self.profile_manager.settings_name,
        )

        yield

        # open editor
        if not exists(editor):
            self.logger.print(f"[warn]Unknown editor: {self.logger.escape(editor, type='all')}[/]")
            return

        self.logger.print(f"[data/] Editing...")

        edited_text = yield from edit(text, editor, ".py").join()

        # parse result
        try:
            res = cfg.parse(
                self.profile_manager.config_type, edited_text, self.profile_manager.settings_name
            )

        except pc.ParseError as error:
            line, col = pc.ParseError.locate(error.text, error.index)

            self.logger.print(
                f"[warn]parse fail at ln {line+1}, col {col+1} (marked with ◊)[/]"
            )
            self.logger.print(
                self.logger.format_code(error.text, marked=(line, col), title=title)
            )
            if error.__cause__ is not None:
                self.logger.print(
                    f"[warn]{self.logger.escape(str(error.__cause__))}[/]"
                )

        except:
            self.logger.print(f"[warn]An unexpected error occurred[/]")
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)

        else:
            self.logger.print(f"[data/] Your changes")

            self.logger.print(
                self.logger.format_code_diff(
                    text, edited_text, title=title, is_changed=True
                )
            )

            self.profile_manager.current = res
            self.profile_manager.set_as_changed()

    @get.arg_parser("field")
    @has.arg_parser("field")
    @unset.arg_parser("field")
    @set.arg_parser("field")
    def _field_parser(self):
        return FieldParser(self.profile_manager.config_type)

    @set.arg_parser("value")
    def _set_value_parser(self, field):
        annotation = self.profile_manager.config_type.get_field_type(field)
        default = self.profile_manager.get(field)
        return cmd.LiteralParser(annotation, default)

    # profiles

    @cmd.function_command
    def reload(self):
        """[rich]Reload the configuration.

        usage: [cmd]reload[/]
        """
        if not self.profile_manager.is_uptodate():
            self.profile_manager.update(self.logger)

        self.profile_manager.load(self.logger)

    @cmd.function_command
    def save(self):
        """[rich]Save the configuration.

        usage: [cmd]save[/]
        """
        if not self.profile_manager.is_uptodate():
            self.profile_manager.update(self.logger)

        self.profile_manager.save(self.logger)

    @cmd.function_command
    def set_default(self):
        """[rich]Set the current profile as default.

        usage: [cmd]set_default[/]
        """
        if not self.profile_manager.is_uptodate():
            self.profile_manager.update(self.logger)

        self.profile_manager.set_default(self.logger)

    @cmd.function_command
    def use(self, profile):
        """[rich]Change the current profile.

        usage: [cmd]use[/] [arg]{profile}[/]
                       ╱
              The profile name.
        """
        if not self.profile_manager.is_uptodate():
            self.profile_manager.update(self.logger)

        self.profile_manager.use(self.logger, profile)

    @cmd.function_command
    def new(self, profile, clone=None):
        """[rich]Make a new profile.

        usage: [cmd]new[/] [arg]{profile}[/] [[[kw]--clone[/] [arg]{PROFILE}[/]]]
                       ╱                    ╲
              The profile name.      The profile to be cloned.
        """
        if not self.profile_manager.is_uptodate():
            self.profile_manager.update(self.logger)

        self.profile_manager.new(self.logger, profile, clone)

    @new.arg_parser("profile")
    def _new_profile_parser(self):
        return cmd.RawParser()

    @new.arg_parser("clone")
    @use.arg_parser("profile")
    def _old_profile_parser(self, *_, **__):
        return cmd.OptionParser(
            self.profile_manager.profiles,
            desc="It should be the name of the profile that exists in the configuration.",
        )
