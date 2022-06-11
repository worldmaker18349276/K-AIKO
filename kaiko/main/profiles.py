import os
import traceback
import tempfile
import subprocess
import shutil
from pathlib import Path
from ..utils import config as cfg
from ..utils import parsec as pc
from ..utils import commands as cmd
from ..utils import datanodes as dn
from .loggers import Logger
from .files import (
    RecognizedFilePath,
    RecognizedDirPath,
    InvalidFileOperation,
    as_pattern,
    as_child,
    rename_path,
    validate_path,
    FileManager,
)


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


class ProfilesDirPath(RecognizedDirPath):
    "(The place to manage your profiles)"

    def mk(self, provider):
        file_manager = provider.get(FileManager)
        validate_path(self, should_exist=False, root=file_manager.root, file_type="all")
        self.abs.mkdir()

    @as_pattern("*.kaiko-profile")
    class profile(RecognizedFilePath):
        EXTENSION = ".kaiko-profile"

        def desc(self, provider):
            profile_manager = provider.get(ProfileManager)
            note = "(Your custom profile)"
            if self.abs == profile_manager.default_path.abs:
                note += " (default)"
            if self.abs == profile_manager.current_path.abs:
                note += " (current)"
            return note

        def mk(self, provider):
            profile_manager = provider.get(ProfileManager)
            succ = profile_manager.create(self)
            if not succ:
                return
            profile_manager.update()

        def rm(self, provider):
            profile_manager = provider.get(ProfileManager)
            succ = profile_manager.delete(self)
            if not succ:
                return
            profile_manager.update()

        def mv(self, dst, provider):
            profile_manager = provider.get(ProfileManager)
            succ = profile_manager.rename(self, dst)
            if not succ:
                return
            profile_manager.update()

        def cp(self, src, provider):
            profile_manager = provider.get(ProfileManager)
            succ = profile_manager.create(self, src)
            if not succ:
                return
            profile_manager.update()

    @as_child(".default-profile")
    class default(RecognizedFilePath):
        "(The file of default profile name)"

        def mk(self, provider):
            file_manager = provider.get(FileManager)
            profile_manager = provider.get(ProfileManager)
            validate_path(self, should_exist=False, root=file_manager.root, file_type="all")
            self.abs.touch()
            profile_manager.update()

        def rm(self, provider):
            file_manager = provider.get(FileManager)
            profile_manager = provider.get(ProfileManager)
            validate_path(self, should_exist=True, root=file_manager.root, file_type="file")
            self.abs.unlink()
            self.abs.touch()
            profile_manager.update()


class ProfileManager:
    """Profile manager for Configurable object.

    Attributes
    ----------
    config_type : cfg.Configurable
        The configuration type to manage.
    profiles_dir : ProfilesDirPath
        The path of profiles directory.
    profile_paths : list of ProfilesDirPath.profile
        A list of paths of profiles.
    default_path : ProfilesDirPath.profile or None
        The path of default profile.
    current_path : ProfilesDirPath.profile
        The path of current profile.
    current : config.Configurable
        The current configuration.
    provider : utils.provider.Provider
    """

    SETTINGS_NAME = "settings"

    def __init__(self, config_type, profiles_dir, provider):
        self.config_type = config_type
        self.profiles_dir = profiles_dir
        self.provider = provider

        self._profiles_mtime = None
        self._default_mtime = None
        self._profile_paths = []
        self._default_path = None

        self.current_path = None
        self.current = None
        self._current_mtime = None

        self.on_change_handlers = []

    def on_change(self, on_change_handler):
        self.on_change_handlers.append(on_change_handler)

    @property
    def logger(self):
        return self.provider.get(Logger)

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
        return self.provider.get(FileManager).as_relative_path(self.current_path)

    def format(self):
        return cfg.format(self.config_type, self.current, name=self.SETTINGS_NAME)

    def is_changed(self):
        if not self.current_path.abs.exists():
            return True
        return self._current_mtime != self.current_path.abs.stat().st_mtime

    def set_as_changed(self, mtime=None):
        self._current_mtime = mtime
        for on_change_handler in self.on_change_handlers:
            on_change_handler(self.current)

    # profiles management

    def update(self):
        if self._profiles_mtime != self.profiles_dir.abs.stat().st_mtime:
            # update profiles
            profiles_mtime = self.profiles_dir.abs.stat().st_mtime
            self._profile_paths = list(self.profiles_dir.profile)
            self._profiles_mtime = profiles_mtime

        default_meta_path = self.profiles_dir.default
        if self._default_mtime != default_meta_path.abs.stat().st_mtime:
            # update default_path
            default_mtime = default_meta_path.abs.stat().st_mtime
            default_path = default_meta_path.abs.read_text().rstrip("\n")
            default_path = os.path.join(self.profiles_dir.abs, default_path)
            default_path = self.profiles_dir.recognize(default_path)
            default_path = default_path.normalize()
            if not isinstance(default_path, ProfilesDirPath.profile):
                default_path = None
            self._default_path = default_path
            self._default_mtime = default_mtime

    @property
    def profile_paths(self):
        self.update()
        return self._profile_paths

    @property
    def default_path(self):
        self.update()
        return self._default_path

    def validate_profile_path(self, path, should_exist=None):
        file_manager = self.provider.get(FileManager)
        if not isinstance(path, ProfilesDirPath.profile):
            raise InvalidFileOperation(f"Not a valid profile path: {logger.as_uri(path.abs)}")
        validate_path(path, should_exist=should_exist, root=file_manager.root, file_type="file")

    def set_default(self, path=None):
        """Set given profile path as default.

        Parameters
        ----------
        path : ProfilesDirPath.profile, optional

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        path = path if path is not None else self.current_path

        try:
            self.validate_profile_path(path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        path_str = path.try_relative_to(self.profiles_dir)

        logger.print(
            f"[data/] Set {logger.emph(path_str, type='all')} as the default profile..."
        )

        default_meta_path = self.profiles_dir.default
        default_meta_path.abs.write_text(path_str)

        return True

    def save(self):
        """Save the current configuration.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        try:
            self.validate_profile_path(self.current_path)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        logger.print(
            f"[data/] Save configuration to {logger.as_uri(self.current_path.abs)}..."
        )

        try:
            cfg.write(
                self.config_type, self.current, self.current_path.abs, name=self.SETTINGS_NAME
            )
        except Exception:
            logger.print("[warn]Fail to format configuration[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return False

        self._current_mtime = self.current_path.abs.stat().st_mtime

        return True

    def load(self):
        """Load the current configuration.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        try:
            self.validate_profile_path(self.current_path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        logger.print(
            f"[data/] Load configuration from {logger.as_uri(self.current_path.abs)}..."
        )

        current_mtime = self.current_path.abs.stat().st_mtime

        try:
            self.current = cfg.read(
                self.config_type, self.current_path.abs, name=self.SETTINGS_NAME
            )
        except Exception:
            logger.print("[warn]Fail to parse configuration[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)
            return False

        self.set_as_changed(current_mtime)
        return True

    def use(self, path=None):
        """Switch the current configuration.

        Parameters
        ----------
        path : ProfilesDirPath.profile, optional

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        if path is None:
            path = self.default_path

            if path is None:
                logger.print("[warn]No default profile[/]")

                # make a new profile
                path = rename_path(self.profiles_dir.abs, "new profile", ProfilesDirPath.profile.EXTENSION)
                path = self.profiles_dir.recognize(path).normalize()
                assert isinstance(path, ProfilesDirPath.profile)

                logger.print(f"[data/] Load empty configuration as {logger.as_uri(path.abs)}...")

                self.current_path = path
                self.current = self.config_type()
                self.set_as_changed()
                return True

        path = path.normalize()
        path_str = path.try_relative_to(self.profiles_dir)

        logger.print(
            f"[data/] Switch to profile {logger.emph(path_str, type='all')}..."
        )

        profile_paths = self.profile_paths
        if path not in profile_paths:
            logger.print(f"[warn]No such profile: {logger.emph(path_str, type='all')}[/]")
            return False

        previous_path = self.current_path
        self.current_path = path
        succ = self.load()
        if not succ:
            self.current_name = previous_path
            return False
        return True

    def create(self, path, src=None):
        """Create a new profile.

        Parameters
        ----------
        path : ProfilesDirPath.profile
        src : RecognizedPath, None
            The source file to copy, or None for empty profile.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        path = path.normalize()
        try:
            self.validate_profile_path(path, should_exist=False)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        path_str = path.try_relative_to(self.profiles_dir)
        logger.print(f"[data/] Create a new profile {logger.emph(path_str, type='all')}...")

        if src is not None:
            logger.print(f"[data/] Copy profile from {logger.as_uri(src.abs)} to {logger.as_uri(path.abs)}...")

            if not src.abs.exists():
                logger.print(f"[warn]No such file: {logger.as_uri(src.abs)}[/]")
                return False

            shutil.copy(src.abs, path.abs)

        else:
            config = self.config_type()

            logger.print(f"[data/] Save an empty configuration to {logger.as_uri(path.abs)}...")

            try:
                cfg.write(
                    self.config_type, config, path.abs, name=self.SETTINGS_NAME
                )
            except Exception:
                logger.print("[warn]Fail to format configuration[/]")
                with logger.warn():
                    logger.print(traceback.format_exc(), end="", markup=False)
                return False

        return True

    def delete(self, path):
        """Delete a profile.

        Parameters
        ----------
        path : ProfilesDirPath.profile

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        path = path.normalize()
        try:
            self.validate_profile_path(path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        path_str = path.try_relative_to(self.profiles_dir)
        logger.print(f"[data/] Delete profile {logger.emph(path_str, type='all')}...")

        path.abs.unlink()

        return True

    def rename(self, path, newpath):
        """Rename a profile.

        Parameters
        ----------
        path : ProfilesDirPath.profile
            The old path of profile.
        newpath : ProfilesDirPath.profile
            The new path of profile.

        Returns
        -------
        succ : bool
        """
        logger = self.logger

        path = path.normalize()
        newpath = newpath.normalize()

        try:
            self.validate_profile_path(path, should_exist=True)
            self.validate_profile_path(newpath, should_exist=False)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{str(e)}[/]")
            return False

        logger.print(f"[data/] Rename profile {logger.as_uri(path.abs)} to {logger.as_uri(newpath.abs)}...")

        is_current = self.current_path == path
        is_default = self.default_path == path

        path.abs.rename(newpath.abs)
        if is_current:
            self.current_path = newpath
        if is_default:
            self.set_default(newpath)

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
        profile_manager = self.profile_manager
        logger = self.logger

        text = profile_manager.format()
        is_changed = profile_manager.is_changed()
        title = profile_manager.get_title()

        if diff:
            if not profile_manager.current_path.abs.exists():
                old = ""
            else:
                old = profile_manager.current_path.abs.read_text()

            res = logger.format_code_diff(old, text, title=title, is_changed=is_changed)

        else:
            res = logger.format_code(text, title=title, is_changed=is_changed)

        logger.print(res)

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
        profile_manager = self.profile_manager
        logger = self.logger

        title = profile_manager.get_title()

        editor = profile_manager.current.devices.terminal.editor

        text = cfg.format(
            profile_manager.config_type,
            profile_manager.current,
            profile_manager.SETTINGS_NAME,
        )

        yield

        # open editor
        if not exists(editor):
            logger.print(f"[warn]Unknown editor: {logger.emph(editor, type='all')}[/]")
            return

        logger.print(f"[data/] Editing...")

        edited_text = yield from edit(text, editor, ".py").join()

        # parse result
        try:
            res = cfg.parse(
                profile_manager.config_type, edited_text, profile_manager.SETTINGS_NAME
            )

        except pc.ParseError as error:
            line, col = pc.ParseError.locate(error.text, error.index)

            logger.print(
                f"[warn]parse fail at ln {line+1}, col {col+1} (marked with ◊)[/]"
            )
            logger.print(
                logger.format_code(error.text, marked=(line, col), title=title)
            )
            if error.__cause__ is not None:
                logger.print(
                    f"[warn]{self.logger.escape(str(error.__cause__))}[/]"
                )

        except:
            logger.print(f"[warn]An unexpected error occurred[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            logger.print(f"[data/] Your changes")

            logger.print(
                logger.format_code_diff(
                    text, edited_text, title=title, is_changed=True
                )
            )

            profile_manager.current = res
            profile_manager.set_as_changed()

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
        self.profile_manager.load()
        self.profile_manager.update()

    @cmd.function_command
    def save(self):
        """[rich]Save the configuration.

        usage: [cmd]save[/]
        """
        self.profile_manager.save()
        self.profile_manager.update()

    @cmd.function_command
    def set_default(self, profile):
        """[rich]Set the current profile as default.

        usage: [cmd]use_default[/] [arg]{profile}[/]
                               ╱
                      The profile path.
        """
        profiles_dir = self.profile_manager.profiles_dir
        profile = profiles_dir.recognize(profiles_dir.abs / profile)
        self.profile_manager.set_default(profile)
        self.profile_manager.update()

    @cmd.function_command
    def use(self, profile):
        """[rich]Change the current profile.

        usage: [cmd]use[/] [arg]{profile}[/]
                       ╱
              The profile path.
        """
        profiles_dir = self.profile_manager.profiles_dir
        profile = profiles_dir.recognize(profiles_dir.abs / profile)
        self.profile_manager.use(profile)
        self.profile_manager.update()

    @use.arg_parser("profile")
    @set_default.arg_parser("profile")
    def _use_profile_parser(self):
        profiles_paths = [path.abs.name for path in self.profile_manager.profile_paths]
        default_path = self.profile_manager.default_path
        if default_path is not None:
            return cmd.OptionParser(
                profiles_paths,
                default=default_path.abs.name,
                desc="It should be the path of profile.",
            )
        else:
            return cmd.OptionParser(
                profiles_paths,
                desc="It should be the path of profile.",
            )
