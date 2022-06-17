import os
import tempfile
import subprocess
import shutil
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
    FileManager,
)


def check_exists_shell_command(program):
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
def edit_by_external_editor(text, editor, suffix=""):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".tmp" + suffix) as file:
        file.write(text)
        file.flush()

        yield from dn.subprocess_task([editor, file.name]).join()

        return open(file.name, mode="r").read()


class ProfilesDirPath(RecognizedDirPath):
    """The place to manage your profiles

    [rich][color=bright_blue]⠀⡠⠤⣀⠎⠉⢆⡠⠤⡀⠀[/]
    [color=bright_blue]⠀⢣⠀⠀⠀⠀⠀⠀⢠⠃⠀[/]
    [color=bright_blue]⡎⠁⠀⠀⡎⠉⡆⠀⠀⠉⡆[/] Your settings will be managed here.  Use the commands [cmd]get[/], [cmd]set[/]
    [color=bright_blue]⠈⡱⠀⠀⠈⠉⠀⠀⠰⡉⠀[/] and [cmd]unset[/] to configure your settings.  You can switch to
    [color=bright_blue]⠀⠣⠤⠒⡄⠀⡔⠢⠤⠃⠀[/] different set of settings by the command [cmd]use[/].
    [color=bright_blue]⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀[/]
    """

    def rm(self, provider):
        raise InvalidFileOperation("Deleting important directories or files may crash the program")

    @as_pattern("*.kaiko-profile")
    class profile(RecognizedFilePath):
        EXTENSION = ".kaiko-profile"

        def info(self, provider):
            profile_manager = provider.get(ProfileManager)
            note = "Your custom profile"
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
        "The file of default profile name"

        def mk(self, provider):
            super().mk(provider)
            profile_manager.update()

        def rm(self, provider):
            super().rm(provider)
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

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

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
        # update profiles
        profiles_mtime = self.profiles_dir.abs.stat().st_mtime
        if self._profiles_mtime != profiles_mtime:
            self.logger.print("[data/] Update profiles...")
            self._profile_paths = list(self.profiles_dir.profile)
            self._profiles_mtime = profiles_mtime

        # update default_path
        default_meta_path = self.profiles_dir.default
        default_mtime = default_meta_path.abs.stat().st_mtime if default_meta_path.abs.exists() else None
        if default_mtime is None or self._default_mtime != default_mtime:
            self.logger.print("[data/] Update default profile...")
            default_path = default_meta_path.abs.read_text().rstrip("\n") if default_meta_path.abs.exists() else ""
            default_path = self.profiles_dir.abs / default_path
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
        file_manager = self.file_manager
        if not isinstance(path, ProfilesDirPath.profile):
            relpath = file_manager.as_relative_path(path)
            raise InvalidFileOperation(f"Not a valid profile path: {relpath}")
        file_manager.validate_path(path, should_exist=should_exist, file_type="file")

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
        file_manager = self.file_manager

        path = path if path is not None else self.current_path

        try:
            self.validate_profile_path(path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)

        logger.print(f"[data/] Set {path_mu} as the default profile...")

        default_meta_path = self.profiles_dir.default
        default_meta_path.abs.touch()
        default_meta_path.abs.write_text(path.try_relative_to(self.profiles_dir))

        return True

    def save(self):
        """Save the current configuration.

        Returns
        -------
        succ : bool
        """
        logger = self.logger
        file_manager = self.file_manager

        try:
            self.validate_profile_path(self.current_path)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(self.current_path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Save configuration to {path_mu}...")

        try:
            cfg.write(
                self.config_type, self.current, self.current_path.abs, name=self.SETTINGS_NAME
            )
        except Exception:
            logger.print("[warn]Fail to format configuration[/]")
            logger.print_traceback()
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
        file_manager = self.file_manager

        try:
            self.validate_profile_path(self.current_path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(self.current_path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Load configuration from {path_mu}...")

        current_mtime = self.current_path.abs.stat().st_mtime

        try:
            self.current = cfg.read(
                self.config_type, self.current_path.abs, name=self.SETTINGS_NAME
            )
        except Exception:
            logger.print("[warn]Fail to parse configuration[/]")
            logger.print_traceback()
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
        file_manager = self.file_manager

        if path is None:
            path = self.default_path

            if path is None:
                logger.print("[warn]No default profile[/]")
                return False

        path = path.normalize()

        try:
            self.validate_profile_path(path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Switch to profile {path_mu}...")

        current_mtime = path.abs.stat().st_mtime

        try:
            self.current = cfg.read(
                self.config_type, path.abs, name=self.SETTINGS_NAME
            )
        except Exception:
            logger.print("[warn]Fail to parse configuration[/]")
            logger.print_traceback()
            return False

        self.current_path = path
        self.set_as_changed(current_mtime)
        return True

    def use_empty(self):
        logger = self.logger
        file_manager = self.file_manager

        path = rename_path(self.profiles_dir.abs, "new profile", ProfilesDirPath.profile.EXTENSION)
        path = self.profiles_dir.recognize(path).normalize()
        assert isinstance(path, ProfilesDirPath.profile)

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Load empty configuration as {path_mu}...")

        self.current_path = path
        self.current = self.config_type()
        self.set_as_changed()
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
        file_manager = self.file_manager

        path = path.normalize()
        try:
            self.validate_profile_path(path, should_exist=False)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Create a new profile {path_mu}...")

        if src is not None:
            src_mu = file_manager.as_relative_path(src, self.profiles_dir, markup=True)
            logger.print(f"[data/] Copy profile from {src_mu} to {path_mu}...")

            if not src.abs.exists():
                logger.print(f"[warn]No such file: {src_mu}[/]")
                return False

            shutil.copy(src.abs, path.abs)

        else:
            config = self.config_type()

            logger.print(f"[data/] Save an empty configuration to {path_mu}...")

            try:
                cfg.write(
                    self.config_type, config, path.abs, name=self.SETTINGS_NAME
                )
            except Exception:
                logger.print("[warn]Fail to format configuration[/]")
                logger.print_traceback()
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
        file_manager = self.file_manager

        path = path.normalize()
        try:
            self.validate_profile_path(path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)
        logger.print(f"[data/] Delete profile {path_mu}...")

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
        file_manager = self.file_manager

        path = path.normalize()
        newpath = newpath.normalize()

        try:
            self.validate_profile_path(path, should_exist=True)
            self.validate_profile_path(newpath, should_exist=False)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(path, self.profiles_dir, markup=True)
        newpath_mu = file_manager.as_relative_path(newpath, self.profiles_dir, markup=True)
        logger.print(f"[data/] Rename profile {path_mu} to {newpath_mu}...")

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
    def file_manager(self):
        return self.provider.get(FileManager)

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
        file_manager = self.file_manager
        logger = self.logger

        text = profile_manager.format()
        is_changed = profile_manager.is_changed()
        title = file_manager.as_relative_path(profile_manager.current_path)

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
        file_manager = self.file_manager
        logger = self.logger

        title = file_manager.as_relative_path(profile_manager.current_path)

        editor = profile_manager.current.devices.terminal.editor

        text = cfg.format(
            profile_manager.config_type,
            profile_manager.current,
            profile_manager.SETTINGS_NAME,
        )

        yield

        # open editor
        if not check_exists_shell_command(editor):
            editor_ = logger.escape(editor, type='all')
            logger.print(f"[warn]Unknown editor: [emph]{editor_}[/][/]")
            return

        logger.print(f"[data/] Editing...")

        edited_text = yield from edit_by_external_editor(text, editor, ".py").join()

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
                    f"[warn]{logger.escape(str(error.__cause__))}[/]"
                )

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

        usage: [cmd]set_default[/] [arg]{profile}[/]
                               ╱
                      The profile path.
        """
        self.profile_manager.set_default(profile)
        self.profile_manager.update()

    @cmd.function_command
    def use(self, profile):
        """[rich]Change the current profile.

        usage: [cmd]use[/] [arg]{profile}[/]
                       ╱
              The profile path.
        """
        self.profile_manager.use(profile)
        self.profile_manager.update()

    @use.arg_parser("profile")
    @set_default.arg_parser("profile")
    def _use_profile_parser(self):
        return self.file_manager.make_parser(
            desc="It should be the path of profile",
            filter=lambda path: isinstance(path, ProfilesDirPath.profile),
        )
