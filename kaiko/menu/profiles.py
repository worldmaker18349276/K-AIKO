import os
import traceback
import tempfile
import subprocess
from pathlib import Path
from kaiko.utils import config as cfg
from kaiko.utils import biparsers as bp
from kaiko.utils import commands as cmd
from kaiko.utils import datanodes as dn

def exists(program):
    if os.name == 'nt':
        rc = subprocess.call(["where", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        rc = subprocess.call(["which", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0

@dn.datanode
def edit(text, editor):
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".tmp") as file:
        file.write(text)
        file.flush()

        with dn.subprocess_task([editor, file.name]) as task:
            yield from task.join((yield))

        return open(file.name, mode='r').read()

class ProfileTypeError(Exception):
    pass

class ProfileNameError(Exception):
    pass

class ProfileManager:
    """Profile manager for Configurable type.

    Attributes
    ----------
    logger : KAIKOLogger
    config_type : type
        The Configurable type to manage.
    path : Path
        The path of profiles directory.
    profiles : list of str
        A list of names of profiles.
    default_name : str or None
        The name of default configuration.
    current_name : str
        The name of current configuration.
    current : Configurable
        The current configuration.
    """

    default_meta = ".default-config"
    extension = ".config"
    settings_name = "settings"

    def __init__(self, config_type, path, logger):
        """Constructor.

        Parameters
        ----------
        config_type : type
            The Configurable type to manage.
        path : str or Path
            The path of profiles directory.

        Raises
        ------
        ProfileTypeError
            If file type is wrong.
        DecodeError
            If decoding fails.
        """
        if isinstance(path, str):
            path = Path(path)

        self.logger = logger
        self.config_type = config_type
        self.path = path
        self.profiles = []
        self.default_name = None
        self.current_name = None
        self.current = None
        self._profiles_mtime = None

        self.update()

    def is_uptodate(self):
        return self._profiles_mtime == os.stat(str(self.path)).st_mtime

    def update(self):
        """Update the list of profiles."""
        self._profiles_mtime = os.stat(str(self.path)).st_mtime
        logger = self.logger

        self.profiles = []
        self.default_name = None

        logger.print(f"Update profiles...", prefix="data")

        if not self.path.exists():
            with logger.warn():
                logger.print(f"The profile directory doesn't exist: {logger.emph(self.path.as_uri())}")
            return

        if not self.path.is_dir():
            with logger.warn():
                logger.print(f"Wrong file type for profile directory: {logger.emph(self.path.as_uri())}")
            return

        for subpath in self.path.iterdir():
            if subpath.suffix == self.extension:
                self.profiles.append(subpath.stem)

        default_meta_path = self.path / self.default_meta
        if default_meta_path.exists():
            if not default_meta_path.is_file():
                with logger.warn():
                    logger.print(f"Wrong file type for default profile: {logger.emph(default_meta_path.as_uri())}")
                return
            self.default_name = default_meta_path.read_text()

    def set_default(self):
        """Set the current configuration as default configuration."""
        logger = self.logger

        logger.print(f"Set {logger.emph(self.default_name)} as the default configuration...", prefix="data")

        if self.current_name is None:
            raise ValueError("No profile")

        if not self.path.exists():
            with logger.warn():
                logger.print(f"No such profile directory: {logger.emph(self.path.as_uri())}")
            return

        self.default_name = self.current_name
        default_meta_path = self.path / self.default_meta
        if default_meta_path.exists() and not default_meta_path.is_file():
            with logger.warn():
                logger.print(f"Wrong file type for default profile: {logger.emph(default_meta_path.as_uri())}")
            return

        default_meta_path.write_text(self.default_name)

    def save(self):
        """Save the current configuration."""
        logger = self.logger

        if self.current_name is None:
            raise ValueError("No profile")

        current_path = self.path / (self.current_name + self.extension)
        logger.print(f"Save configuration to {logger.emph(current_path.as_uri())}...", prefix="data")

        if not self.path.exists():
            with logger.warn():
                logger.print(f"The profile directory doesn't exist: {logger.emph(self.path.as_uri())}")
            return

        if current_path.exists() and not current_path.is_file():
            with logger.warn():
                logger.print(f"Wrong file type for profile: {logger.emph(current_path.as_uri())}")
            return

        try:
            self.current.write(current_path, self.settings_name)
        except bp.EncodeError:
            with logger.warn():
                logger.print("Fail to encode configuration")
                logger.print(traceback.format_exc(), end="")

        self.update()

    def load(self):
        """Load the current configuration."""
        logger = self.logger

        if self.current_name is None:
            raise ValueError("No profile")

        current_path = self.path / (self.current_name + self.extension)
        logger.print(f"Load configuration from {logger.emph(current_path.as_uri())}...", prefix="data")

        self.current = self.config_type()

        if not current_path.exists():
            return

        if not current_path.is_file():
            with logger.warn():
                logger.print(f"Wrong file type for profile: {logger.emph(current_path.as_uri())}")
            return

        try:
            self.current = self.config_type.read(current_path, name=self.settings_name)
        except bp.DecodeError:
            with logger.warn():
                logger.print("Fail to decode configuration")
                logger.print(traceback.format_exc(), end="")

    def use(self, name=None):
        """change the current profile of configuration.

        Parameters
        ----------
        name : str, optional
            The name of profile, or None for default.
        """
        logger = self.logger

        if name is None:
            if self.default_name is None:
                with logger.warn():
                    logger.print("No default profile")
                return

            name = self.default_name

        if name not in self.profiles:
            with logger.warn():
                logger.print(f"No such profile: {logger.emph(name)}")
            return

        self.current_name = name
        self.load()

    def new(self, name=None, clone=None):
        """make a new profile of configuration.

        Parameters
        ----------
        name : str, optional
            The name of profile.
        clone : str, optional
            The name of profile to clone.
        """
        logger = self.logger

        logger.print("Make new configuration...")

        if clone is not None and clone not in self.profiles:
            with logger.warn():
                logger.print(f"No such profile: {logger.emph(clone)}")
            return

        if not name.isprintable() or "/" in name:
            with logger.warn():
                logger.print(f"Invalid profile name: {logger.emph(name)}")
            return

        if name in self.profiles:
            with logger.warn():
                logger.print(f"This profile name {logger.emph(name)} already exists.")
            return

        if name is None:
            name = "new profile"
            n = 1
            while name in self.profiles:
                n += 1
                name = f"new profile ({str(n)})"

        if clone is None:
            self.current_name = name
            self.current = self.config_type()
        else:
            self.current_name = clone
            self.load()
            self.current_name = name

    def delete(self, name):
        """Delete a profile.

        Parameters
        ----------
        name : str
            The name of profile to delete.
        """
        logger = self.logger

        target_path = self.path / (name + self.extension)
        logger.print(f"Delete configuration {logger.emph(target_path.as_uri())}...", prefix="data")

        if name not in self.profiles:
            with logger.warn():
                logger.print(f"No such profile: {logger.emph(name)}")
            return

        self.profiles.remove(name)

        if target_path.exists():
            if not target_path.is_file():
                with logger.warn():
                    logger.print(f"Wrong file type for profile: {logger.emph(target_path.as_uri())}")
                return
            target_path.unlink()

    def rename(self, name):
        """Rename the current profile.

        Parameters
        ----------
        name : str
            The new name of profile.
        """
        logger = self.logger

        if self.current_name is None:
            raise ValueError("No profile")

        if self.current_name == name:
            return

        current_path = self.path / (self.current_name + self.extension)
        target_path = self.path / (name + self.extension)
        current_name = logger.emph(current_path.as_uri())
        target_name = logger.emph(target_path.as_uri())
        logger.print(f"Rename configuration file {current_name} to {target_name}...", prefix="data")

        if not name.isprintable() or "/" in name:
            with logger.warn():
                logger.print(f"Invalid profile name: {logger.emph(name)}")
            return

        if name in self.profiles:
            with logger.warn():
                logger.print(f"This profile name {logger.emph(name)} already exists.")
            return

        if self.current_name in self.profiles:
            self.profiles.remove(self.current_name)
            self.profiles.append(name)

            if current_path.exists():
                current_path.rename(target_path)

        if self.current_name == self.default_name:
            self.current_name = name
            self.set_default()
        else:
            self.current_name = name

class FieldParser(cmd.ArgumentParser):
    def __init__(self, config_type):
        self.config_type = config_type
        self.biparser = cfg.FieldBiparser(config_type)

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except bp.DecodeError:
            raise cmd.CommandParseError("No such field")

    def suggest(self, token):
        try:
            self.biparser.decode(token)
        except bp.DecodeError as e:
            sugg = cmd.fit(token, [token[:e.index] + ex for ex in e.expected])
        else:
            sugg = []

        return sugg

    def info(self, token):
        fields = self.parse(token)
        return self.config_type.get_field_doc(fields)

class ConfigCommand:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    # configuration

    @cmd.function_command
    def show(self):
        """Show configuration."""
        biparser = cfg.ConfigurationBiparser(self.config.config_type, name=self.config.settings_name)
        text = biparser.encode(self.config.current)

        self.logger.print(f"profile name: {self.logger.emph(self.config.current_name)}")
        self.logger.print(text)

    @cmd.function_command
    def has(self, field):
        """Check whether this field is set in the configuration.

        usage: \x1b[94mconfig\x1b[m \x1b[94mhas\x1b[m \x1b[92m{field}\x1b[m
                            ╱
                     The field name.
        """
        return self.config.current.has(field)

    @cmd.function_command
    def get(self, field):
        """Get the value of this field in the configuration.

        usage: \x1b[94mconfig\x1b[m \x1b[94mget\x1b[m \x1b[92m{field}\x1b[m
                            ╱
                     The field name.
        """
        return self.config.current.get(field)

    @cmd.function_command
    def set(self, field, value):
        """Set this field in the configuration.

        usage: \x1b[94mconfig\x1b[m \x1b[94mset\x1b[m \x1b[92m{field}\x1b[m \x1b[92m{value}\x1b[m
                            ╱         ╲
                   The field name.   The value.
        """
        self.config.current.set(field, value)

    @cmd.function_command
    def unset(self, field):
        """Unset this field in the configuration.

        usage: \x1b[94mconfig\x1b[m \x1b[94munset\x1b[m \x1b[92m{field}\x1b[m
                              ╱
                       The field name.
        """
        self.config.current.unset(field)

    @cmd.function_command
    @dn.datanode
    def edit(self, field):
        """Edit the value of this field in the configuration.

        usage: \x1b[94mconfig\x1b[m \x1b[94medit\x1b[m \x1b[92m{field}\x1b[m
                             ╱
                      The field name.
        """
        editor = self.config.current.menu.editor

        field_type = self.config.config_type.get_field_type(field)
        biparser = bp.from_type_hint(field_type, multiline=True)

        if self.config.current.has(field):
            value = self.config.current.get(field)
            value_str = biparser.encode(value)
        else:
            value_str = ""

        yield

        # open editor
        if not exists(editor):
            with self.logger.warn():
                self.logger.print(f"Unknown editor: {editor}")
                return

        with edit(value_str, editor) as edit_task:
            yield from edit_task.join((yield))
            res_str = edit_task.result

        # parse result
        res_str = res_str.strip()

        if res_str == "":
            self.config.current.unset(field)
            return

        try:
            res, _ = biparser.decode(res_str)

        except bp.DecodeError as e:
            with self.logger.warn():
                self.logger.print("Invalid syntax:")
                self.logger.print(e)

        else:
            self.config.current.set(field, res)

    @get.arg_parser("field")
    @has.arg_parser("field")
    @unset.arg_parser("field")
    @set.arg_parser("field")
    @edit.arg_parser("field")
    def _field_parser(self):
        return FieldParser(self.config.config_type)

    @set.arg_parser("value")
    def _set_value_parser(self, field):
        annotation = self.config.config_type.get_field_type(field)
        default = self.config.current.get(field)
        return cmd.LiteralParser(annotation, default)

    # profiles

    @cmd.function_command
    def profiles(self):
        """Show all profiles.

        usage: \x1b[94mconfig\x1b[m \x1b[94mprofiles\x1b[m
        """
        logger = self.logger

        if not self.config.is_uptodate():
            self.config.update()

        for profile in self.config.profiles:
            note = ""
            if profile == self.config.default_name:
                note += " (default)"
            if profile == self.config.current_name:
                note += " (current)"
            logger.print(logger.emph(profile + self.config.extension) + note)

    @cmd.function_command
    def reload(self):
        """Reload configuration."""
        logger = self.logger

        if not self.config.is_uptodate():
            self.config.update()

        self.config.load()

    @cmd.function_command
    def save(self):
        """Save configuration."""
        logger = self.logger

        if not self.config.is_uptodate():
            self.config.update()

        self.config.save()

    @cmd.function_command
    def set_default(self):
        """Set the current configuration profile as default.

        usage: \x1b[94mconfig\x1b[m \x1b[94mset_default\x1b[m
        """
        if not self.config.is_uptodate():
            self.config.update()

        self.config.set_default()

    @cmd.function_command
    def use(self, profile):
        """Change the current configuration profile.

        usage: \x1b[94mconfig\x1b[m \x1b[94muse\x1b[m \x1b[92m{profile}\x1b[m
                              ╱
                     The profile name.
        """

        if not self.config.is_uptodate():
            self.config.update()

        self.config.use(profile)

    @cmd.function_command
    def rename(self, profile):
        """Rename current configuration profile.

        usage: \x1b[94mconfig\x1b[m \x1b[94mrename\x1b[m \x1b[92m{profile}\x1b[m
                                ╱
                      The profile name.
        """
        if not self.config.is_uptodate():
            self.config.update()

        self.config.rename(profile)

    @cmd.function_command
    def new(self, profile, clone=None):
        """Make new configuration profile.

        usage: \x1b[94mconfig\x1b[m \x1b[94mnew\x1b[m \x1b[92m{profile}\x1b[m [\x1b[95m--clone\x1b[m \x1b[92m{PROFILE}\x1b[m]
                              ╱                    ╲
                     The profile name.      The profile to be cloned.
        """
        if not self.config.is_uptodate():
            self.config.update()

        self.config.new(profile, clone)

    @cmd.function_command
    def delete(self, profile):
        """Delete a configuration profile.

        usage: \x1b[94mconfig\x1b[m \x1b[94mdelete\x1b[m \x1b[92m{profile}\x1b[m
                                ╱
                       he profile name.
        """
        if not self.config.is_uptodate():
            self.config.update()

        self.config.delete(profile)

    @rename.arg_parser("profile")
    @new.arg_parser("profile")
    def _new_profile_parser(self):
        return cmd.RawParser()

    @new.arg_parser("clone")
    @use.arg_parser("profile")
    @delete.arg_parser("profile")
    def _old_profile_parser(self, *_, **__):
        return cmd.OptionParser(self.config.profiles,
                                desc="It should be the name of the profile that exists in the configuration.")
