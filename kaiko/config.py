"""
A configuration system using biparser.
The format of configuration file is a sub-language of python.
"""

import re
import typing
from collections import OrderedDict
from pathlib import Path
from . import biparsers as bp


class FieldBiparser(bp.Biparser):
    """Biparser for fields of configuration."""
    def __init__(self, config_type):
        self.config_type = config_type

    def decode(self, text, index=0, partial=False):
        current_fields = []
        current_type = self.config_type

        while hasattr(current_type, '__configurable_fields__'):
            fields = {}
            for field_name, field_type in current_type.__configurable_fields__.items():
                field_key = field_name
                if hasattr(field_type, '__configurable_fields__'):
                    field_key = field_key + "."
                fields[field_key] = (field_name, field_type)

            option, index = bp.startswith(list(fields.keys()), text, index, partial=True)

            current_field, current_type = fields[option]
            current_fields.append(current_field)

        if not partial:
            bp.eof(text, index)

        return tuple(current_fields), index

    def encode(self, value):
        current_type = self.config_type

        for i, field_name in enumerate(value):
            if not hasattr(current_type, '__configurable_fields__'):
                raise EncodeError(value, "[" + ".".join(value[:i]) + "].fields")
            current_type = current_type.__configurable_fields__[field_name]

        if hasattr(current_type, '__configurable_fields__'):
            raise EncodeError(value, "[" + ".".join(value) + "]")

        return ".".join(value)

class ConfigurationBiparser(bp.Biparser):
    vindent = r"(#[^\n]*|[ ]*)(\n|$)"
    name = r"#### ([^\n]*) ####(\n|$)"
    profile = r"# #### ([^\n]*) ####(\n#[^\n]*)*(\n|$)"
    equal = r"[ ]*=[ ]*"
    nl = r"[ ]*(\n|$)"

    def __init__(self, config_type):
        self.config_type = config_type
        self.field_biparser = FieldBiparser(config_type)

    def decode(self, text, index=0, partial=False):
        # exec(text, globals(), config.current.__dict__)

        config = Configuration(self.config_type)
        field_hints = self.config_type.get_configurable_fields()
        is_named = False

        while index < len(text):
            if not is_named:
                m, index = bp.match(self.name, [], text, index, optional=True, partial=True)
                if m:
                    config.name = m.group(1)
                    is_named = True
                    continue

            m, index = bp.match(self.profile, [], text, index, optional=True, partial=True)
            if m and m.group(1) not in config.profiles:
                config.profiles[m.group(1)] = m.group(0)
                continue

            m, index = bp.match(self.vindent, ["\n"], text, index, optional=True, partial=True)
            if m: continue

            field, index = self.field_biparser.decode(text, index, partial=True)

            _, index = bp.match(self.equal, [" = "], text, index, partial=True)

            value_biparser = bp.from_type_hint(field_hints[field])
            value, index = value_biparser.decode(text, index, partial=True)
            config.set(field, value)

            _, index = bp.match(self.nl, ["\n"], text, index, partial=True)

        return config, index

    def encode(self, value):
        res = ""

        if value.name is not None:
            res += f"#### {value.name} ####\n"

        for field_key, field_type in self.config_type.get_configurable_fields().items():
            field_name = self.field_biparser.encode(field_key)
            value_biparser = bp.from_type_hint(field_type)
            if not value.has(field_key):
                continue
            field_value = value.get(field_key)
            field_value_str = value_biparser.encode(field_value)
            res += field_name + " = " + field_value_str + "\n"

        if value.profiles:
            res += "\n" + "\n".join(value.profiles.values())

        return res

class ConfigurableMeta(type):
    def __init__(self, name, supers, attrs):
        if not hasattr(self, '__configurable_excludes__'):
            self.__configurable_excludes__ = []
        annotations = typing.get_type_hints(self)

        fields = OrderedDict()
        for name in dir(self):
            if name not in self.__configurable_excludes__:
                if name in annotations:
                    fields[name] = annotations[name]
                elif isinstance(getattr(self, name), ConfigurableMeta):
                    fields[name] = getattr(self, name)
        self.__configurable_fields__ = fields

    def __configurable_init__(self, instance):
        for field_name, field_type in self.__configurable_fields__.items():
            if hasattr(field_type, '__configurable_fields__'):
                instance.__dict__[field_name] = field_type()

    def __call__(self, *args, **kwargs):
        instance = self.__new__(self, *args, **kwargs)
        self.__configurable_init__(instance)
        self.__init__(instance, *args, **kwargs)
        return instance

    def get_configurable_fields(self):
        """Get configurable fields of a configuration.
        
        Returns
        -------
        field_hints : dict
            A dictionary from a series of field names to field type.
            If it has item `(('a', 'b', 'c'), float)`, then this configuration
            should have the field `config.a.b.c` with type `float`.
        """
        field_hints = {}
        for field_name, field_type in self.__configurable_fields__.items():
            if not hasattr(field_type, '__configurable_fields__'):
                field_hints[(field_name,)] = field_type
            else:
                subfield_hints = field_type.get_configurable_fields()
                for subfield_names, subfield_type in subfield_hints.items():
                    field_hints[(field_name, *subfield_names)] = subfield_type
        return field_hints

class Configurable(metaclass=ConfigurableMeta):
    """The super class for configuration.

    In this type, the configuration can be easily defined::
    
        class SomeSettings(Configurable):
            field1: int = 123
            field2: str = 'abc'

            class subsettings(Configurable):
                field3: bool = True
                field4: float = 3.14

        settings = SomeSettings()
        
        print(settings.field1)  # 123
        print(settings.subsettings.field3)  # True
        
        settings.field1 = 456
        settings.subsettings.field3 = False
        
        print(settings.field1)  # 456
        print(settings.subsettings.field3)  # False

    The field that is annotated or is Configurable will be assigned as a
    field of this configuration.  One can define an exclusion list
    `__configurable_excludes__` to exclude them.  The field with type
    Configurable will become sub-configuration, and will be created
    before initializing object.  The others fields will become the field
    of this configuration, which is initially absent.  So in the above
    example, the field access at the beginning is the fallback value of
    the static field in the class.
    """
    pass

class Configuration:
    def __init__(self, config_type, name=None, current=None, profiles=None):
        self.config_type = config_type
        self.name = name or "default"
        self.current = current or config_type()
        self.profiles = profiles or {}
        self.biparser = ConfigurationBiparser(config_type)

    def set(self, fields, value):
        if len(fields) == 0:
            raise ValueError("empty field")

        parent = None
        curr = self.current

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            parent.__dict__[field] = value

    def unset(self, fields):
        if len(fields) == 0:
            raise ValueError("empty field")

        parent = None
        curr = self.current

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            if field in parent.__dict__:
                del parent.__dict__[field]

    def get(self, fields):
        if len(fields) == 0:
            raise ValueError("empty field")

        parent = None
        curr = self.current

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return getattr(parent, field)

    def has(self, fields):
        if len(fields) == 0:
            raise ValueError("empty field")

        parent = None
        curr = self.current

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                return False

            if field not in curr.__configurable_fields__:
                return False

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return field in parent.__dict__

    def read(self, path):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError("No such file: " + str(path))

        res, _ = self.biparser.decode(open(path, 'r').read())
        self.current = res.current
        self.profiles = res.profiles

    def __str__(self):
        return self.biparser.encode(self)

    def write(self, path):
        if isinstance(path, str):
            path = Path(path)
        open(path, 'w').write(self.biparser.encode(self))

    def use(self, name):
        if name == self.name:
            return
        if name not in self.profiles:
            raise ValueError("no such profile: " + name)

        curr = Configuration(self.config_type, self.name, self.current, {})
        res = re.sub(r"((?<=\n)|^)(?!$)", "# ", self.biparser.encode(curr))
        self.profiles[self.name] = res

        profile = self.profiles.pop(name)
        res, _ = self.biparser.decode(re.sub(r"((?<=\n)|^)#[ ]?", "", profile))
        self.name = name
        self.current = res.current

    def new(self, name, clone=None):
        if clone is not None and clone != self.name and clone not in self.profiles:
            raise ValueError("no such profile: " + clone)

        curr = Configuration(self.config_type, self.name, self.current, {})
        res = re.sub(r"((?<=\n)|^)(?!$)", "# ", self.biparser.encode(curr))
        self.profiles[self.name] = res

        if clone is None:
            self.name = name
            self.current = self.config_type()
        else:
            profile = self.profiles[clone]
            res, _ = self.biparser.decode(re.sub(r"((?<=\n)|^)#[ ]?", "", profile))
            self.name = name
            self.current = res.current
