import re
import typing
from collections import OrderedDict
from pathlib import Path
from . import biparser


class FieldBiparser(biparser.Biparser):
    def __init__(self, config_type):
        self.config_type = config_type

    @property
    def name(self):
        return f"Field[{config_type.__name__}]"

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

            options = sorted(list(fields.keys()), reverse=True)
            option, index = biparser.startswith(options, text, index, partial=True)

            current_field, current_type = fields[option]
            current_fields.append(current_field)

        if not partial:
            biparser.eof(text, index)

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

class ConfigurationBiparser(biparser.Biparser):
    vindent = r"(\#[^\r\n]*|[ ]*)(\n|$)"
    equal = r"[ ]*=[ ]*"
    nl = r"[ ]*(\n|$)"

    def __init__(self, config_type):
        self.config_type = config_type
        self.field_biparser = FieldBiparser(config_type)

    @property
    def name(self):
        return f"Configuration[{config_type.__name__}]"

    def decode(self, text, index=0, partial=False):
        # exec(text, globals(), config.__dict__)

        config = self.config_type()
        field_hints = self.config_type.get_configurable_fields()

        while index < len(text):
            m, index = biparser.match(self.vindent, ["\n"], text, index, optional=True, partial=True)
            if m: continue

            field, index = self.field_biparser.decode(text, index, partial=True)

            _, index = biparser.match(self.equal, [" = "], text, index, partial=True)

            value_biparser = biparser.from_type_hint(field_hints[field])
            value, index = value_biparser.decode(text, index, partial=True)
            config.set(field, value)

            _, index = biparser.match(self.nl, ["\n"], text, index, partial=True)

        return config, index

    def encode(self, value):
        res = ""
        for field_key, field_type in self.config_type.get_configurable_fields().items():
            field_name = self.field_biparser.encode(field_key)
            value_biparser = biparser.from_type_hint(field_type)
            if not value.has(field_key):
                continue
            field_value = value.get(field_key)
            field_value_str = value_biparser.encode(field_value)
            res += field_name + " = " + field_value_str + "\n"
        return res

class ConfigurableMeta(type):
    def __init__(self, name, supers, attrs):
        if not hasattr(self, '__configurable_excludes__'):
            self.__configurable_excludes__ = []
        annotations = typing.get_type_hints(self)

        fields = OrderedDict()
        for name in dir(self):
            if name in annotations and name not in self.__configurable_excludes__:
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
    def set(self, fields, value):
        if len(fields) == 0:
            raise ValueError("empty field")

        parent = None
        curr = self

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
        curr = self

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
        curr = self

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
        curr = self

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                return False

            if field not in curr.__configurable_fields__:
                return False

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return field in parent.__dict__

    @classmethod
    def read(clz, path):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            return clz()
        biparser = ConfigurationBiparser(clz)
        res, _ = biparser.decode(open(path, 'r').read())
        return res

    def __str__(self):
        biparser = ConfigurationBiparser(type(self))
        return biparser.encode(self)

    def write(self, path):
        if isinstance(path, str):
            path = Path(path)
        biparser = ConfigurationBiparser(type(self))
        res = biparser.encode(self)
        open(path, 'w').write(res)

