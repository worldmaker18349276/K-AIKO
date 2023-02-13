import contextlib
import functools
from dataclasses import dataclass


class ServiceError(Exception):
    pass


@dataclass
class ServiceItem:
    service: object


_services = []


def get(typ):
    global _services
    for item in reversed(_services):
        if typ == type(item.service):
            return item.service
    else:
        raise ServiceError(f"Service not found: {typ}")


def set_static(service):
    global _services
    _services.append(ServiceItem(service))


@contextlib.contextmanager
def set(*services):
    global _services
    items = [ServiceItem(service) for service in services]
    _services.extend(items)
    try:
        yield
    finally:
        for item in items:
            _services.remove(item)


def inject(**services):
    def inject_decorator(func):
        @functools.wraps(func)
        def injected_func(*args, **kw):
            for name, typ in services.items():
                if name not in kw:
                    kw[name] = get(typ)
            return func(*args, **kw)

        return injected_func

    return inject_decorator
