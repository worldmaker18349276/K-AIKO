import contextlib
import functools


class ServiceError(Exception):
    pass


_services = []


def get(typ):
    global _services
    for service in reversed(_services):
        if typ == type(service):
            return service
    else:
        raise ServiceError(f"Service not found: {typ}")


def set_static(service):
    global _services
    _services.insert(0, service)


@contextlib.contextmanager
def set(*services):
    global _services
    length = len(services)
    _services.extend(services)
    try:
        yield
    finally:
        if length > 0:
            del _services[-length:]


def inject(**services):
    def inject_decorator(func):
        @functools.wraps(func)
        def injected_func(*args, **kw):
            for name, service in services.items():
                if name not in kw:
                    kw[name] = get(service)
            return func(*args, **kw)

        return injected_func

    return inject_decorator
