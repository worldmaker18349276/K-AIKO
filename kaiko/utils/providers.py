
class ServiceError(Exception):
    pass

class Provider:
    def __init__(self):
        self.services_provider = {}

    def get(self, type):
        if type in self.services_provider:
            return self.services_provider[type]
        else:
            raise ServiceError(f"Service not found: {type}")

    def set(self, obj):
        if type(obj) in self.services_provider:
            raise ServiceError(f"Service already set: {type}")
        else:
            self.services_provider[type(obj)] = obj

