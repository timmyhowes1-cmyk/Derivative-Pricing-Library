from abc import ABC, abstractmethod

class NumericalScheme(ABC):
    def __init__(self, x0:float):
        self.x0 = x0

    @abstractmethod
    def get_paths(self, dt:float, *args):
        pass

def create_scheme(scheme_name, *args, **kwargs):
    # 1. Get all classes that inherit from NumericalScheme
    subclasses = NumericalScheme.__subclasses__()

    # 2. Find the one matching the string
    for cls in subclasses:
        if cls.__name__ == scheme_name:
            # 3. Instantiate and return
            return cls(*args, **kwargs)

    # 4. Fallback if not found
    raise ValueError(f"Scheme '{scheme_name}' not found. Options are: {[c.__name__ for c in subclasses]}")