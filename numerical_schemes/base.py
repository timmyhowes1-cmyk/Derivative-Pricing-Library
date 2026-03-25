from abc import ABC, abstractmethod

class NumericalScheme(ABC):
    def __init__(self, x0):
        self.x0 = x0

    @abstractmethod
    def get_paths(self, dt, dw):
        pass