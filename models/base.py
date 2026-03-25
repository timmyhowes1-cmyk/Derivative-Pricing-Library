from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, x0, **kwargs):
        self.x0 = x0

    @abstractmethod
    def generate_paths(self, iterations, timestep, expiry, dw=None):
        pass

