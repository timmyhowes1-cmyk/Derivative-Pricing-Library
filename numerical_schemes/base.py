from abc import ABC, abstractmethod

class NumericalScheme(ABC):
    def __init__(self, x0:float):
        self.x0 = x0

    @abstractmethod
    def get_paths(self, dt:float, *args):
        pass

def itself(c, x, t):
    return c

def standard_drift_vol(c, x, t):
    return c * x