from abc import ABC, abstractmethod
import numpy as np
from numerical_schemes import NumericalScheme

class Model(ABC):
    def __init__(self, x0:float, **kwargs):
        self.x0 = x0

    @abstractmethod
    def generate_paths(self, iterations:int, timestep:float, expiry:float, dw:np.ndarray=None, antithetic_variates:bool=False):
        pass

