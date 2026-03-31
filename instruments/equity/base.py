from abc import ABC, abstractmethod
import numpy as np
from typing import Union

class EquityOption(ABC):
    def __init__(self, strike:float, expiry:float, call:bool=True, european:bool=True, **kwargs):
        self.K = strike
        self.T = expiry
        self.call = call
        self.european = european

    @abstractmethod
    def payoff(self, price_path:Union[float, np.ndarray]):
        pass

    def raiseStrikeError(self):
        if self.K is None or self.K < 0:
            raise ValueError("Strike must be non-negative")

    @staticmethod
    def raisePriceError(price_path:Union[float, np.ndarray]):
        if np.any(price_path < 0):
            raise ValueError("Price path must be non-negative")

    def get_payoff_spot(self, price_path: np.ndarray):
        if isinstance(price_path, np.ndarray) and np.ndim(price_path) >= 2 and price_path.shape[1] > 0:
            return price_path[..., -1] if self.european else price_path
        return price_path

