from abc import ABC, abstractmethod
import numpy as np

class Option(ABC):

    def __init__(self, strike, expiry, call=True, european=True, **kwargs):
        self.K = strike
        self.T = expiry
        self.call = call
        self.european = european

    @abstractmethod
    def payoff(self, price_path):
        pass

    def raiseStrikeError(self):
        if self.K is None or self.K < 0:
            raise ValueError("Strike must be non-negative")

    @staticmethod
    def raisePriceError(price_path):
        if np.any(price_path < 0):
            raise ValueError("Price path must be non-negative")

    def get_payoff_spot(self, price_path):
        if isinstance(price_path, np.ndarray):
            return price_path[..., -1] if self.european else price_path
        return price_path

