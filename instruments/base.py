from abc import ABC, abstractmethod
import numpy as np

class Option(ABC):

    def __init__(self, option_type, strike, expiry, **kwargs):
        self.option_type = option_type
        self.K = strike
        self.T = expiry

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

