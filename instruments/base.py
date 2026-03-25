from abc import ABC, abstractmethod

class Option(ABC):

    def __init__(self, option_type, strike, expiry, **kwargs):
        self.option_type = option_type
        self.K = strike
        self.T = expiry

    @abstractmethod
    def payoff(self, price_path):
        pass

