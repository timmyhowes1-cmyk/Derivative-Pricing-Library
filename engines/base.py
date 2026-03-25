from abc import ABC, abstractmethod

class Engine(ABC):
    def __init__(self, quiet=False):
        self.quiet = quiet

    @abstractmethod
    def get_price(self, instrument, model):
        pass

    def get_greeks(self, instrument, model, greek_type):
        pass


