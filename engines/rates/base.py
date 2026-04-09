from abc import ABC, abstractmethod
from term_structure.curves import YieldCurve

class Engine(ABC):
    @abstractmethod
    def get_price(self, instrument):
        pass

class DiscountingEngine(Engine):
    def __init__(self, curve:YieldCurve):
        self.curve = curve
