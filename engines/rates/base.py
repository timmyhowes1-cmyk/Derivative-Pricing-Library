from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def get_price(self, instrument):
        pass
