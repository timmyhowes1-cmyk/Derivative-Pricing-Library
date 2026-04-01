from abc import ABC, abstractmethod
from models.equity import Model
from instruments.equity import EquityOption
from typing import Union

class Engine(ABC):
    @abstractmethod
    def get_price(self, instrument:EquityOption, model:Model):
        pass

    def get_greeks(self, instrument:EquityOption, model:Model, greek_type:Union[list, str]):
        pass


