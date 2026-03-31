import numpy as np
from instruments.equity.base import EquityOption
from typing import Union

class Vanilla(EquityOption):
    def payoff(self, price_path:Union[float, np.ndarray]):
        self.raisePriceError(price_path)
        self.raiseStrikeError()
        spot_to_use = self.get_payoff_spot(price_path)
        return np.maximum(spot_to_use - self.K, 0) if self.call else np.maximum(self.K - spot_to_use, 0)



