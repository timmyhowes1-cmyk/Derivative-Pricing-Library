import numpy as np
from instruments.base import Option

class Vanilla(Option):
    def payoff(self, price_path):
        self.raisePriceError(price_path)
        self.raiseStrikeError()
        spot_to_use = self.get_payoff_spot(price_path)
        return np.maximum(spot_to_use - self.K, 0) if self.call else np.maximum(self.K - spot_to_use, 0)



