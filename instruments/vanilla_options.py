import numpy as np
from instruments.base import Option

class European(Option):
    def payoff(self, price_path):
        self.raisePriceError(price_path)
        self.raiseStrikeError()

        if isinstance(price_path, np.ndarray):
            spot_price = price_path[..., -1]
        else:
            spot_price = price_path

        if self.option_type == "call":
            return np.maximum(spot_price - self.K, 0)
        return np.maximum(self.K - spot_price, 0)

