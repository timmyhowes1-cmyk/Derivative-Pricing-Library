from instruments.base import Option
import numpy as np

class Asian(Option):
    def __init__(self, option_type, strike, expiry, average_type="arithmetic", strike_type="fixed", **kwargs):
        super().__init__(option_type, strike, expiry)
        self.average_type = average_type
        self.strike_type = strike_type

        self.raiseStrikeError()

    def payoff(self, price_path):
        if np.any(price_path < 0):
            raise ValueError("Price path must be non-negative")

        if self.average_type == "arithmetic":
            avg = np.mean(price_path, axis=np.ndim(price_path)-1)
        else:
            avg = np.exp(np.mean(np.log(price_path), axis=np.ndim(price_path)-1))

        if self.strike_type == "fixed":
            return np.maximum(avg - self.K, 0) if self.option_type == "call" \
                else np.maximum(self.K - avg, 0)
        return np.maximum(price_path[..., -1] - avg, 0) if self.option_type == "call" \
            else np.maximum(avg - price_path[..., -1], 0)


class Lookback(Option):
    def __init__(self, option_type, strike, expiry, strike_type="fixed", **kwargs):
        super().__init__(option_type, strike, expiry)
        self.strike_type = strike_type

        if self.strike_type == "fixed":
            self.raiseStrikeError()

    def payoff(self, price_path):
        self.raisePriceError(price_path)
        if self.strike_type == "fixed":
            return np.maximum(np.max(price_path, axis=np.ndim(price_path)-1) - self.K, 0) if self.option_type == "call" \
                else np.maximum(self.K - np.min(price_path, axis=np.ndim(price_path)-1), 0)
        else:
            return np.maximum(np.max(price_path, axis=np.ndim(price_path)-1) - price_path[:, -1], 0) if self.option_type == "call" \
                else np.maximum(price_path[:, -1] - np.min(price_path, axis=np.ndim(price_path)-1), 0)

class Digital(Option):
    def __init__(self, option_type, strike, expiry, cash_payoff=1, **kwargs):
        super().__init__(option_type, strike, expiry)
        self.cash_payoff = cash_payoff

        self.raiseStrikeError()
        if self.cash_payoff < 0:
            raise ValueError("Cash payoff must be non-negative")

    def payoff(self, price_path):
        self.raisePriceError(price_path)
        if isinstance(price_path, np.ndarray):
            spot_price = price_path[..., -1]
        else:
            spot_price = price_path

        return self.cash_payoff * (spot_price - self.K > 0) if self.option_type == "call" \
            else self.cash_payoff * (self.K - spot_price > 0)


class Barrier(Option):
    def __init__(self, option_type, strike, expiry, b=None, up=True, out=True, **kwargs):
        super().__init__(option_type, strike, expiry)
        self.b = self.K if b is None else b
        self.up = up
        self.out = out
        self.raiseStrikeError()
        if self.b < 0:
            raise ValueError("Barrier must be non-negative")


    def _get_barrier_flag(self, price_path):
        extreme_price = np.max(price_path, axis=np.ndim(price_path)-1) if self.up \
            else np.min(price_path, axis=np.ndim(price_path)-1)

        if self.up and self.out:
            return extreme_price <= self.b
        elif not self.up and not self.out:
            return extreme_price < self.b
        elif self.up and not self.out:
            return extreme_price > self.b
        else:
            return extreme_price >= self.b

    def payoff(self, price_path):
        self.raisePriceError(price_path)
        flag = self._get_barrier_flag(price_path)
        if isinstance(price_path, np.ndarray):
            spot_price = price_path[..., -1]
        else:
            spot_price = price_path
        return flag * np.maximum(spot_price - self.K, 0) if self.option_type == "call" \
            else flag * np.maximum(self.K - spot_price, 0)
