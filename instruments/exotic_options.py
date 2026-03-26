from instruments.base import Option
import numpy as np

class Asian(Option):
    def __init__(self, option_type, strike, expiry, average_type="arithmetic", strike_type="fixed", **kwargs):
        super().__init__(option_type, strike, expiry)
        self.average_type = average_type
        self.strike_type = strike_type

    def payoff(self, price_path):
        if self.average_type == "arithmetic":
            avg = np.mean(price_path, axis=1)
        else:
            avg = np.exp(np.mean(np.log(price_path), axis=1))

        if self.strike_type == "fixed":
            return np.maximum(avg - self.K, 0) if self.option_type == "call" \
                else np.maximum(self.K - avg, 0)
        return np.maximum(price_path[:, -1] - avg, 0) if self.option_type == "call" \
            else np.maximum(avg - price_path[:, -1], 0)


class Lookback(Option):
    def __init__(self, option_type, strike, expiry, strike_type="fixed", **kwargs):
        super().__init__(option_type, strike, expiry)
        self.strike_type = strike_type

    def payoff(self, price_path):
        if self.strike_type == "fixed":
            return np.maximum(np.max(price_path, axis=1) - self.K, 0) if self.option_type == "call" \
                else np.maximum(self.K - np.min(price_path, axis=1), 0)
        else:
            return np.maximum(np.max(price_path, axis=1) - price_path[:, -1], 0) if self.option_type == "call" \
                else np.maximum(price_path[:, -1] - np.min(price_path, axis=1), 0)

class Digital(Option):
    def __init__(self, option_type, strike, expiry, cash_payoff=1, **kwargs):
        super().__init__(option_type, strike, expiry)
        self.cash_payoff = cash_payoff

    def payoff(self, price_path):
        if isinstance(price_path, np.ndarray):
            spot_price = price_path[:, -1]
        else:
            spot_price = price_path

        return self.cash_payoff * (spot_price - self.K > 0) if self.option_type == "call" \
            else self.cash_payoff * (self.K - spot_price > 0)


class Barrier(Option):
    def __init__(self, option_type, strike, expiry, b=None, up=True, out=True, **kwargs):
        super().__init__(option_type, strike, expiry)
        self.b = b
        self.up = up
        self.out = out

    def _get_barrier_flag(self, price_path):
        if self.b is None:
            self.b = self.K
        extreme_price = np.max(price_path, axis=1) if self.up \
            else np.min(price_path, axis=1)

        if self.up == self.out:
            return extreme_price < self.b
        return extreme_price > self.b

    def payoff(self, price_path):
        flag = self._get_barrier_flag(price_path)
        return flag * np.maximum(price_path[:, -1] - self.K, 0) if self.option_type == "call" \
            else flag * np.maximum(self.K - price_path[:, -1], 0)
