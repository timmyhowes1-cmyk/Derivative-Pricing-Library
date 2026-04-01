from instruments.equity.base import EquityOption
from utils.math_utils import expanding_mean_axis1
import numpy as np
from typing import Union

class Asian(EquityOption):
    def __init__(self, strike, expiry, call, arithmetic_mean:bool=True, fixed_strike:bool=True, **kwargs):
        super().__init__(strike, expiry, call)
        self.arithmetic_mean = arithmetic_mean
        self.fixed_strike = fixed_strike
        self.raiseStrikeError()

    def payoff(self, price_path:Union[float, np.ndarray]):
        if np.any(price_path < 0):
            raise ValueError("Price path must be non-negative")
        if self.arithmetic_mean:
            avg = np.mean(price_path, axis=np.ndim(price_path) - 1) if self.european \
                else expanding_mean_axis1(price_path, arithmetic_mean=arithmetic_mean)
        else:
            avg = np.exp(np.mean(np.log(price_path), axis=np.ndim(price_path) - 1)) if self.european \
                else expanding_mean_axis1(price_path, arithmetic_mean=arithmetic_mean)
        if self.fixed_strike:
            return np.maximum(avg - self.K, 0) if self.call \
                else np.maximum(self.K - avg, 0)
        spot_to_use = self.get_payoff_spot(price_path)
        return np.maximum(spot_to_use - avg, 0) if self.call \
            else np.maximum(avg - spot_to_use, 0)


class Lookback(EquityOption):
    def __init__(self, strike, expiry, call, fixed_strike:bool=True, **kwargs):
        super().__init__(strike, expiry, call)
        self.fixed_strike = fixed_strike
        if self.fixed_strike:
            self.raiseStrikeError()

    def payoff(self, price_path: Union[float, np.ndarray]):
        self.raisePriceError(price_path)
        if self.call:
            extreme = np.max(price_path, axis=np.ndim(price_path) - 1) if self.european \
                else np.maximum.accumulate(price_path, axis=1)
        else:
            extreme = np.min(price_path, axis=np.ndim(price_path) - 1) if self.european \
                else np.minimum.accumulate(price_path, axis=1)
        if self.fixed_strike:
            return np.maximum(extreme - self.K, 0) if self.call \
                else np.maximum(self.K - extreme, 0)
        else:
            spot_to_use = self.get_payoff_spot(price_path)
            return np.maximum(extreme - spot_to_use, 0) if self.call \
                else np.maximum(spot_to_use - extreme, 0)

class Digital(EquityOption):
    def __init__(self, strike, expiry, call, cash_payoff:float=1, **kwargs):
        super().__init__(strike, expiry, call)
        self.cash_payoff = cash_payoff
        self.raiseStrikeError()
        if self.cash_payoff < 0:
            raise ValueError("Cash payoff must be non-negative")

    def payoff(self, price_path:Union[float, np.ndarray]):
        self.raisePriceError(price_path)
        spot_to_use = self.get_payoff_spot(price_path)
        return self.cash_payoff * (spot_to_use - self.K > 0) if self.call \
            else self.cash_payoff * (self.K - spot_to_use > 0)

class Barrier(EquityOption):
    def __init__(self, strike, expiry, call, b:float=None, up:bool=True, out:bool=True, **kwargs):
        super().__init__(strike, expiry, call)
        self.b = self.K if b is None else b
        self.up = up
        self.out = out
        self.raiseStrikeError()
        if self.b < 0:
            raise ValueError("Barrier must be non-negative")

    def _get_barrier_flag(self, price_path:Union[float, np.ndarray]):
        if self.european:
            extreme = np.max(price_path, axis=np.ndim(price_path) - 1) if self.up \
                else np.min(price_path, axis=np.ndim(price_path) - 1)
        else:
            extreme = np.maximum.accumulate(price_path, axis=1) if self.up \
                else np.mininum.accumulate(price_path, axis=1)
        if self.up and self.out:
            return extreme <= self.b
        elif not self.up and not self.out:
            return extreme < self.b
        elif self.up and not self.out:
            return extreme > self.b
        else:
            return extreme >= self.b

    def payoff(self, price_path:Union[float, np.ndarray]):
        self.raisePriceError(price_path)
        flag = self._get_barrier_flag(price_path)
        spot_to_use = self.get_payoff_spot(price_path)
        return flag * np.maximum(spot_to_use - self.K, 0) if self.call \
            else flag * np.maximum(self.K - spot_to_use, 0)
