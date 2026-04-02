import copy
from abc import ABC, abstractmethod
import numpy as np
import bisect
import datetime as dt
from term_structure.date_convention import DateConvention

class YieldCurve(ABC):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention, compounding:str="continuous"):
        self.reference_date = reference_date
        self.date_convention = date_convention
        self.compounding = compounding

    @abstractmethod
    def get_discount_factor(self, t:float):
        pass

    def get_zero_rate(self, t:float):
        if t <= 0:
            raise ValueError("t must be positive for zero_rate")
        df = self.get_discount_factor(t)

        if self.compounding == "continuous":
            return -np.log(df) / t
        elif self.compounding == "annual":
            return (df ** (-1 / t)) - 1
        else:
            raise ValueError(f"Unknown compounding: {self.compounding}")

    def get_forward_rate(self, t1:float, t2:float):
        if t2 <= t1:
            raise ValueError("Need t2 > t1 for forward_rate")
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        t_diff = t2 - t1

        if self.compounding == "continuous":
            return (np.log(df1) - np.log(df2)) / t_diff
        elif self.compounding == "annual":
            return (df1 / df2) ** (1 / t_diff) - 1
        else:
            raise ValueError(f"Unknown compounding: {self.compounding}")

    def get_time_from_reference(self, date):
        return self.date_convention.get_year_fraction(self.reference_date, date)

class FlatYieldCurve(YieldCurve):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention, rate:float, compounding:str="continuous"):
        super().__init__(reference_date, date_convention, compounding)
        self.rate = rate

    def get_discount_factor(self, t:float):
        if self.compounding == "continuous":
            return np.exp(-self.rate * t)
        elif self.compounding == "annual":
            return (1 + self.rate) ** -t
        else:
            raise ValueError(f"Unknown compounding: {self.compounding}")

    def parallel_shift(self, bump):
        new_curve = copy.deepcopy(self)
        new_curve.rate += bump
        return new_curve


class PiecewiseLinearDiscountCurve(YieldCurve):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention, times:np.ndarray, discount_factors:np.ndarray, compounding:str="continuous"):
        super().__init__(reference_date, date_convention, compounding)
        if len(times) != len(discount_factors):
            raise ValueError("Times and discount_factors must have same length")
        if not np.all(times[:-1] < times[1:]):
            raise ValueError("Times must be strictly increasing")
        self.times = times
        self.discount_factors = discount_factors

    def get_discount_factor(self, t:float):
        if t <= self.times[0]:
            return self.discount_factors[0]
        if t >= self.times[-1]:
            return self.discount_factors[-1]

        i = bisect.bisect_right(self.times, t) - 1
        t1, t2 = self.times[i], self.times[i+1]
        df1, df2 = self.discount_factors[i], self.discount_factors[i+1]
        w = (t - t1) / (t2 - t1)
        return np.exp(np.log(df1) + w * (np.log(df2) - np.log(df1)))

    def parallel_shift(self, bump):
        new_curve = copy.deepcopy(self)
        rates = [new_curve.get_zero_rate(t) + bump for t in new_curve.times]
        new_dfs = []
        for r, t in zip(rates, new_curve.times):
            if new_curve.compounding == "continuous":
                new_dfs.append(np.exp(-r * t))
            elif new_curve.compounding == "annual":
                new_dfs.append((1 + r) ** -t)
            else:
                raise ValueError(f"Unknown compounding: {new_curve.compounding}")
        new_curve.discount_factors = np.array(new_dfs)
        return new_curve
