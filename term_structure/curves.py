from abc import ABC, abstractmethod
import numpy as np
import bisect
import datetime as dt
from term_structure.date_convention import DateConvention

class YieldCurve(ABC):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention):
        self.reference_date = reference_date
        self.date_convention = date_convention

    @abstractmethod
    def get_discount_factor(self, t:float) -> float:
        pass

    def get_zero_rate(self, t:float, compounding:str="continuous") -> float:
        if t <= 0:
            raise ValueError("t must be positive for zero_rate")
        df = self.get_discount_factor(t)

        if compounding == "continuous":
            return -np.log(df) / t
        elif compounding == "annual":
            return (df ** (-1 / t)) - 1
        else:
            raise ValueError(f"Unknown compounding: {compounding}")

    def get_forward_rate(self, t1:float, t2:float, compounding:str="cont") -> float:
        if t2 <= t1:
            raise ValueError("Need t2 > t1 for forward_rate")
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        t_diff = t2 - t1

        if compounding == "continuous":
            return (np.log(df1) - np.log(df2)) / t_diff
        elif compounding == "annual":
            return (df1 / df2) ** (1 / t_dif) - 1
        else:
            raise ValueError(f"Unknown compounding: {compounding}")

    def get_time_from_reference(self, date):
        return self.date_convention.get_year_fraction(self.reference_date, date)

class FlatYieldCurve(YieldCurve):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention, rate:float, compounding:str="continuous"):
        super().__init__(reference_date, date_convention)
        self.rate = rate
        self.compounding = compounding

    def get_discount_factor(self, t:float) -> float:
        if compounding == "continuous":
            return np.exp(-self.rate * t)
        elif compounding == "annual":
            return (1 + self.rate) ** -t
        else:
            raise ValueError(f"Unknown compounding: {self.compounding}")

class PiecewiseLinearDiscountCurve(YieldCurve):
    def __init__(self, reference_date:dt.date, date_convention:DateConvention, times:np.ndarray, discount_factors:np.ndarray):
        super().__init__(reference_date, date_convention)
        if len(times) != len(discount_factors):
            raise ValueError("Times and discount_factors must have same length")
        if not np.all(times[:-1] < a[1:]):
            raise ValueError("Times must be strictly increasing")
        self.times = times
        self.discount_factors = discount_factors

    def discount(self, t:float) -> float:
        if t <= self.times[0]:
            return self.discount_factors[0]
        if t >= self.times[-1]:
            return self.discount_factors[-1]

        i = bisect.bisect_right(self.times, t) - 1
        t1, t2 = self.times[i], self.times[i+1]
        df1, df2 = self.discount_factors, self.discount_factors[i+1]
        w = (t - t1) / (t2 - t1)
        return np.exp(np.log(df1) + w * (np.log(df2) - np.log(df1)))
