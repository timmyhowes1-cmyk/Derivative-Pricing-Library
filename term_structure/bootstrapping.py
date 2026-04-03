from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt
from term_structure.curves import *
from term_structure.date_conventions import DateConvention
from instruments.rates import Deposit, Bond, InterestRateSwap

class RateHelper(ABC):
    @abstractmethod
    def npv(self, curve:YieldCurve):
        pass

def bootstrap_curve(helpers:list, reference_date:dt.date, date_convention:DateConvention,
                    curve_cls:YieldCurve=PiecewiseLinearDiscountCurve, compounding:str="continuous"):
    time_helper = lambda h: date_convention.get_year_fraction(reference_date, h.instrument.get_maturity_date())
    helpers = sorted(helpers, key=time_helper)
    times = []
    discount_factors = []

    for h in helpers:
        t_new = time_helper(h)

        def objective(df_new):
            trial_times = np.array(times + [t_new], dtype=float)
            trial_dfs = np.array(discount_factors + [df_new], dtype=float)
            trial_curve = curve_cls(reference_date=reference_date, date_convention=date_convention, times=trial_times,
                                    discount_factors=trial_dfs, compounding=compounding)
            return helper.npv(trial_curve)

        df_new = opt.brentq(objective, 1e-8, 2)
        times.append(t_new)
        discount_factors.append(df_new)

    return curve_cls(reference_date=reference_date, date_convention=date_convention, times=np.array(times, dtype=float),
                     discount_factors=np.array(discount_factors, dtype=float))


class DepositHelper(RateHelper):
    def __init__(self, market_rate:float, deposit:Deposit):
        self.instrument = deposit
        self.market_rate = market_rate

    def maturity_time(self, curve:YieldCurve):
        return curve.get_time_from_reference(self.instrument.get_maturity_date())

    def npv(self, curve:YieldCurve):
        t = curve.get_time_from_reference(self.instrument.get_maturity_date())
        df = curve.get_discount_factor(t)
        par_value = self.instrument.notional * (1 + self.market_rate * t)
        return par_value * df - self.notional

class BondHelper(RateHelper):
    def __init__(self, market_price:float, bond:Bond):
        self.instrument = bond
        self.market_price = market_price

    def maturity_time(self, curve:YieldCurve):
        return curve.get_time_from_reference(self.instrument.get_maturity_date())

    def npv(self, curve:YieldCurve):
        engine = BondDiscountingEngine(curve)
        model_price = engine.get_price(self.instrument)["value"]
        return model_price - self.market_price

class SwapHelper(RateHelper):
    def __init__(self, market_rate:float, swap:InterestRateSwap):
        self.instrument = swap
        self.market_rate = market_rate

    def maturity_time(self, curve:YieldCurve):
        return curve.get_time_from_reference(self.instrument.get_maturity_date())

    def npv(self, curve:YieldCurve):
        engine = SwapDiscountingEngine(curve)
        npv = engine.get_price(self.instrument)["value"]
        return npv