from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt
from .curves import *
from .date_convention import DateConvention
from engines.rates import calculate_leg_npv
from instruments.rates.deposits import Deposit, SimpleDeposit
from instruments.rates.bonds import Bond
from instruments.rates.swaps import InterestRateSwap
from instruments.rates.futures import InterestRateFutures

def bootstrap_curve(helpers:list, reference_date:dt.date, date_convention:DateConvention,
                    curve_cls:YieldCurve=PiecewiseLinearDiscountCurve, compounding:str="continuous"):
    time_helper = lambda h: date_convention.get_year_fraction(reference_date, h.instrument.get_maturity_date())
    helpers = sorted(helpers, key=time_helper)
    times = []
    discount_factors = []

    for helper in helpers:
        t_new = time_helper(helper)

        def objective(df_new):
            trial_times = np.array(times + [t_new], dtype=float)
            trial_dfs = np.array(discount_factors + [df_new], dtype=float)
            trial_curve = curve_cls(reference_date=reference_date, date_convention=date_convention, times=trial_times,
                                    discount_factors=trial_dfs, compounding=compounding)
            print(f"{helper.__class__.__name__}: df_new={df_new:.6f}, npv={helper.npv(trial_curve):.2e}")
            return helper.npv(trial_curve)

        df_new = opt.brentq(objective, 1e-8, 2, xtol=1e-10, rtol=1e-10)
        times.append(t_new)
        discount_factors.append(df_new)

    return curve_cls(reference_date=reference_date, date_convention=date_convention, times=np.array(times, dtype=float),
                     discount_factors=np.array(discount_factors, dtype=float), compounding=compounding)


class RateHelper(ABC):
    def __init__(self, instrument):
        self.instrument = instrument

    @abstractmethod
    def npv(self, curve:YieldCurve):
        pass

    def maturity_time(self, curve:YieldCurve):
        return curve.get_time_from_reference(self.instrument.get_maturity_date())

class DepositHelper(RateHelper):
    def __init__(self, market_rate:float, instrument:Deposit):
        self.instrument = instrument
        self.market_rate = market_rate
        super().__init__(instrument=instrument)

    def npv(self, curve:YieldCurve):
        t = curve.get_time_from_reference(self.instrument.get_maturity_date())
        df = curve.get_discount_factor(t)
        par_value = self.instrument.notional * (1 + self.market_rate * t)
        return par_value * df - self.instrument.notional

class BondHelper(RateHelper):
    def __init__(self, market_price:float, instrument:Bond):
        self.instrument = instrument
        self.market_price = market_price
        super().__init__(instrument=instrument)

    def npv(self, curve:YieldCurve):
        engine = BondDiscountingEngine(curve)
        model_price = engine.get_price(self.instrument)["value"]
        return model_price - self.market_price

class SwapHelper(RateHelper):
    def __init__(self, instrument:InterestRateSwap):
        self.instrument = instrument
        super().__init__(instrument=instrument)

    def npv(self, curve:YieldCurve):
        fixed_npv = calculate_leg_npv(leg=self.instrument.fixed_leg, curve=curve)
        float_npv = calculate_leg_npv(leg=self.instrument.floating_leg, curve=curve)

        return fixed_npv - float_npv

class FuturesHelper(RateHelper):
    def __init__(self, instrument:InterestRateFutures):
        self.instrument = instrument
        super().__init__(instrument=instrument)

    def npv(self, curve:YieldCurve):

        implied_fwd_rate = self.instrument.implied_forward_rate()
        # implied fwd rate is quoted in simple terms so get curve forward rate in simple terms
        df1 = curve.get_discount_factor(self.instrument.reference_start_date)
        df2 = curve.get_discount_factor(self.instrument.reference_end_date)
        t_accrual = curve.date_convention.get_year_fraction(self.instrument.reference_start_date, self.instrument.reference_end_date)

        curve_fwd_rate = (df1 / df2 - 1) / t_accrual
        return implied_fwd_rate - curve_fwd_rate