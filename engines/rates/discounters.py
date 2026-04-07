import copy
from abc import abstractmethod
from term_structure.yield_curve import YieldCurve
from cashflows import Leg
from engines.rates.base import DiscountingEngine
from instruments.rates.bonds import Bond
from instruments.rates.swaps import InterestRateSwap

class DiscountingEngine(Engine):
    def __init__(self, curve:YieldCurve):
        self.curve = curve

class BondDiscountingEngine(DiscountingEngine):
    def get_price(self, bond:Bond):
        value = calculate_leg_npv(bond.cashflows, self.curve)
        return {"value": value}

    def get_pv01(self, bond:Bond):
        return {"pv01": pv01(curve=self.curve, engine=BondDiscountingEngine, instrument=bond)}

    def get_bucket_pv01(self, bond:Bond, date:dt.date):
        t = self.curve.get_time_from_reference(date)
        return {f"bucket_pv01_{t:.1f}Y": bucket_pv01(curve=self.curve, engine=BondDiscountingEngine, instrument=bond, date=date)}

    def get_modified_duration(self, bond:Bond):
        pv01 = self.get_pv01(bond)["pv01"]
        price = self.get_price(bond)["value"]
        duration = -pv01 / price * 10000 if abs(price) > 1e-8 else float('nan')
        return {"modified_duration": duration}

    def get_key_rate_duration(self, bond:Bond, date:dt.date):
        t = self.curve.get_time_from_reference(date)
        bucket_pv01 = next(iter(self.get_bucket_pv01(bond, date).values()))
        price = self.get_price(bond)["value"]
        duration = -bucket_pv01 / price * 10000 if abs(price) > 1e-8 else float('nan')
        return {f"key_rate_duration_{t:.1f}Y": duration}


class SwapDiscountingEngine(DiscountingEngine):
    def get_price(self, swap:InterestRateSwap):
        if isinstance(swap, FRA):
            return {"value": calculate_fra_npv(swap, self.curve)}

        fixed_leg, floating_leg = swap.get_legs()
        fixed_npv = calculate_leg_npv(fixed_leg, self.curve)
        floating_npv = calculate_leg_npv(floating_leg, self.curve)

        value = swap.fixed_leg_sign() * fixed_npv + swap.floating_leg_sign() * floating_npv
        return {"value": value}

    def get_pv01(self, swap:InterestRateSwap):
        return {"pv01": pv01(curve=self.curve, engine=SwapDiscountingEngine, instrument=swap)}

    def get_bucket_pv01(self, swap:InterestRateSwap, date: dt.date):
        t = self.curve.get_time_from_reference(date)
        return {f"bucket_pv01_{t:.1f}Y": bucket_pv01(curve=self.curve, engine=SwapDiscountingEngine, instrument=swap, date=date)}

class FuturesDiscountingEngine(DiscountingEngine):
    def get_price(self, futures:InterestRateFutures):
        return {"futures_price": futures.futures_price}

    def get_value(self, futures:InterestRateFutures):
        fwd_rate = futures.index.get_forward_rate(futures.reference_start, futures.reference_end)
        market_fwd = futures.implied_forward_rate()

        payoff = self.notional * (market_fwd - fwd_rate) / 100
        t_contract = self.curve.get_time_from_reference(futures.contract_date)
        return {"value": curve.get_discount_factor(t_contract) * payoff}

    def get_pv01(self, futures:InterestRateFutures):
        return {"pv01": pv01(curve=self.curve, engine=FuturesDiscountingEngine, instrument=futures)}

    def get_bucket_pv01(self, futures:InsterestRateFutures, date:dt.date):
        t = self.curve.get_time_from_reference(date)
        return {f"bucket_pv01_{t:.1f}Y": bucket_pv01(curve=self.curve, engine=FuturesDiscountingEngine, instrument=futures, date=date)}

def pv01(curve:YieldCurve, engine:DiscountingEngine, instrument:cls):
    bump = 0.0001
    curve_down = curve.parallel_shift(-bump)
    curve_up = curve.parallel_shift(bump)
    engine_down = engine(curve_down)
    engine_up = engine(curve_up)
    v_down = engine_down.get_value(instrument)["value"] if isinstance(instrument, InterestRateFutures) else \
        engine_down.get_price(instrument)["value"]
    v_up = engine_up.get_value(instrument)["value"] if isinstance(instrument, InterestRateFutures) else \
        engine_up.get_price(instrument)["value"]
    return (v_down - v_up) / 2

def bucket_pv01(curve:YieldCurve, engine:DiscountingEngine, instrument:cls, date:dt.date):
    bump = 0.0001
    curve_down = curve.key_rate_shift(date, -bump)
    curve_up = curve.key_rate_shift(date, bump)
    engine_down = engine(curve_down)
    engine_up = engine(curve_up)
    v_down = engine_down.get_value(instrument)["value"] if isinstance(instrument, InterestRateFutures) else \
        engine_down.get_price(instrument)["value"]
    v_up = engine_up.get_value(instrument)["value"] if isinstance(instrument, InterestRateFutures) else \
        engine_up.get_price(instrument)["value"]
    t = np.round(curve.get_time_from_reference(date), 2)
    return {f"bucket_pv01_{t}Y": (v_down - v_up) / 2}

def calculate_leg_npv(leg:Leg, curve:YieldCurve):
    total = 0
    for cf in leg.cashflows:
        t = curve.get_time_from_reference(cf.payment_date)
        df = curve.get_discount_factor(t)  # or discount(t) via time_from_reference
        total += cf.amount() * df
    return total

def calculate_fra_npv(fra:FRA, curve:YieldCurve):
    fwd_rate = fra.floating_leg.cashflows[0].rate()

    numerator = fra.floating_leg_sign() * fra.floating_leg.cashflows[0].amount() + fra.fixed_leg_sign() * fra.fixed_leg.cashflows[0].amount()
    t_accrual = fra.floating_leg.cashflows[0].index.date_convention.get_year_fraction(fra.floating_leg.cashflows[0].accrual_start_date, fra.floating_leg.cashflows[0].accrual_end_date)
    denominator = 1 + fwd_rate * t_accrual
    payoff = numerator / denominator

    t_settle = curve.get_time_from_reference(fra.settlement_date)
    df_settle = curve.get_discount_factor(t_settle)

    return payoff * df_settle
