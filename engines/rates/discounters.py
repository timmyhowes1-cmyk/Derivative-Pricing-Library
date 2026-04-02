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
    def __init__(self, curve:YieldCurve):
        super().__init__(curve)

    def get_price(self, bond:Bond):
        value = alculate_leg_npv(bond.cashflows, self.curve)
        return {"value": value}

    def get_pv01(self, bond:Bond, bump:float=0.0001):
        curve_down = self.parallel_shift(-bump)
        curve_up = self.parallel_shift(bump)
        v_down = calculate_leg_npv(bond.cashflows, curve_down)
        v_up = calculate_leg_npv(bond.cashflows, curve_up)
        return {"PV01": (v_down - v_up) / 2}

class SwapDiscountingEngine(DiscountingEngine):
    def __init__(self, curve:YieldCurve):
        super().__init__(curve)

    def get_price(self, swap:InterestRateSwap):
        fixed_leg, floating_leg = swap.get_legs()
        fixed_npv = self.calculate_leg_npv(fixed_leg, self.curve)
        floating_npv = self.calculate_leg_npv(floating_leg, self.curve)

        value = swap.fixed_leg_sign() * fixed_npv + swap.floating_leg_sign() * floating_npv
        return {"value": value}

    def get_pv01(self, swap:InterestRateSwap):
        curve_down = self.parallel_shift(-bump)
        curve_up = self.parallel_shift(bump)

        engine_down = SwapDiscountingEngine(curve_down)
        engine_up = SwapDiscountingEngine(curve_up)

        pv_down = engine_down.get_price(swap)["value"]
        pv_up = engine_up.get_price(swap)["value"]

        return {"PV01": (pv_down - pv_up) / 2}


def calculate_leg_npv(leg:Leg, curve:YieldCurve):
    total = 0
    for cf in leg.cashflows:
        t = curve.get_time_from_reference(cf.payment_date)
        df = curve.get_discount_factor(t)  # or discount(t) via time_from_reference
        total += cf.amount() * df
    return total
