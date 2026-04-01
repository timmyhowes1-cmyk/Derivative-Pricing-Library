from abc import abstractmethod
from term_structure.yield_curve import YieldCurve
from cashflows import Leg
from engines.rates.base import DiscountingEngine
from instruments.rates.bonds import Bond
from instruments.rates.swaps import InterestRateSwap

class DiscountingEngine(Engine):
    def __init__(self, curve:YieldCurve):
        self.curve = curve

    def calculate_leg_npv(leg:Leg, curve:YieldCurve):
        total = 0
        for cf in leg.cashflows:
            t = curve.get_time_from_reference(cf.payment_date)
            df = curve.get_discount_factor(t)  # or discount(t) via time_from_reference
            total += cf.amount() * df
        return total

class BondDiscountingEngine(DiscountingEngine):
    def __init__(self, curve:YieldCurve):
        super().__init__(curve)

    def get_price(self, bond:Bond):
        value = self.calculate_leg_npv(bond.cashflows, self.curve)
        return {"value": value}

class SwapDiscountingEngine(DiscountingEngine):
    def __init__(self, curve:YieldCurve):
        super().__init__(curve)

    def get_price(self, swap:InterestRateSwap):
        fixed_leg, floating_leg = swap.get_legs()

        fixed_npv = self.calculate_leg_npv(fixed_leg, self.curve)
        float_npv = self.calculate_leg_npv(floating_leg, self.curve)

        value = swap.fixed_leg_sign() * fixed_npv + swap.floating_leg_sign() * float_npv
        return {"value": value}
