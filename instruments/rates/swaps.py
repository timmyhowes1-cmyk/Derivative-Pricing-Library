from term_structure.cashflow_schedule import Schedule
from term_structure.cashflows import Leg, Redemption, make_fixed_leg, make_floating_leg
from term_structure.indices import InterestRateIndex
from term_structure.date_convention import DateConvention

class InterestRateSwap():
    def __init__(self, fixed_leg:Leg, floating_leg:Leg, pay_fixed:bool=True):
        self.fixed_leg = fixed_leg
        self.floating_leg = floating_leg
        self.pay_fixed = pay_fixed

    def get_legs(self):
        return self.fixed_leg, self.floating_leg

    def fixed_leg_sign(self):
        return -1 if self.pay_fixed else 1

    def floating_leg_sign(self):
        return 1 if self.pay_fixed else -1

def make_vanilla_swap(notional:float, fixed_schedule:Schedule, floating_schedule:Schedule, fixed_rate:float, floating_index:InterestRateIndex,
                      fixed_date_convention:DateConvention, spread:float=0.0, pay_fixed:bool=True):
    fixed_leg = make_fixed_leg(schedule=fixed_schedule, notional=notional, rate=fixed_rate, date_conventionfixed_date_convention)
    floating_leg = make_floating_leg(schedule=floating_schedule, notional=notional, index=floating_index, spread=spread)

    return InterestRateSwap(fixed_leg=fixed_leg, floating_leg=floating_leg, pay_fixed=pay_fixed)