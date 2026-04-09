from dateutil import relativedelta
import dateutil
import datetime as dt
from term_structure.cashflow_schedule import Schedule
from term_structure.curves import YieldCurve
from term_structure.cashflows import Leg, Redemption, make_fixed_leg, make_floating_leg
from term_structure.indices import InterestRateIndex
from term_structure.date_convention import DateConvention

class InterestRateSwap:
    def __init__(self, fixed_leg:Leg, floating_leg:Leg, pay_fixed:bool=True):
        self.fixed_leg = fixed_leg
        self.floating_leg = floating_leg
        self.pay_fixed = pay_fixed

    def get_legs(self):
        return self.fixed_leg, self.floating_leg

    def get_maturity_date(self):
        return max(self.fixed_leg.payment_dates()[-1], self.floating_leg.payment_dates()[-1])

    def fixed_leg_sign(self):
        return -1 if self.pay_fixed else 1

    def floating_leg_sign(self):
        return 1 if self.pay_fixed else -1

class FRA(InterestRateSwap):
    def __init__(self, notional:float, settlement_date:dt.date, fixed_leg:Leg, floating_leg:Leg, pay_fixed:bool=True):
        self.settlement_date = settlement_date
        super().__init__(fixed_leg, floating_leg, pay_fixed)

def make_vanilla_swap(notional:float, fixed_schedule:Schedule, floating_schedule:Schedule, fixed_rate:float, floating_index:InterestRateIndex,
                      fixed_date_convention:DateConvention, spread:float=0.0, pay_fixed:bool=True):
    fixed_leg = make_fixed_leg(schedule=fixed_schedule, notional=notional, rate=fixed_rate, date_convention=fixed_date_convention)
    floating_leg = make_floating_leg(schedule=floating_schedule, notional=notional, index=floating_index, spread=spread)
    return InterestRateSwap(fixed_leg=fixed_leg, floating_leg=floating_leg, pay_fixed=pay_fixed)

def make_fra(notional:float, settlement_date:dt.date, accrual_start_date:dt.date, accrual_end_date:dt.date, fixed_rate:float, index:InterestRateIndex,
                   fixed_date_convention:DateConvention, pay_fixed:bool=True, spread:float=0.0):
    schedule = Schedule(start_date=accrual_start_date, end_date=accrual_end_date,
                        months_per_period=months_between(accrual_start_date, accrual_end_date))
    fixed_leg = make_fixed_leg(schedule=schedule, notional=notional, rate=fixed_rate, date_convention=fixed_date_convention)
    floating_leg = make_floating_leg(schedule=schedule, notional=notional, index=index, spread=spread)
    return FRA(notional=notional, settlement_date=settlement_date, fixed_leg=fixed_leg, floating_leg=floating_leg, pay_fixed=pay_fixed)

def par_swap_rate(schedule:Schedule, curve:YieldCurve):
    maturity = schedule.end_date
    t_end = curve.get_time_from_reference(maturity)
    numerator, denominator = 0, 0
    for accrual_start_date, accrual_end_date in schedule.periods():
        fwd_rate = curve.get_forward_rate(accrual_start_date, accrual_end_date)
        t_accrual = curve.date_convention.get_year_fraction(accrual_start_date, accrual_end_date)
        t_end = curve.get_time_from_reference(accrual_end_date)
        df = curve.get_discount_factor(t_end)
        numerator += t_accrual * fwd_rate * df
        denominator += t_accrual * df
    return numerator / denominator

def months_between(start:dt.date, end:dt.date):
    delta = dateutil.relativedelta.relativedelta(end, start)
    return delta.years * 12 + delta.months