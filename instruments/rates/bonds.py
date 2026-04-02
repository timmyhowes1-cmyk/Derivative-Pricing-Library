from term_structure.cashflow_schedule import Schedule
from term_structure.cashflows import Leg, Redemption, make_fixed_leg, make_floating_leg
from term_structure.indices import InterestRateIndex
from term_structure.date_convention import DateConvention

class Bond():
    def __init__(self, cashflows:Leg):
        self.cashflows = cashflows

    def get_maturity(self):
        return max(cashflows.payment_dates())

    def get_payment_dates(self):
        return cashflows.payment_dates()

class FixedRateBond(Bond):
    def __init__(self, notional, schedule:Schedule, coupon_rate:float, date_convention:DateConvention):
        coupons = make_fixed_leg(schedule=schedule, notional=notional, rate=coupon_rate, date_convention=date_convention)
        coupons.cashflows.append(Redemption(payment_date=schedule.dates[-1], notional=notional))

        super().__init__(cashflows=coupons)
        self.notional = notional
        self.coupon_rate = coupon_rate
        self.date_convention = date_convention

class PlainFRN(Bond):
    def __init__(self, notional, schedule:Schedule, index:InterestRateIndex, spread:float=0.0):
        coupons = make_floating_leg(schedule=schedule, notional=notional, index=index, spread=spread)
        coupons.cashflows.append(Redemption(payment_date=schedule.dates[-1], notional=notional))

        super().__init__(cashflows=coupons)
        self.notional = notional
        self.index = index
        self.spread = spread


