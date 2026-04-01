import datetime as dt
from term_structure.cashflow_schedule import Schedule
from term_structure.date_convention import DateConvention
from term_structure.indices import InterestRateIndex
from abc import ABC, abstractmethod

class Cashflow(ABC):
    def __init__(self, payment_date:dt.date):
        self.payment_date = payment_date

    @abstractmethod
    def amount(self):
        pass

class FixedRateCoupon(Cashflow):
    def __init__(self, payment_date, notional:float, rate:float, accrual_start_date:dt.date, accrual_end_date:dt.date, date_convention:DateConvention):
        super().__init__(payment_date)
        self.notional = notional
        self.rate = rate
        self.accrual_start_date = accrual_start_date
        self.accrual_end_date = accrual_end_date
        self.date_convention = date_convention

    def amount(self):
        return self.notional * self.rate * self.date_convention.get_year_fraction(self.accrual_start_date, self.accrual_end_date)

class Redemption(Cashflow):
    def __init__(self, payment_date, notional:float):
        super().__init__(payment_date)
        self.notional = notional

    def amount(self):
        return self.notional

class FloatingRateCoupon(Cashflow):
    def __init__(self, notional:float, accrual_start_date:dt.date, accrual_end_date:dt.date, payment_date:dt.date, index:InterestRateIndex, spread:float=0.0):
        super().__init__(payment_date)
        self.notional = notional
        self.accrual_start = accrual_start_date
        self.accrual_end = accrual_end_date
        self.index = index
        self.spread = spread

    def rate(self):
        fwd = self.index.forward_rate(self.accrual_start_date, self.accrual_end_date)
        return fwd + self.spread

    def amount(self):
        return self.notional * self.rate() * self.index.get_year_fraction(self.accrual_start_date, self.accrual_end_date)

class Leg():
    def __init__(self, cashflows:Cashflow):
        self.cashflows = list(cashflows)

    def payment_dates(self):
        return [cf.payment_date for cf in self.cashflows]

    def amounts(self):
        return [cf.cf_amount() for cf in self.cashflows]

def make_fixed_leg(schedule:Schedule, notional:float, rate:float, date_convention:DateConvention):
    cashflows = []
    for accrual_start_date, accrual_end_date in schedule.periods():
        payment_date = accrual_end_date
        cf = FixedRateCoupon(payment_date=payment_date, notional=notional, rate=rate,
                             accrual_start_date=accrual_start_date, accrual_end_date=accrual_end_date, date_convention=date_convention)
        cashflows.append(cf)
    return Leg(cashflows)

def make_floating_leg(schedule:Schedule, notional:float, index:InterestRateIndex, spread:float=0.0):
    cashflows = []
    for accrual_start_date, accrual_end_date in schedule.periods():
        payment_date = accrual_end_date
        cf = FloatingRateCoupon(payment_date=payment_date, notional=notional, index=index, spread=spread,
                             accrual_start_date=accrual_start_date, accrual_end_date=accrual_end_date)
        cashflows.append(cf)
    return Leg(cashflows)
