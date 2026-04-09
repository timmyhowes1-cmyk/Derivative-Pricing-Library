import datetime as dt
from term_structure.date_convention import DateConvention
from term_structure.cashflows import Leg, Redemption
from .bonds import Bond

class Deposit(Bond):
    ...

class SimpleDeposit(Deposit):
    def __init__(self, notional:float, start_date:dt.date, maturity_date:dt.date, rate:float, date_convention:DateConvention):
        accrual = date_convention.get_year_fraction(start_date, maturity_date)
        maturity_amount = notional * (1 + rate * accrual)

        cashflows = Leg([Redemption(payment_date=maturity_date, notional=maturity_amount)])

        super().__init__(cashflows=cashflows)
        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.rate = rate
        self.date_convention = date_convention
        self.accrual = accrual