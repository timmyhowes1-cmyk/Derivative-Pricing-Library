import datetime as dt
from abc import ABC, abstractmethod
from term_structure.cashflow_schedule import add_months
from term_structure.date_convention import DateConvention
from term_structure.curves import YieldCurve

class InterestRateIndex(ABC):
    def __init__(self, name, date_convention:DateConvention, forward_curve:YieldCurve):
        self.name = name
        self.date_convention = date_convention
        self.forward_curve = forward_curve

    def year_fraction(self, start_date:dt.date, end_date:dt.date):
        return self.day_count.year_fraction(start_date, end_date)

    @abstractmethod
    def forward_rate(self, start_date:dt.date, end_date:dt.date):
        pass


class Ibor(InterestRateIndex):
    def __init__(self, name, tenor_months, date_convention, forward_curve, fixing_days=0):
        super().__init__(name, date_convention, forward_curve)
        self.tenor_months = tenor_months
        self.fixing_days = fixing_days

    def maturity_date(self, start_date:dt.date):
        return add_months(start_date, self.tenor_months)

    def forward_rate(self, start_date:dt.date, end_date:dt.date = None):
        if end_date is None:
            end_date = self.maturity_date(start_date)

        t1 = self.forward_curve.time_from_reference(start_date)
        t2 = self.forward_curve.time_from_reference(end_date)

        return self.forward_curve.forward_rate(t1, t2, compounding="annual")