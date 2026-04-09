import datetime as dt
from abc import ABC, abstractmethod
from .cashflow_schedule import add_months
from .date_convention import DateConvention
from .curves import YieldCurve

class InterestRateIndex(ABC):
    def __init__(self, name, date_convention:DateConvention, forward_curve:YieldCurve):
        self.name = name
        self.date_convention = date_convention
        self.forward_curve = forward_curve

    def year_fraction(self, start_date:dt.date, end_date:dt.date):
        return self.day_count.year_fraction(start_date, end_date)

    @abstractmethod
    def get_forward_rate(self, start_date:dt.date, end_date:dt.date):
        pass


class Ibor(InterestRateIndex):
    def __init__(self, name:str, tenor_months:int, date_convention:DateConvention, forward_curve:YieldCurve, fixing_days=0):
        super().__init__(name, date_convention, forward_curve)
        self.tenor_months = tenor_months
        self.fixing_days = fixing_days

    def maturity_date(self, start_date:dt.date):
        return add_months(start_date, self.tenor_months)

    def get_forward_rate(self, start_date:dt.date, end_date:dt.date = None):
        if end_date is None:
            end_date = self.maturity_date(start_date)

        t1 = self.forward_curve.get_time_from_reference(start_date)
        t2 = self.forward_curve.get_time_from_reference(end_date)

        return self.forward_curve.get_forward_rate(t1, t2, compounding=self.forward_curve.compounding)


class OvernightIndex(InterestRateIndex):
    def __init__(self, name:str, date_convention:DateConvention, forward_curve:YieldCurve, overnight_tenor_days:int=1, fixing_days:int=0):
        super().__init__(name, date_convention, forward_curve)
        self.overnight_tenor_days = overnight_tenor_days  # Usually 1
        self.fixing_days = fixing_days

    def maturity_date(self, start_date: dt.date):
        return self.date_convention.adjust(start_date + dt.timedelta(days=self.overnight_tenor_days))

    def get_forward_rate(self, accrual_start:dt.date, accrual_end:dt.date):
        t1 = self.forward_curve.get_time_from_reference(accrual_start)
        t2 = self.forward_curve.get_time_from_reference(accrual_end)

        df1 = self.forward_curve.get_discount_factor(t1)
        df2 = self.forward_curve.get_discount_factor(t2)
        t = self.date_convention.get_year_fraction(accrual_start, accrual_end)

        if self.forward_curve.compounding == "continuous":
            return np.log(df1 / df2) / t
        elif self.forward_curve.compounding == "annual":
            return (df1 / df2) ** (1 / t) - 1
        else:
            raise ValueError(f"Unknown compounding: {self.forward_curve.compounding}")
