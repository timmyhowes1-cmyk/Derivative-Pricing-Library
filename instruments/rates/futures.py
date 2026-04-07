import datetime as dt
from term_structure import InterestRateIndex


class InterestRateFuture:
    def __init__(self, notional:float, contract_date:dt.date, reference_start_date:dt.date, reference_end_date:dt.date, index:InterestRateIndex,
                 futures_price:float, convexity_adjustment:float=0.0):
        self.notional = notional
        self.contract_date = contract_date
        self.reference_start_date = reference_start_date
        self.reference_end_date = reference_end_date
        self.index = index
        self.futures_price = futures_price
        self.convexity_adjustment = convexity_adjustment

    def implied_forward_rate(self):
        implied_rate = 100 - self.futures_price
        return implied_rate / 100 + self.convexity_adjustment

