import datetime as dt
import calendar

class Schedule:
    def __init__(self, start_date:dt.date, end_date:dt.date, months_per_period:int):
        self.start_date = start_date
        self.end_date = end_date
        self.months_per_period = months_per_period
        self.dates = self._build_dates()

    def _build_dates(self):
        dates = [self.start_date]
        current = self.start_date

        while current < self.end_date:
            next_date = add_months(current, self.months_per_period)
            if next_date >= self.end_date:
                dates.append(self.end_date)
                break
            dates.append(next_date)
            current = next_date

        return dates

    def periods(self):
        return list(zip(self.dates[:-1], self.dates[1:]))


def add_months(date:dt.date, months:int):
    new_month = date.month + months
    new_year = date.year + (new_month - 1) // 12
    new_month = new_month % 12

    last_day_of_month = calendar.monthrange(new_year, new_month)[1]
    new_day = min(date.day, last_day_of_month)

    return dt.date(new_year, new_month, new_day)
