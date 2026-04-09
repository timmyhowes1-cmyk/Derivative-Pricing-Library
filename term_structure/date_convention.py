from abc import ABC, abstractmethod
import datetime as dt
import calendar

class DateConvention(ABC):
    @abstractmethod
    def get_year_fraction(self, start_date:dt.date, end_date:dt.date):
        pass

class Actual365Fixed(DateConvention):
    def get_year_fraction(self, start_date:dt.date, end_date:dt.date):
        return (end_date - start_date).days / 365

class Actual360(DateConvention):
    def get_year_fraction(self, start_date:dt.date, end_date:dt.date):
        return (end_date - start_date).days / 360

class ActualActual(DateConvention):
    def get_year_fraction(self, start_date:dt.date, end_date:dt.date):
        result = 0
        current = start_date

        while current < end_date:
            year_end = dt.date(current.year + 1, 1, 1)
            segment_end = min(end_date, year_end)

            days_in_segment = (segment_end - current).days
            days_in_year = 366 if calendar.isleap(current.year) else 365

            result += days_in_segment / days_in_year
            current = segment_end

        return result

class Thirty360(DateConvention):
    def get_year_fraction(self, start_date:dt.date, end_date:dt.date):
        days = 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + (end_date.day - start_date.day)
        return days / 360

