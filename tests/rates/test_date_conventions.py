import numpy as np
import datetime as dt
import pytest
from term_structure.date_convention import *

@pytest.mark.parametrize(
    "start, end, expected",
    [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 1), 0),
        (dt.date(2025, 1, 1), dt.date(2025, 1, 2), 1 / 365),
        (dt.date(2025, 1, 1), dt.date(2026, 1, 1), 1.0),
        (dt.date(2024, 1, 1), dt.date(2025, 1, 1), 366 / 365),
        (dt.date(2025, 2, 28), dt.date(2025, 3, 1), 1 / 365),
        (dt.date(2024, 2, 28), dt.date(2024, 3, 1), 2 / 365),
        (dt.date(2025, 1, 1), dt.date(2025, 7, 1), 181 /365),
        (dt.date(2025, 1, 1), dt.date(2025, 4, 1), 90 / 365),
    ],
)
def test_actual365(start, end, expected):
    conv = Actual365Fixed()
    assert conv.get_year_fraction(start, end) == pytest.approx(expected, 1e-6)

@pytest.mark.parametrize(
    "start, end, expected",
    [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 1), 0),
        (dt.date(2025, 1, 1), dt.date(2025, 1, 2), 1 / 360),
        (dt.date(2025, 1, 1), dt.date(2025, 12, 27), 1.0),
        (dt.date(2025, 1, 1), dt.date(2026, 1, 1), 365 / 360),
        (dt.date(2024, 1, 1), dt.date(2025, 1, 1), 366 / 360),
        (dt.date(2025, 2, 28), dt.date(2025, 3, 1), 1 / 360),
        (dt.date(2024, 2, 28), dt.date(2024, 3, 1), 2 / 360),
        (dt.date(2025, 1, 1), dt.date(2025, 7, 1), 181 /360),
        (dt.date(2025, 1, 1), dt.date(2025, 4, 1), 90 / 360),
    ],
)
def test_actual360(start, end, expected):
    conv = Actual360()
    assert conv.get_year_fraction(start, end) == pytest.approx(expected, 1e-6)

@pytest.mark.parametrize(
    "start, end, expected",
    [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 1), 0),
        (dt.date(2025, 1, 1), dt.date(2025, 1, 2), 1 / 365),
        (dt.date(2025, 1, 1), dt.date(2026, 1, 1), 1.0),
        (dt.date(2024, 1, 1), dt.date(2025, 1, 1), 1.0),
        (dt.date(2023, 7, 1), dt.date(2024, 7, 1), 184 / 365 + 182 / 366),
        (dt.date(2024, 2, 28), dt.date(2024, 3, 1), 2 / 366),
        (dt.date(2025, 1, 1), dt.date(2025, 7, 1), 181 /365),
        (dt.date(2025, 1, 1), dt.date(2025, 4, 1), 90 / 365),
    ],
)
def test_actual_actual(start, end, expected):
    conv = ActualActual()
    assert conv.get_year_fraction(start, end) == pytest.approx(expected, 1e-6)

@pytest.mark.parametrize(
    "start, end, expected",
    [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 1), 0),
        (dt.date(2025, 1, 1), dt.date(2025, 1, 2), 1 / 360),
        (dt.date(2025, 1, 1), dt.date(2026, 1, 1), 1.0),
        (dt.date(2024, 1, 1), dt.date(2025, 1, 1), 1.0),
        (dt.date(2024, 2, 28), dt.date(2024, 3, 1), 3 / 360),
        (dt.date(2025, 1, 1), dt.date(2025, 2, 1), 30 /360),
        (dt.date(2025, 1, 1), dt.date(2025, 7, 1), 180 /360),
        (dt.date(2025, 1, 1), dt.date(2025, 4, 1), 90 / 360),
    ],
)
def test_thirty360(start, end, expected):
    conv = Thirty360()
    assert conv.get_year_fraction(start, end) == pytest.approx(expected, 1e-6)