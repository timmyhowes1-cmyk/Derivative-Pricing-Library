import numpy as np
import datetime as dt
import pytest
from term_structure import FlatYieldCurve, Actual365Fixed

@pytest.mark.parametrize(
    "end_date, compounding, expected",
    [
        (dt.date(2025, 1, 1), "continuous", 1.0),
        (dt.date(2025, 1, 1), "annual", 1.0),
        (dt.date(2025, 1, 1), "simple", 1.0),
        (dt.date(2026, 1, 1), "continuous", np.exp(-1.0 * 0.05)),
        (dt.date(2026, 1, 1), "annual", 1 / (1.0 + 0.05)),
        (dt.date(2026, 1, 1), "simple", 1 / (1.0 + 0.05)),
        (dt.date(2027, 1, 1), "continuous", np.exp(-2.0 * 0.05)),
        (dt.date(2027, 1, 1), "annual", 1 / ((1.0 + 0.05) ** 2.0)),
        (dt.date(2027, 1, 1), "simple", 1 / (1.0 + 2 * 0.05)),
    ],
)
def test_flat_curve_df(end_date, compounding, expected):
    curve = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual365Fixed(), flat_rate=0.05, compounding=compounding)
    t = curve.date_convention.get_year_fraction(start_date=curve.reference_date, end_date=end_date)
    df = curve.get_discount_factor(t)
    assert df == pytest.approx(expected, 1e-8)

@pytest.mark.parametrize(
    "t, compounding, expected",
    [
        (0.01, "continuous", 0.05),
        (0.01, "annual", 0.05),
        (0.01, "simple", 0.05),
        (0.5, "continuous", 0.05),
        (0.5, "annual", 0.05),
        (0.5, "simple", 0.05),
        (1, "continuous", 0.05),
        (1, "annual", 0.05),
        (1, "simple", 0.05),
    ],
)
def test_flat_curve_zero_rate(t, compounding, expected):
    curve = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual365Fixed(), flat_rate=0.05, compounding=compounding)
    df = curve.get_zero_rate(t)
    assert df == pytest.approx(expected, 1e-8)

@pytest.mark.parametrize(
    "t1, t2, compounding, expected",
    [
        (0.01, 0.02, "continuous", 0.05),
        (0.01, 0.02, "annual", 0.05),
        (0.01, 0.02, "simple", 0.05),
        (0.5, 1, "continuous", 0.05),
        (0.5, 1, "annual", 0.05),
        (0.5, 1, "simple", 0.05),
        (1, 100, "continuous", 0.05),
        (1, 100, "annual", 0.05),
        (1, 100, "simple", 0.05),
    ],
)
def test_flat_curve_fwd_rate(t1, t2, compounding, expected):
    curve = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual365Fixed(), flat_rate=0.05, compounding=compounding)
    df = curve.get_forward_rate(t1, t2)
    assert df == pytest.approx(expected, 1e-8)