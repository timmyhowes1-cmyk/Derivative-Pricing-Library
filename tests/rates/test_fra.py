import numpy as np
import datetime as dt
import pytest
from engines import SwapDiscountingEngine
from instruments import FRA, make_fra
from term_structure import FlatYieldCurve, Actual360

def test_fra_flat_curve_par_rate(std_flat_curve):
    fra = make_fra(notional=1, settlement_date=dt.date(2025, 3, 30), accrual_start_date=dt.date(2025, 4, 1),
                   accrual_end_date=dt.date(2025, 7, 1), fixed_rate=0.05, index=std_flat_curve,
                   fixed_date_convention=Actual360())
    engine = SwapDiscountingEngine(std_flat_curve)
    npv = engine.get_price(fra)["value"]
    assert npv == pytest.approx(0, 1e-10)


def test_fra_flat_curve_off_rate(std_flat_curve):
    fra = make_fra(notional=1, settlement_date=dt.date(2025, 3, 30), accrual_start_date=dt.date(2025, 4, 1),
                   accrual_end_date=dt.date(2025, 7, 1), fixed_rate=0.06, index=std_flat_curve,
                   fixed_date_convention=Actual360())
    t = Actual360().get_year_fraction(dt.date(2025, 4, 1), dt.date(2025, 7, 1))
    df = std_flat_curve.get_discount_factor(dt.date(2025, 3, 30))
    true_npv = df * (0.05 - 0.06) * t / (1 + 0.05 * t)
    engine = SwapDiscountingEngine(std_flat_curve)
    model_npv = engine.get_price(fra)["value"]
    assert model_npv == pytest.approx(true_npv, 1e-10)