import numpy as np
import datetime as dt
import pytest
from engines import SwapDiscountingEngine
from instruments import InterestRateSwap, make_vanilla_swap, make_fra, par_swap_rate
from term_structure import FlatYieldCurve, Actual360, Schedule

def test_swap_flat_curve_par_rate(std_flat_curve):
    float_index = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual360(), flat_rate=0.05, compounding="simple")
    schedule = Schedule(start_date=dt.date(2025, 1, 1), end_date=dt.date(2027, 1, 1), months_per_period=3)
    r = par_swap_rate(schedule, std_flat_curve, float_index)
    swap = make_vanilla_swap(notional=1.0, fixed_schedule=schedule, floating_schedule=schedule, fixed_rate=r, floating_index=float_index,
                      fixed_date_convention=Actual360())
    engine = SwapDiscountingEngine(std_flat_curve)
    npv = engine.get_price(swap)["value"]
    assert npv == pytest.approx(0, abs=1e-10)

def test_swap_flat_curve_sum_of_fra(std_flat_curve):
    #  conceptually messy since FRA settles up front, accept high tolerance
    schedule = Schedule(start_date=dt.date(2025, 1, 1), end_date=dt.date(2027, 1, 1), months_per_period=3)
    float_index = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual360(), flat_rate=0.05,
                                 compounding="simple")
    r = par_swap_rate(schedule=schedule, discount_curve=std_flat_curve, forward_curve=float_index)
    engine = SwapDiscountingEngine(std_flat_curve)
    fra_npv = 0
    for st_date, end_date in schedule.periods():
        fra = make_fra(notional=1, settlement_date=end_date, accrual_start_date=st_date,
                       accrual_end_date=end_date, fixed_rate=r, index=float_index,
                       fixed_date_convention=Actual360())
        fra_npv += engine.get_price(fra)["value"]
    swap = make_vanilla_swap(notional=1.0, fixed_schedule=schedule, floating_schedule=schedule, fixed_rate=r, floating_index=float_index,
                      fixed_date_convention=Actual360())
    npv = engine.get_price(swap)["value"]
    assert abs(npv - fra_npv) == pytest.approx(0, abs=1e-5)