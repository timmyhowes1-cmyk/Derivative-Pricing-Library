import numpy as np
import datetime as dt
import pytest
from instruments import SimpleDeposit, InterestRateSwap, InterestRateFutures,make_vanilla_swap, par_swap_rate
from term_structure import FlatYieldCurve, Actual360, Schedule, PiecewiseLinearDiscountCurve
from term_structure.bootstrapping import bootstrap_curve, DepositHelper, SwapHelper, FuturesHelper

def test_bootstrap_simple(std_flat_curve):
    deposit = SimpleDeposit(notional=1, start_date=dt.date(2025, 1, 1), maturity_date=dt.date(2025, 7, 1), rate=0.05, date_convention=Actual360())

    swap_schedule = Schedule(start_date=dt.date(2025, 1, 1), end_date=dt.date(2026, 1, 1))
    float_index = FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual360(), flat_rate=0.05, compounding="simple")
    par_rate = 0.049393
    par_swap = make_vanilla_swap(notional=1, fixed_schedule=swap_schedule, floating_schedule=swap_schedule, fixed_rate=par_rate,
                             floating_index=float_index, fixed_date_convention=Actual360())

    futures = InterestRateFutures(notional=1, contract_date=dt.date(2025, 7, 1), reference_start_date=dt.date(2026, 1, 1), reference_end_date=dt.date(2026, 7, 1), index=None,
                 futures_price=93)

    helpers = [DepositHelper(market_rate=0.05, instrument=deposit), SwapHelper(instrument=par_swap),
               FuturesHelper(instrument=futures)]
    curve = bootstrap_curve(helpers=helpers, reference_date=dt.date(2025, 1, 1), date_convention=Actual360(),
                            curve_cls=PiecewiseLinearDiscountCurve, compounding="annual")
    rates = [curve.get_zero_rate(t) for t in curve.times]
    assert rates == pytest.approx([0.05062147059093269, 0.062051926573951866, 0.06508178552563759], abs=1e-8)