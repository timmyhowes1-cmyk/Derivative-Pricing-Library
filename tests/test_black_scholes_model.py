import numpy as np
import pytest
from instruments import *
from models.bsm import BSM
from engines.analytical import BSMAnalyticalEngine

@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 0.0245641162),
        (10.0, 0.7886308735),
        (13.0, 3.0700328188),
    ],
)

def test_call_prices(vanilla_call, bs_analytical_engine, spot, expected):
    model = BlackScholes(x0=spot, vol=0.2, r=0.01, q=0.01)
    price = bs_analytical_engine.get_price(vanilla_call, model)["value"]
    assert price == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 0.0245641162),
        (10.0, 0.7886308735),
        (13.0, 3.0700328188),
    ],
)

def test_call_prices(vanilla_call, bs_analytical_engine, spot, expected):
    model = BlackScholes(x0=spot, vol=0.2, r=0.01, q=0.01)
    price = bs_analytical_engine.get_price(vanilla_call, model)["value"]
    assert price == pytest.approx(expected, abs=1e-10)