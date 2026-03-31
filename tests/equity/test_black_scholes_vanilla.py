import numpy as np
import pytest
from instruments.equity import *
from models.equity.bsm import BSM
from engines.equity.analytical import BSMAnalyticalEngine

@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 0.0245641162),
        (10.0, 0.7886308735),
        (13.0, 3.0700328188),
    ],
)
def test_bs_call_prices(vanilla_call, bs_analytical_engine, spot, expected):
    model = BSM(x0=spot, vol=0.2, r=0.01, q=0.01)
    price = bs_analytical_engine.get_price(vanilla_call, model)["value"]
    assert price == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 2.9947136175),
        (10.0, 0.7886308735),
        (13.0, 0.0998833176),
    ],
)
def test_bs_put_prices(vanilla_put, bs_analytical_engine, spot, expected):
    model = BSM(x0=spot, vol=0.2, r=0.01, q=0.01)
    price = bs_analytical_engine.get_price(vanilla_put, model)["value"]
    assert price == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "greek, spot, expected",
    [
        ("delta", 7.0, 0.0456920740),
        ("delta", 10.0, 0.5344564605),
        ("delta", 13.0, 0.9118346204),
        ("vega", 7.0, 0.6703846629),
        ("vega", 10.0, 3.9300280364),
        ("vega", 13.0, 1.8953264658),
        ("rho", 7.0, 0.29528040199),
        ("rho", 10.0, 4.5559337320),
        ("rho", 13.0, 8.7838172469),
        ("theta", 7.0, -0.0667928251),
        ("theta", 10.0, -0.3851164949),
        ("theta", 13.0, -0.1588323184),
        ("gamma", 7.0, 0.0684065983),
        ("gamma", 10.0, 0.1965014018),
        ("gamma", 13.0, 0.0560747475),
        ("volga", 7.0, 10.6270227846),
        ("volga", 10.0, -0.1965014018),
        ("volga", 13.0, 16.2133350580),
        ("vanna", 7.0, 0.9018468047),
        ("vanna", 10.0, 0.1965014018),
        ("vanna", 13.0, -0.8833834708),
        (["delta", "volga"], 10, [0.5344564605, -0.1965014018])
    ],
)
def test_bs_call_greeks(vanilla_call, bs_analytical_engine, greek, spot, expected):
    model = BSM(x0=spot, vol=0.2, r=0.01, q=0.01)
    greeks = bs_analytical_engine.get_greeks(vanilla_call, model, greek_type=greek)
    result = [greeks[k] for k in greeks.keys()] if isinstance(greek, list) else greeks[greek]
    assert result == pytest.approx(expected, abs=1e-9)

@pytest.mark.parametrize(
    "greek, spot, expected",
    [
        ("delta", 7.0, -0.9443577597),
        ("delta", 10.0, -0.4555933732),
        ("delta", 13.0, -0.0782152133),
        ("vega", 7.0, 0.6703846629),
        ("vega", 10.0, 3.9300280364),
        ("vega", 13.0, 1.8953264658),
        ("rho", 7.0, -9.6052179356),
        ("rho", 10.0, -5.3445646055),
        ("rho", 13.0, -1.1166810906),
        ("theta", 7.0, -0.0370913301),
        ("theta", 10.0, -0.3851164949),
        ("theta", 13.0, -0.1885338134),
        ("gamma", 7.0, 0.0684065983),
        ("gamma", 10.0, 0.1965014018),
        ("gamma", 13.0, 0.0560747475),
        ("volga", 7.0, 10.6270227846),
        ("volga", 10.0, -0.1965014018),
        ("volga", 13.0, 16.2133350580),
        ("vanna", 7.0, 0.9018468047),
        ("vanna", 10.0, 0.1965014018),
        ("vanna", 13.0, -0.8833834708),
        (["delta", "volga"], 10, [-0.4555933732, -0.1965014018])
    ],
)
def test_bs_put_greeks(vanilla_put, bs_analytical_engine, greek, spot, expected):
    model = BSM(x0=spot, vol=0.2, r=0.01, q=0.01)
    greeks = bs_analytical_engine.get_greeks(vanilla_put, model, greek_type=greek)
    result = [greeks[k] for k in greeks.keys()] if isinstance(greek, list) else greeks[greek]
    assert result == pytest.approx(expected, abs=1e-9)
