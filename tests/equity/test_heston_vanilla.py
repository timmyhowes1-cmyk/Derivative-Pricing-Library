import numpy as np
import pytest
from instruments.equity import *
from models.equity.heston import Heston
from engines.equity.analytical import HestonAnalyticalEngine

@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 0.0034636862),
        (10.0, 0.7539968970),
        (13.0, 3.1294990630),
    ],
)
def test_heston_call_prices(vanilla_call, heston_analytical_engine, spot, expected):
    model = Heston(x0=spot, vol=0.2, r=0.01, q=0.01, mean_vol=0.2, reversion_speed=2, sigma=0.3, covariance=np.array([[1, -0.7], [-0.7, 1]]))
    price = heston_analytical_engine.get_price(vanilla_call, model)["value"]
    assert price == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "spot, expected",
    [
        (7.0, 2.9736131874),
        (10.0, 0.7539968970),
        (13.0, 0.1593495618),
    ],
)
def test_heston_put_prices(vanilla_put, heston_analytical_engine, spot, expected):
    model = Heston(x0=spot, vol=0.2, r=0.01, q=0.01, mean_vol=0.2, reversion_speed=2, sigma=0.3, covariance=np.array([[1, -0.7], [-0.7, 1]]))
    price = heston_analytical_engine.get_price(vanilla_put, model)["value"]
    assert price == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "greek, spot, expected",
    [
        ("delta", 7.0, 0.0117678953),
        ("delta", 10.0, 0.5903153417),
        ("delta", 13.0, 0.9115369723),
        ("vega", 7.0, 0.0741983570),
        ("vega", 10.0, 1.6758945423),
        ("vega", 13.0, 0.8962149682),
        ("rho", 7.0, 0.0789115807),
        ("rho", 10.0, 5.1491565204),
        ("rho", 13.0, 8.7204815769),
        ("theta", 7.0, -0.0156874336),
        ("theta", 10.0, -0.3609730613),
        ("theta", 13.0, -0.1879649316),
        ("gamma", 7.0, 0.0352126738),
        ("gamma", 10.0, 0.2055909450),
        ("gamma", 13.0, 0.0420334135),
        ("volga", 7.0, 1.4638469885),
        ("volga", 10.0, 4.9596869909),
        ("volga", 13.0, 4.5148142034),
        ("vanna", 7.0, 0.2114351769),
        ("vanna", 10.0, 0.0199822422),
        ("vanna", 13.0, -0.2813458494),
        (["delta", "volga"], 10, [0.5903153417, 4.9596869909])
    ],
)
def test_heston_call_greeks(vanilla_call, heston_analytical_engine, greek, spot, expected):
    model = Heston(x0=spot, vol=0.2, r=0.01, q=0.01, mean_vol=0.2, reversion_speed=2, sigma=0.3, covariance=np.array([[1, -0.7], [-0.7, 1]]))
    greeks = heston_analytical_engine.get_greeks(vanilla_call, model, greek_type=greek)
    result = [greeks[k] for k in greeks.keys()] if isinstance(greek, list) else greeks[greek]
    assert result == pytest.approx(expected, abs=1e-7)
#
@pytest.mark.parametrize(
    "greek, spot, expected",
    [
        ("delta", 7.0, -0.9782819385),
        ("delta", 10.0, -0.3997344920),
        ("delta", 13.0, -0.0785128614),
        ("vega", 7.0, 0.0741983570),
        ("vega", 10.0, 1.6758945423),
        ("vega", 13.0, 0.8962149682),
        ("rho", 7.0, -9.8215867568),
        ("rho", 10.0, -4.7513418171),
        ("rho", 13.0, -1.1800167605),
        ("theta", 7.0, 0.0140140614),
        ("theta", 10.0, -0.3609730613),
        ("theta", 13.0, -0.2176664266),
        ("gamma", 7.0, 0.0352126738),
        ("gamma", 10.0, 0.2055909450),
        ("gamma", 13.0, 0.0420334135),
        ("volga", 7.0, 1.4638469885),
        ("volga", 10.0, 4.9596869909),
        ("volga", 13.0, 4.5148142034),
        ("vanna", 7.0, 0.2114351769),
        ("vanna", 10.0, 0.0199822422),
        ("vanna", 13.0, -0.2813458494),
        (["delta", "volga"], 10, [-0.3997344920, 4.9596869909])
    ],
)
def test_heston_put_greeks(vanilla_put, heston_analytical_engine, greek, spot, expected):
    model = Heston(x0=spot, vol=0.2, r=0.01, q=0.01, mean_vol=0.2, reversion_speed=2, sigma=0.3, covariance=np.array([[1, -0.7], [-0.7, 1]]))
    greeks = heston_analytical_engine.get_greeks(vanilla_put, model, greek_type=greek)
    result = [greeks[k] for k in greeks.keys()] if isinstance(greek, list) else greeks[greek]
    assert result == pytest.approx(expected, abs=1e-7)