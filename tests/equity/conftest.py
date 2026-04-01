import pytest
from instruments.equity import Vanilla
from engines.equity.analytical import *
from models.equity import *

@pytest.fixture
def vanilla_call():
    return Vanilla(strike=10, expiry=1, call=True, european=True)

@pytest.fixture
def vanilla_put():
    return Vanilla(strike=10, expiry=1, call=False, european=True)

@pytest.fixture
def std_heston_model():
    return Heston(
        x0=10,
        r=0.01,
        q=0.01,
        vol=0.2,
        mean_vol=0.2,
        reversion_speed=2,
        sigma=0.3,
        correlation=-0.7,
    )

@pytest.fixture
def std_bs_model():
    return BSM(
        x0=10,
        r=0.01,
        q=0.01,
        vol=0.2,
    )

@pytest.fixture
def bs_analytical_engine():
    return BSMAnalyticalEngine()

@pytest.fixture
def heston_analytical_engine():
    return HestonAnalyticalEngine()

@pytest.fixture
def binomial_tree():
    BinomialTree(timestep=1/252)

