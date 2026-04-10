import pytest
from instruments import Vanilla
from term_structure import FlatYieldCurve, Actual360
from engines import *
from models import *

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
    return BinomialTree(timestep=1/252)

@pytest.fixture
def std_flat_curve():
    return FlatYieldCurve(reference_date=dt.date(2025, 1, 1), date_convention=Actual360(), flat_rate=0.05, compounding="annual")

