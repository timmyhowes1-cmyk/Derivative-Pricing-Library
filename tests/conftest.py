import pytest
from instruments import European
from engines.analytical import BSMAnalyticalEngine, HestonAnalyticalEngine
from models import BSM, Heston

@pytest.fixture
def vanilla_call():
    return European(strike=10, expiry=1, option_type="call")

@pytest.fixture
def vanilla_put():
    return European(strike=10, expiry=1, option_type="put")

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
    return BSMAnalyticalEngine(quiet=True)

@pytest.fixture
def heston_analytical_engine():
    return HestonAnalyticalEngine(quiet=True)