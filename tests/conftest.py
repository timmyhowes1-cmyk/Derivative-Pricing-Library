import pytest
from instruments import European
from models import Heston, BSM

@pytest.fixture
def vanilla_call():
    return EuropeanCall(strike=10, expiry=1, option_tye="call")

@pytest.fixture
def vanilla_put():
    return EuropeanCall(strike=10, expiry=1, option_tye="put")

@pytest.fixture
def std_heston_model():
    return Heston(
        x0=10,
        r=0.01,
        q=0.0,
        vol=np.sqrt(0.04),
        mean_vol=np.sqrt(0.04),
        kappa=2,
        sigma=0.3,
        correlation=-0.7,
    )

@pytest.fixture
def std_bs_model():
    return BSM(
        x0=10,
        r=0.01,
        q=0.0,
        vol=0.2,
    )