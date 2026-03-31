import numpy as np
import pytest
import math
from instruments.equity import Vanilla
from engines.equity.trees import BinomialTree
from engines.equity.analytical import BSMAnalyticalEngine
from models.equity.bsm import BSM

cases = [
    (7, True),
    (10, True),
    (13, True),
    (7, False),
    (10, False),
    (13, False)
]

@pytest.mark.parametrize("spot, is_call", cases)
@pytest.mark.parametrize("timestep, tol", [
    (1/25, 8e-2),
    (1/100, 2e-2),
    (1/500, 5e-3),
])
def test_european_binomial_converges_to_bs(bs_analytical_engine, spot, is_call, timestep, tol):
    model = BSM(x0=spot, r=0.01, q=0.01, vol=0.2)
    european_option = Vanilla(strike=10, expiry=1, european=True, call=is_call)
    tree_engine = BinomialTree(timestep=timestep, quiet=True)

    tree_price = tree_engine.get_price(european_option, model)["value"]
    true_price = bs_analytical_engine.get_price(european_option, model)["value"]

    assert abs(tree_price - true_price) < tol

@pytest.mark.parametrize("spot, is_call", cases)
@pytest.mark.parametrize("timestep", [
    (1/25),
    (1/100),
    (1/500),
])
def test_american_at_least_european(spot, is_call, timestep):
    model = BSM(x0=spot, r=0.01, q=0.01, vol=0.2)
    european_option = Vanilla(strike=10, expiry=1, european=True, call=is_call)
    american_option = Vanilla(strike=10, expiry=1, european=False, call=is_call)
    tree_engine = BinomialTree(timestep=timestep, quiet=True)

    european_price = tree_engine.get_price(european_option, model)["value"]
    american_price = tree_engine.get_price(american_option, model)["value"]

    assert american_price >= european_price

@pytest.mark.parametrize("spot, is_call", cases)
@pytest.mark.parametrize("timestep", [
    (1/25),
    (1/100),
    (1/500),
])
def test_option_respects_intrinsic_lower_bound(spot, is_call, timestep):
    model = BSM(x0=spot, r=0.01, q=0.01, vol=0.2)
    american_option = Vanilla(strike=10, expiry=1, european=False, call=is_call)
    tree_engine = BinomialTree(timestep=timestep, quiet=True)
    val = tree_engine.get_price(american_option, model)["value"]
    intrinsic = np.maximum(spot - american_option.K, 0) if is_call else np.maximum(american_option.K - spot, 0)

    assert val >= 0
    if True:
        assert val >= intrinsic or math.isclose(val, intrinsic, rel_tol=0, abs_tol=1e-12)