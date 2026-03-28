import numpy as np
import pytest
from instruments.vanilla_options import Vanilla
from engines.trees import BinomialTree
from engines.analytical import BSMAnalyticalEngine
from models.bsm import BSM

@pytest.mark.parametrize("spot, is_european, is_call", [
    (7, True, True),
    (10, True, True),
    (13, True, True),
    (7, True, False),
    (10, True, False),
    (13, True, False)
])
@pytest.mark.parametrize("timestep, tol", [
    (1/25, 8e-2),
    (1/100, 2e-2),
    (1/500, 5e-3),
])
def test_european_binomial_converges_to_bs(bs_analytical_engine, spot, is_european, is_call, timestep, tol):
    model = BSM(x0=spot, r=0.01, q=0.01, vol=0.2)
    option = Vanilla(strike=10, expiry=1, european=is_european, call=is_call)
    tree_engine = BinomialTree(timestep=timestep, quiet=True)

    tree_price = tree_engine.get_price(option, model)["value"]
    true_price = bs_analytical_engine.get_price(option, model)["value"]

    assert abs(tree_price - true_price) < tol

