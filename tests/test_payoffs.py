import numpy as np
import pytest
import conftest
from instruments import European

### EUROPEAN VANILLA OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (8.0, 10.0, 0.0),
        (10.0, 10.0, 0.0),
        (12.0, 10.0, 2.0),
    ],
)
def test_european_call_payoff_scalar(spot, strike, expected):
    option = conftest.vanilla_call()
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (8.0, 10.0, 2.0),
        (10.0, 10.0, 0.0),
        (12.0, 10.0, 0.0),
    ],
)
def test_european_put_payoff_scalar(spot, strike, expected):
    option = conftest.vanilla_put()
    assert option.payoff(spot) == pytest.approx(expected)


def test_european_call_payoff_vector():
    option = conftest.vanilla_call()
    spots = np.array([8, 10, 12], [9, 8, 7])
    expected = np.array([2], [0])

    result = option.payoff(spots)

    assert result == pytest.approx(expected)


def test_european_put_payoff_vector():
    option = conftest.vanilla_put()
    spots = np.array([8, 10, 12], [9, 8, 7])
    expected = np.array([0], [2])

    result = option.payoff(spots)

    assert result == pytest.approx(expected)
