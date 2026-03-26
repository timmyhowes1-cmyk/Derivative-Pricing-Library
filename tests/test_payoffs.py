import numpy as np
import pytest
from instruments import *

### EUROPEAN VANILLA OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (8, 10, 0),
        (10, 10, 0),
        (12, 10, 2),
    ],
)
def test_european_call_payoff_scalar(spot, strike, expected):
    option = European(strike=strike, expiry=1, option_type="call")
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (8, 10, 2),
        (10, 10, 0),
        (12, 10, 0),
    ],
)
def test_european_put_payoff_scalar(spot, strike, expected):
    option = European(strike=strike, expiry=1, option_type="put")
    assert option.payoff(spot) == pytest.approx(expected)

def test_european_call_payoff_vector():
    option = European(strike=10, expiry=1, option_type="call")
    spots = np.array([[8, 10, 12], [9, 8, 7]])
    expected = np.array([2, 0])

    assert option.payoff(spots) == pytest.approx(expected)

def test_european_put_payoff_vector():
    option = European(strike=10, expiry=1, option_type="put")
    spots = np.array([[8, 10, 12], [9, 8, 7]])
    expected = np.array([0, 3])

    assert option.payoff(spots) == pytest.approx(expected)


### ASIAN OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([8, 7, 9]), 10, 0),
        (np.array([5, 10, 15]), 10, 0),
        (np.array([15, 11, 10]), 10, 2),
    ],
)
def test_asian_call_payoff_1d(spot, strike, expected):
    option = Asian(strike=strike, expiry=1, option_type="call")
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([8, 7, 9]), 10, 2),
        (np.array([5, 10, 15]), 10, 0),
        (np.array([15, 11, 10]), 10, 0),
    ],
)
def test_asian_put_payoff_1d(spot, strike, expected):
    option = Asian(strike=strike, expiry=1, option_type="put")
    assert option.payoff(spot) == pytest.approx(expected)

def test_asian_call_payoff_vector():
    option = Asian(strike=10, expiry=1, option_type="call")
    spots = np.array([[14, 12, 10], [9, 8, 7]])
    expected = np.array([2, 0])

    assert option.payoff(spots) == pytest.approx(expected)

def test_asian_put_payoff_vector():
    option = Asian(strike=10, expiry=1, option_type="put")
    spots = np.array([[8, 10, 12], [9, 8, 7]])
    expected = np.array([0, 2])

    assert option.payoff(spots) == pytest.approx(expected)


### LOOKBACK OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([8, 7, 9]), 10, 0),
        (np.array([5, 10, 15]), 10, 5),
        (np.array([13, 6, 10]), 10, 3),
    ],
)
def test_lookback_call_payoff_1d(spot, strike, expected):
    option = Lookback(strike=strike, expiry=1, option_type="call")
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([8, 7, 9]), 10, 3),
        (np.array([5, 10, 15]), 10, 5),
        (np.array([13, 6, 10]), 10, 4),
    ],
)
def test_lookback_put_payoff_1d(spot, strike, expected):
    option = Lookback(strike=strike, expiry=1, option_type="put")
    assert option.payoff(spot) == pytest.approx(expected)

def test_lookback_call_payoff_vector():
    option = Lookback(strike=10, expiry=1, option_type="call")
    spots = np.array([[9, 12, 13], [9, 8, 7]])
    expected = np.array([3, 0])

    assert option.payoff(spots) == pytest.approx(expected)

def test_lookback_put_payoff_vector():
    option = Lookback(strike=10, expiry=1, option_type="put")
    spots = np.array([[7, 10, 12], [11, 8, 7]])
    expected = np.array([3, 3])

    assert option.payoff(spots) == pytest.approx(expected)


### DIGITAL OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, payoff, expected",
    [
        (11, 10, 65, 65),
        (10, 10, 1, 0),
        (8, 10, 3.5, 0),
    ],
)
def test_digital_call_payoff_scalar(spot, strike, payoff, expected):
    option = Digital(strike=strike, expiry=1, option_type="call", cash_payoff=payoff)
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, payoff, expected",
    [
        (11, 10, 1, 0),
        (10, 10, 1, 0),
        (8, 10, 3.5, 3.5),
    ],
)
def test_digital_put_payoff_scalar(spot, strike, payoff, expected):
    option = Digital(strike=strike, expiry=1, option_type="put", cash_payoff=payoff)
    assert option.payoff(spot) == pytest.approx(expected)

def test_digital_call_payoff_vector():
    option = Digital(strike=10, expiry=1, option_type="call", cash_payoff=25)
    spots = np.array([[8, 10, 12], [9, 8, 7]])
    expected = np.array([25, 0])

    assert option.payoff(spots) == pytest.approx(expected)

def test_digital_put_payoff_vector():
    option = Digital(strike=10, expiry=1, option_type="put", cash_payoff=25)
    spots = np.array([[8, 10, 12], [9, 8, 7]])
    expected = np.array([0, 25])

    assert option.payoff(spots) == pytest.approx(expected)
