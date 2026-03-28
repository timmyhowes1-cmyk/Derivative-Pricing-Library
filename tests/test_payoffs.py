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
        (np.array([[8, 10, 12]]), 10, 2),
        (np.array([[8, 10, 12], [9, 8, 7]]), 10, np.array([2, 0])),
    ],
)
def test_european_call_payoff(spot, strike, expected):
    option = Vanilla(strike=strike, expiry=1, call=True, european=True)
    assert option.payoff(spot) == pytest.approx(expected)

@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (8, 10, 2),
        (10, 10, 0),
        (12, 10, 0),
        (np.array([[9, 8, 7]]), 10, 3),
        (np.array([[8, 10, 12], [9, 8, 7]]), 10, np.array([0, 3])),
    ],
)
def test_european_put_payoff(spot, strike, expected):
    option = Vanilla(strike=strike, expiry=1, call=False, european=True)
    assert option.payoff(spot) == pytest.approx(expected)


### ASIAN OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([[8, 7, 9]]), 10, 0),
        (np.array([[5, 10, 15]]), 10, 0),
        (np.array([[15, 11, 10]]), 10, 2),
        (np.array([[14, 12, 10], [9, 8, 7]]), 10, np.array([2, 0]))
    ],
)
def test_asian_call_payoff(spot, strike, expected):
    option = Asian(strike=strike, expiry=1, call=True, european=True, arithmetic_mean=True, fixed_strike=True)
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([[8, 7, 9]]), 10, 2),
        (np.array([[5, 10, 15]]), 10, 0),
        (np.array([[15, 11, 10]]), 10, 0),
        (np.array([[8, 10, 12], [9, 8, 7]]), 10, np.array([0, 2]))
    ],
)
def test_asian_put_payoff(spot, strike, expected):
    option = Asian(strike=strike, expiry=1, call=False, european=True, arithmetic_mean=True, fixed_strike=True)
    assert option.payoff(spot) == pytest.approx(expected)


### LOOKBACK OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([[8, 7, 9]]), 10, 0),
        (np.array([[5, 10, 15]]), 10, 5),
        (np.array([[13, 6, 10]]), 10, 3),
        (np.array([[9, 12, 13], [9, 8, 7]]), 10, np.array([3, 0]))
    ],
)
def test_lookback_call_payoff(spot, strike, expected):
    option = Lookback(strike=strike, expiry=1, call=True, european=True, fixed_strike=True)
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, expected",
    [
        (np.array([[8, 7, 9]]), 10, 3),
        (np.array([[5, 10, 15]]), 10, 5),
        (np.array([[13, 6, 10]]), 10, 4),
        (np.array([[10, 10, 12], [11, 8, 7]]), 10, np.array([0, 3]))
    ],
)
def test_lookback_put_payoff(spot, strike, expected):
    option = Lookback(strike=strike, expiry=1, call=False, european=True, fixed_strike=True)
    assert option.payoff(spot) == pytest.approx(expected)


### DIGITAL OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, payoff, expected",
    [
        (11, 10, 65, 65),
        (10, 10, 1, 0),
        (8, 10, 3.5, 0),
        (np.array([[8, 10, 12]]), 10, 25, 25),
        (np.array([[8, 10, 12], [9, 8, 7]]), 10, 25, np.array([25, 0]))
    ],
)
def test_digital_call_payoff(spot, strike, payoff, expected):
    option = Digital(strike=strike, expiry=1, call=True, european=True, cash_payoff=payoff)
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, payoff, expected",
    [
        (11, 10, 1, 0),
        (10, 10, 1, 0),
        (8, 10, 3.5, 3.5),
        (np.array([[9, 8, 7]]), 10, 25, 25),
        (np.array([[8, 10, 12], [9, 8, 7]]), 10, 25, np.array([0, 25]))
    ],
)
def test_digital_put_payoff_scalar(spot, strike, payoff, expected):
    option = Digital(strike=strike, expiry=1, call=False, european=True, cash_payoff=payoff)
    assert option.payoff(spot) == pytest.approx(expected)


### BARRIER OPTION PAYOFFS
@pytest.mark.parametrize(
    "spot, strike, barrier, up, out, expected",
    [
        (11, 10, 7, False, True, 1),
        (np.array([[8, 7, 12]]), 10, 7, False, True, 2), # down and out
        (np.array([[8, 6.5, 12]]), 10, 7, False, True, 0), # down and out
        (np.array([[8, 6.5, 12]]), 10, 7, False, False, 2), # down and in
        (np.array([[8, 9, 12]]), 10, 7, False, False, 0), # down and in
        (np.array([[8, 9, 12]]), 10, 15, True, True, 2), # up and out
        (np.array([[8, 17, 12]]), 10, 15, True, True, 0), # up and out
        (np.array([[8, 13.5, 12]]), 10, 13, True, False, 2), # up and in
        (np.array([[8, 10, 12]]), 10, 13, True, False, 0), # up and in
        (np.array([[8, 10, 12], [8, 13.5, 12]]), 10, 13, True, False, np.array([0, 2])), # up and in
    ],
)
def test_barrier_call_payoff_scalar(spot, strike, barrier, up, out, expected):
    option = Barrier(strike=strike, expiry=1, call=True, european=True, b=barrier, up=up, out=out)
    assert option.payoff(spot) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot, strike, barrier, up, out, expected",
    [
        (8, 10, 7, False, True, 2),
        (np.array([[12, 7, 8]]), 10, 7, False, True, 2), # down and out
        (np.array([[12, 6.5, 8]]), 10, 7, False, True, 0), # down and out
        (np.array([[12, 6.5, 8]]), 10, 7, False, False, 2), # down and in
        (np.array([[12, 9, 8]]), 10, 7, False, False, 0), # down and in
        (np.array([[12, 9, 8]]), 10, 15, True, True, 2), # up and out
        (np.array([[12, 17, 8]]), 10, 15, True, True, 0), # up and out
        (np.array([[12, 13.5, 8]]), 10, 13, True, False, 2), # up and in
        (np.array([[12, 10, 8]]), 10, 13, True, False, 0), # up and in
        (np.array([[12, 10, 8], [12, 13.5, 8]]), 10, 13, True, False, np.array([0, 2])), # up and in
    ],
)
def test_barrier_put_payoff_scalar(spot, strike, barrier, up, out, expected):
    option = Barrier(strike=strike, expiry=1, call=False, european=True, b=barrier, up=up, out=out)
    assert option.payoff(spot) == pytest.approx(expected)
