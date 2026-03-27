from instruments import *
from models import *
from engines import *
from parameters import *
from utils.math_utils import *

# setup models
derivative = European(**params_derivative)
bsm_model = BSM(**params_bsm)
heston_model = Heston(**params_heston)

# setup pricing engines
bsm_engine = BSMAnalyticalEngine(quiet=True)
heston_engine = HestonAnalyticalEngine(quiet=True)
mc_engine = MonteCarloEngine(**params_MC)

# define greeks to calculate and respective bump sizes for MC calculation
greeks = [k for k in greek_bumps.keys()]

# calculate prices
a_bsm_price = bsm_engine.get_price(derivative, bsm_model)
a_heston_price = heston_engine.get_price(derivative, heston_model)
mc_bsm_price = mc_engine.get_price(derivative, bsm_model)
mc_heston_price = mc_engine.get_price(derivative, heston_model)

# calculate greeks
a_bsm_greeks = bsm_engine.get_greeks(derivative, bsm_model, greek_type=greeks)
a_heston_greeks = heston_engine.get_greeks(derivative, heston_model, greek_type=greeks)
mc_bsm_greeks = mc_engine.get_greeks(derivative, bsm_model, greek_type=greeks)
mc_heston_greeks = mc_engine.get_greeks(derivative, heston_model, greek_type=greeks)

print("*** COMPARING ANALYTICAL VS MC ESTIMATES FOR EUROPEAN STYLE OPTION***\n")
print("***PRICE***\n" +f"HESTON MODEL: {np.round(a_heston_price['value'], 4)} vs {np.round(mc_heston_price['value'], 4)} (std error = {np.round(mc_heston_price['std_error'], 4)})")
print(f"BSM MODEL: {np.round(a_bsm_price['value'], 4)} vs {np.round(mc_bsm_price['value'], 4)} (std error = {np.round(mc_bsm_price['std_error'], 4)})\n")
print("***GREEKS***")
for greek in greeks:
    print("HESTON MODEL " +greek.upper() +
          f": {np.round(a_heston_greeks[greek], 4)} "
          f"vs {np.round(mc_heston_greeks[greek]['value'], 4)} "
          f"(STD ERROR = {np.round(mc_heston_greeks[greek]['std_error'], 4)})")
    print("BSM MODEL " +greek.upper() +
          f": {np.round(a_bsm_greeks[greek], 4)} "
          f"vs {np.round(mc_bsm_greeks[greek]['value'], 4)} "
          f"(STD ERROR = {np.round(mc_bsm_greeks[greek]['std_error'], 4)})\n")