from instruments import *
from models import *
from engines import *
from parameters import *
from utils.math_utils import *

# setup models
derivative = Vanilla(**params_derivative)
bsm_model = BSM(**params_bsm)
heston_model = Heston(**params_heston)

# setup pricing engines
bsm_engine = BSMAnalyticalEngine(quiet=True)
heston_engine = HestonAnalyticalEngine(quiet=True)
tree_engine = BinomialTree(**params_tree)
mc_engine = MonteCarloEngine(**params_MC)

# define greeks to calculate and respective bump sizes for MC calculation
greeks = [k for k in greek_bumps.keys()]

# calculate prices
a_bsm_price = bsm_engine.get_price(derivative, bsm_model)
a_heston_price = heston_engine.get_price(derivative, heston_model)
tree_price = tree_engine.get_price(derivative, bsm_model)
mc_bsm_price = mc_engine.get_price(derivative, bsm_model)
mc_heston_price = mc_engine.get_price(derivative, heston_model)

# calculate greeks
a_bsm_greeks = bsm_engine.get_greeks(derivative, bsm_model, greek_type=greeks)
a_heston_greeks = heston_engine.get_greeks(derivative, heston_model, greek_type=greeks)
tree_greeks = tree_engine.get_greeks(derivative, bsm_model, greek_type=greeks)
mc_bsm_greeks = mc_engine.get_greeks(derivative, bsm_model, greek_type=greeks)
mc_heston_greeks = mc_engine.get_greeks(derivative, heston_model, greek_type=greeks)

print(a_bsm_price["value"], tree_price["value"], mc_bsm_price["value"])