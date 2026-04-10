from instruments import *
from models import *
from engines import *
from parameters import *
from numerical_schemes import *
from term_structure import *
from utils import *

# setup equity models
instrument_list = [Vanilla, Asian, Lookback, Digital, Barrier]
equity_instruments = [instr(**params_equity) for instr in instrument_list]
bsm_model = BSM(**params_bsm)
heston_model = Heston(**params_heston)

# setup pricing engines
# bsm_engine = BSMAnalyticalEngine()
# heston_engine = HestonAnalyticalEngine()
tree_engine = BinomialTree(**params_tree)
mc_engine = MonteCarloEngine(**params_MC)

# define greeks to calculate and respective bump sizes for MC calculation
greeks = [k for k in greek_bumps.keys()]

# calculate prices
# a_bsm_prices = [bsm_engine.get_price(Vanilla, bsm_model), bsm_engine.get_price(Digital, bsm_model)]
# a_heston_prices = [heston_engine.get_price(Vanilla, bsm_model), heston_engine.get_price(Digital, bsm_model)]
# tree_prices = [tree_engine.get_price(Vanilla, bsm_model), tree_engine.get_price(Digital, bsm_model)]
mc_bsm_price = [mc_engine.get_price(instr, bsm_model) for instr in equity_instruments]
mc_heston_price = [mc_engine.get_price(instr, heston_model) for instr in equity_instruments]

# calculate greeks
# a_bsm_greeks = [bsm_engine.get_greeks(Vanilla, bsm_model), bsm_engine.get_greeks(Digital, bsm_model)]
# a_heston_greeks = [heston_engine.get_greeks(Vanilla, bsm_model), heston_engine.get_greeks(Digital, bsm_model)]
# tree_greeks = [tree_engine.get_greeks(Vanilla, bsm_model), tree_engine.get_greeks(Digital, bsm_model)]
mc_bsm_greeks = [mc_engine.get_greeks(instr, bsm_model, greek_type=greeks) for instr in equity_instruments]
mc_heston_greeks = [mc_engine.get_greeks(instr, heston_model, greek_type=greeks) for instr in equity_instruments]

print("***EQUITY DERIVATIVE PRICES***\n")
print("***MC ESTIMATES - BSM MODEL***")
for i in range(len(equity_instruments)):
    instr = equity_instruments[i]
    print(f"{instr.__class__.__name__} - {np.round(mc_bsm_price[i]['value'], 4)} (STD ERROR = {np.round(mc_bsm_price[i]['std_error'], 4)})")

print("\n***MC ESTIMATES - HESTON MODEL***")
for i in range(len(equity_instruments)):
    instr = equity_instruments[i]
    print(f"{instr.__class__.__name__} - {np.round(mc_heston_price[i]['value'], 4)} (STD ERROR = {np.round(mc_heston_price[i]['std_error'], 4)})")