import numpy as np
# bump sizes for MC greeks
greek_bumps = {"delta": 0.01,
          "vega": 0.01,
          "rho": 0.01,
          "theta": 5/252,
          "gamma": 0.01,
          "volga": 0.01,
          "vanna": [0.01, 0.01]
          }

params_derivative = {"strike": 10,
                     "expiry": 1,
                     "call": False,
                     "european": True,
                     "b": 12,
                     "up": True,
                     "out": True,
                     "arithmetic_mean": True,
                     "fixed_strike": True,
                     "cash_payoff": 10
                     }

# MC parameters
params_MC = {"iterations": int(1e5), "timestep": 1/252,
             "greek_bump_size" : [greek_bumps[g] for g in greek_bumps],
             "antithetic_variates":True, "quiet": True}

# parameters for BSM
params_bsm = {
    "x0": 11,
    "r": 0.03,
    "vol": 0.2,
    "q": 0.02
}

#parameters for option pricing under Heston model
params_heston = {"x0": 11,
                 "r": 0.03,
                 "vol": 0.2,
                 "q": 0.02,
                 "mean_vol": 0.2,
                 "reversion_speed": 2,
                 "sigma": 0.3,
                 "correlation": -0.7,
                 "price_scheme": "EulerForPrices",
                 "var_scheme":"ModifiedMilsteinCIR"
                 }

