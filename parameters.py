import numpy as np
# bump sizes for MC greeks
greek_bumps = {"delta": 0.01,
          "vega": 0.0001,
          "rho": 0.0001,
          "theta": 10/252,
          "gamma": 0.01,
          "volga": 0.01,
          "vanna": [0.05, 0.01]
          }

params_derivative = {"option_type": "put",
                     "strike": 10,
                     "expiry": 2,
                     "b": 12,
                     "up": True,
                     "out": True,
                     "average_type": "arithmetic",
                     "strike_type": "fixed",
                     "cash_payoff": 10
                     }

# MC parameters
params_MC = {"iterations": int(1e4), "timestep": 1/252,
             "greek_bump_size" : [greek_bumps[g] for g in greek_bumps],
             "use_antithetic_variates":True, "quiet": True}

# parameters for BSM
params_bsm = {
    "x0": 11,
    "r": 0.01,
    "vol": np.sqrt(0.04),
    "q": 0
}

#parameters for option pricing under Heston model
params_heston = {"x0": 11,
                 "r": 0.01,
                 "vol": np.sqrt(0.04),
                 "q": 0,
                 "mean_vol": np.sqrt(0.04),
                 "reversion_speed": 2,
                 "sigma": 0,
                 "correlation": -0.7,
                 "price_scheme": "EulerForPrices",
                 "var_scheme":"ModifiedMilsteinCIR"
                 }

