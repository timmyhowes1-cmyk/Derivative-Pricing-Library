import numpy as np

### EQUITY PARAMETERS ###
# bump sizes for MC greeks
greek_bumps = {"delta": 0.01,
          "vega": 0.01,
          "rho": 0.01,
          "theta": 5/252,
          "gamma": 0.01,
          "volga": 0.01,
          "vanna": [0.01, 0.01]
          }

params_equity = {"strike": 10,
                     "expiry": 1,
                     "call": False,
                     "european": True,
                     "b": 12,
                     "up": True,
                     "out": True,
                     "arithmetic_mean": True,
                     "fixed_strike": True,
                     "cash_payoff": 10}

# MC parameters
params_MC = {"iterations": int(1e4), "timestep": 1/252,
             "greek_bump_size" : [greek_bumps[g] for g in greek_bumps],
             "antithetic_variates":True}

# parameters for tree pricing
params_tree = {"greek_bump_size" : [greek_bumps[g] for g in greek_bumps],
                "timestep": 1/100}

# parameters for BSM
params_bsm = {"x0": 8,
            "r": 0.03,
            "vol": 0.2,
            "q": 0.02}

#parameters for option pricing under Heston model
params_heston = {"x0": 8,
                 "r": 0.03,
                 "vol": 0.2,
                 "q": 0.02,
                 "mean_vol": 0.2,
                 "reversion_speed": 2,
                 "sigma": 0.3,
                 "covariance": np.array([[1, -0.7], [-0.7, 1]]),
                 "price_scheme": "EulerForPrices",
                 "var_scheme":"ModifiedMilsteinCIR"}

