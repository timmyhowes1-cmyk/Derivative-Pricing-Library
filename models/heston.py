from models.base import Model
from models.cir import CIR
from numerical_schemes import *
from utils.math_utils import *

class Heston(Model):
    def __init__(self, x0, r=0.01, vol=0.1, q=0, mean_vol=0.1, reversion_speed=0.25, sigma=0.1, correlation=0.5, price_scheme="Euler", var_scheme="ModifiedMilsteinCIR", **kwargs):
        super().__init__(x0)
        self.r = r
        self.vol = vol
        self.q = q
        self.mean_vol = mean_vol
        self.reversion_speed = reversion_speed
        self.sigma = sigma
        self.correlation = correlation
        self.price_scheme = price_scheme
        self.var_scheme = var_scheme

    def generate_paths(self, iterations, timestep, expiry, dw=None, antithetic_variates=False):
        if dw is None:
            dw = np.zeros((2, iterations, int(round(expiry / timestep))))
            dw_array = generate_wiener_increments(iterations, timestep, expiry, correlation=self.correlation, antithetic_variates=antithetic_variates)
            dw[0], dw[1] = dw_array[0], dw_array[1]

        if self.sigma > 0:
            var_model = CIR(self.vol**2, mean=self.mean_vol**2, reversion_speed=self.reversion_speed, sigma=self.sigma)
            var_paths = var_model.generate_paths(iterations, timestep, expiry, dw=dw[1], scheme=self.var_scheme)
        else:
            var_paths = self.vol ** 2
        price_scheme = create_scheme(self.price_scheme, self.x0, mu=self.r - self.q, sigma=np.sqrt(var_paths), f_drift=standard_drift_vol, f_vol=heston_vol)

        return price_scheme.get_paths(timestep, dw[0])



