import numerical_schemes
from models.base import Model
from numerical_schemes import *
from utils.math_utils import *

class CIR(Model):
    def __init__(self, x0, mean=0.04, reversion_speed=2, sigma=0.3):
        super().__init__(x0)
        self.theta = mean
        self.k = reversion_speed
        self.sigma = sigma
        self.feller_ratio = 2 * self.k * self.theta / (sigma ** 2) if sigma > 0 else 0
        self.drift_func = cir_drift
        self.vol_func = cir_vol
        self.vol_derivative = cir_vol_derivative

    def setup_scheme(self, scheme):
        assert scheme != "Euler" and scheme != "Milstein", "Euler and Milstein are not supported for CIR process."
        scheme_to_run = getattr(numerical_schemes, scheme)

        return scheme_to_run(self.x0, a=self.theta * self.k, reversion_speed=self.k, sigma=self.sigma)

    def generate_paths(self, iterations, timestep, expiry, scheme=ModifiedMilsteinCIR, dw=None):
        if dw is None:
            dw = generate_wiener_increments(iterations, timestep, expiry)

        scheme_to_use = self.setup_scheme(scheme)

        return scheme_to_use.get_paths(timestep, dw)



