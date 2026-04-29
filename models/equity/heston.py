from models.equity.base import Model
from models.equity.cir import CIR
from numerical_schemes import *
from utils.math_utils import generate_wiener_increments
from typing import Union

class Heston(Model):
    def __init__(self, x0, r:float=0.01, vol:float=0.1, q:float=0, mean_vol:float=0.1, reversion_speed:float=2.0, sigma:float=0.1, covariance:Union[float, np.ndarray]=0.5, price_scheme:str="Euler", var_scheme:str="ModifiedMilsteinCIR", **kwargs):
        super().__init__(x0)
        self.r = r
        self.vol = vol
        self.q = q
        self.mean_vol = mean_vol
        self.reversion_speed = reversion_speed
        self.sigma = sigma
        self.covariance = covariance
        self.price_scheme = price_scheme
        self.var_scheme = var_scheme

    def generate_paths(self, iterations:int, timestep:float, expiry:float, dw:np.ndarray=None, antithetic_variates:bool=False):
        if dw is None:
            dw = np.zeros((2, iterations, int(round(expiry / timestep))))
            dw = generate_wiener_increments(n=iterations, dt=timestep, expiry=expiry, cov=self.covariance, antithetic_variates=antithetic_variates)
        if self.sigma > 0:
            var_model = CIR(x0=self.vol**2, mean=self.mean_vol**2, reversion_speed=self.reversion_speed, sigma=self.sigma)
            var_paths = var_model.generate_paths(iterations=iterations, timestep=timestep, expiry=expiry, dw=dw[:, :, 1], antithetic_variates=antithetic_variates, scheme=self.var_scheme)
        else:
            var_paths = self.vol ** 2
        price_scheme = retrieve_scheme(scheme_name=self.price_scheme, x0=self.x0, mu=self.r - self.q, sigma=np.sqrt(var_paths), f_drift=standard_drift_vol, f_vol=heston_vol)

        return price_scheme.get_paths(dt=timestep, dw=dw[:, :, 0])

def heston_vol(sigma, x, t):
    return np.sqrt(sigma) * x
