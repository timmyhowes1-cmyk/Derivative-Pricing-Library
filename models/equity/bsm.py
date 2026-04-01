from models.equity.base import Model
from utils.math_utils import *

class BSM(Model):
    def __init__(self, x0, r:float=0.01, vol:float=0.1, q:float=0):
        super().__init__(x0)
        self.r = r
        self.vol = vol
        self.q = q

    def generate_paths(self, iterations:int, timestep:float, expiry:float, dw:np.ndarray=None, antithetic_variates:bool=False):
        t = np.linspace(0, expiry, int(round(expiry / timestep)) + 1)
        if dw is None:
            dw = generate_wiener_increments(n=iterations, dt=timestep, expiry=expiry, antithetic_variates=antithetic_variates)
        w = np.zeros((dw.shape[0], dw.shape[1] + 1))
        w[:, 1:] = np.cumsum(dw, axis=1)

        return self.x0 * np.exp((self.r - self.q - (self.vol ** 2) / 2) * np.tile(t, (iterations, 1)) + self.vol * w)

    def get_tree_factors(self, timestep:float):
        return np.exp(self.vol * np.sqrt(timestep)), np.exp(-self.vol * np.sqrt(timestep))



