from models.base import Model
from utils.math_utils import *

class BSM(Model):
    def __init__(self, x0, r=0.01, vol=0.1, q=0):
        super().__init__(x0)
        self.r = r
        self.vol = vol
        self.q = q


    def generate_paths(self, iterations, timestep, expiry, dw=None):
        t = np.linspace(0, expiry, int(round(expiry / timestep)) + 1)
        if dw is None:
            dw = generate_wiener_increments(iterations, timestep, expiry)
        w = np.zeros((dw.shape[0], dw.shape[1] + 1))
        w[:, 1:] = np.cumsum(dw, axis=1)

        return self.x0 * np.exp((self.r - self.q - (self.vol ** 2) / 2) * np.tile(t, (iterations, 1)) + self.vol * w)




