import utils.math_utils
from numerical_schemes.base import NumericalScheme
from utils.math_utils import *

class Euler(NumericalScheme):
    def __init__(self, x0, mu=0.01, sigma=0.01, f_drift=None, f_vol=None):
        super().__init__(x0)
        self.mu = mu
        self.sigma = sigma
        self.f_drift = f_drift if f_drift is not None else standard_drift_vol
        self.f_vol = f_vol if f_vol is not None else itself

    def get_paths(self, dt, dw):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0

        mu = format_for_scheme(self.mu, x.shape)
        sigma = format_for_scheme(self.sigma, x.shape)

        for i in range(1, x.shape[1]):
            x[:, i] = x[:, i-1] + self.f_drift(mu[:, i-1], x[:, i-1], i*dt) * dt + self.f_vol(sigma[:, i-1], x[:, i-1], i*dt) * dw[:, i-1]

        return x

class Milstein(NumericalScheme):
    def __init__(self, x0, mu, sigma, f_drift=None, f_vol=None, dvol_dx=None):
        super().__init__(x0)
        self.mu = mu
        self.sigma = sigma
        self.f_drift = f_drift if f_drift is None else standard_drift_vol
        self.f_vol = f_vol if f_vol is not None else itself
        self.vol_derivative = dvol_dx

    def get_paths(self, dt, dw):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0

        mu = format_for_scheme(self.mu, x.shape)
        sigma = format_for_scheme(self.sigma, x.shape)

        for i in range(1, x.shape[1]):
            b = self.f_drift(mu[:, i-1], x[:, i-1], t=i*dt)
            v = self.f_vol(sigma[:, i-1], x=x[:, i-1], t=i*dt)
            if self.vol_derivative is not None:
                db = self.vol_derivative(mu[:, i-1], x[:, i-1], t=i*dt)
            else:
                db = self.get_vol_x_derivative(x[:, i-1], t=i*dt)
            x[:, i] = x[:, i-1] + b * dt + v * dw[:, i-1] + 0.5 * b * db * (dw[:, i-1] ** 2 - i * dt)

        return x

    def get_vol_x_derivative(self, x, t, h=1e-5):
        v1 = self.f_vol(self.sigma, x+h, t)
        v2 = self.f_vol(self.sigma, x-h, t)

        return (v1 - v2) / (2 * h)

class EulerForPrices(NumericalScheme):
    def __init__(self, x0, mu, sigma, **kwargs):
        super().__init__(x0)
        self.drift = mu
        self.vol = sigma

    def get_paths(self, dt, dw):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0

        drift = utils.math_utils.format_for_scheme(self.drift, x.shape)
        vol = utils.math_utils.format_for_scheme(self.vol, x.shape)

        for i in range(1, x.shape[1]):
            x[:, i] = x[:, i-1] * np.exp((drift[:, i-1] - 0.5 * vol[:, i-1]**2) * dt + vol[:, i-1] * dw[:, i-1])

        return x

class ModifiedMilsteinCIR(NumericalScheme):
    def __init__(self, x0, a, reversion_speed, sigma):
        super().__init__(x0)
        self.a = a
        self.k = reversion_speed
        self.sigma = sigma

    def get_paths(self, dt, dw):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0

        for i in range(1, x.shape[1]):
            x[:, i] = (np.maximum(((1 - 0.5 * dt * self.k) * np.sqrt(np.maximum(x[:, i-1], 0)) + self.sigma * dw[:, i-1] / (2 * (1 - 0.5 * dt * self.k))) ** 2 +
                                  dt * (self.a - 0.25 * (self.sigma ** 2)), 0))

        return x