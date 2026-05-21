from .base import NumericalScheme, standard_drift_vol, itself
import numpy as np

class Euler(NumericalScheme):
    def __init__(self, x0, mu:float=0.01, sigma:float=0.01, f_drift=None, f_vol=None):
        super().__init__(x0)
        self.mu = mu
        self.sigma = sigma
        self.f_drift = f_drift if f_drift is not None else standard_drift_vol
        self.f_vol = f_vol if f_vol is not None else itself

    def get_paths(self, dt:float, dw:np.ndarray):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0
        for i in range(1, x.shape[1]):
            t = i * dt
            x[:, i] = (x[:, i-1]
                       + self.f_drift(self.mu, x[:, i-1], t) * dt
                       + self.f_vol(self.sigma, x[:, i-1], t) * dw[:, i-1])
        return x

class Milstein(NumericalScheme):
    def __init__(self, x0, mu:float, sigma:float, f_drift=None, f_vol=None, dvol_dx=None):
        super().__init__(x0)
        self.mu = mu
        self.sigma = sigma
        self.f_drift = f_drift if f_drift is None else standard_drift_vol
        self.f_vol = f_vol if f_vol is not None else itself
        self.vol_derivative = dvol_dx

    def get_paths(self, dt, dw):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0
        for i in range(1, x.shape[1]):
            t = i * dt
            b = self.f_drift(mu=self.mu, x=x[:, i-1], t=t)
            v = self.f_vol(sigma=self.sigma, x=x[:, i-1], t=t)
            db = (self.vol_derivative(mu=self.mu, x=x[:, i-1], t=t)
                  if self.vol_derivative is not None
                  else self.get_vol_x_derivative(x=x[:, i-1], t=t))
            x[:, i] = x[:, i-1] + b * dt + v * dw[:, i-1] + 0.5 * b * db * (dw[:, i-1] ** 2 - i * dt)
        return x

    def get_vol_x_derivative(self, x, t, h=1e-5):
        v1 = self.f_vol(sigma=self.sigma, x=x+h, t=t)
        v2 = self.f_vol(sigma=self.sigma, x=x-h, t=t)

        return (v1 - v2) / (2 * h)

class EulerForPrices(NumericalScheme):
    def __init__(self, x0, mu:float, sigma:float, **kwargs):
        super().__init__(x0)
        self.drift = mu
        self.vol = sigma

    def get_paths(self, dt:float, dw:np.ndarray):
        n_paths, n_steps = dw.shape
        # vol may be a path-dependent array (e.g. stochastic vol); align to dw's time axis
        vol = self.vol[:, :n_steps] if isinstance(self.vol, np.ndarray) else self.vol
        log_increments = (self.drift - 0.5 * vol**2) * dt + vol * dw
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = self.x0
        paths[:, 1:] = self.x0 * np.exp(np.cumsum(log_increments, axis=1))
        return paths

class ModifiedMilsteinCIR(NumericalScheme):
    def __init__(self, x0, a:float, reversion_speed:float, sigma:float):
        super().__init__(x0)
        self.a = a
        self.k = reversion_speed
        self.sigma = sigma

    def get_paths(self, dt:float, dw:np.ndarray):
        x = np.zeros((dw.shape[0], dw.shape[1] + 1))
        x[:, 0] = self.x0

        for i in range(1, x.shape[1]):
            x[:, i] = (np.maximum(((1 - 0.5 * dt * self.k) * np.sqrt(np.maximum(x[:, i-1], 0)) + self.sigma * dw[:, i-1] / (2 * (1 - 0.5 * dt * self.k))) ** 2 +
                                  dt * (self.a - 0.25 * (self.sigma ** 2)), 0))

        return x