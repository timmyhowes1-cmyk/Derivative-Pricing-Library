import numpy as np
from scipy.integrate import quad
from typing import Union
from scipy.stats import norm
from engines.equity.base import Engine
from instruments.equity import EquityOption
from models.equity import Model

class BSMAnalyticalEngine(Engine):
    def get_price(self, instrument:EquityOption, model:Model):
        d1, d2 = self._calculate_d1_d2(instrument=instrument, model=model)

        if instrument.call:
            return {"value": (np.exp(-model.q * instrument.T) * model.x0 * norm.cdf(d1)
                    - np.exp(-model.r * instrument.T) * instrument.K * norm.cdf(d2))}
        return {"value": (instrument.K * np.exp(-model.r * instrument.T) * norm.cdf(-d2)
                - np.exp(-model.q * instrument.T) * model.x0 * norm.cdf(-d1))}

    def get_greeks(self, instrument:EquityOption, model:Model, greek_type:Union[list, str]):
        d1, d2 = self._calculate_d1_d2(instrument=instrument, model=model)
        greeks = {}

        if isinstance(greek_type, str):
            greek_type = [greek_type]

        for i in range(len(greek_type)):
            func_name = f"calculate_{greek_type[i]}"
            func = getattr(self, func_name)
            greeks[greek_type[i]] = func(d=np.array([d1, d2]), instrument=instrument, model=model)

        return greeks

    @staticmethod
    def _calculate_d1_d2(instrument:EquityOption, model:Model):
        d1 = ((np.log(model.x0 / instrument.K) + instrument.T * (model.r - model.q + (model.vol ** 2) / 2))
              / (model.vol * np.sqrt(instrument.T)))
        d2 = d1 - model.vol * np.sqrt(instrument.T)

        return d1, d2

    @staticmethod
    def calculate_delta(d:np.ndarray, instrument:EquityOption, model:Model):
        return np.exp(-model.q * instrument.T) * norm.cdf(d[0]) if instrument.call else -np.exp(-model.q * instrument.T) * norm.cdf(-d[0])

    @staticmethod
    def calculate_vega(d:np.ndarray, instrument:EquityOption, model:Model):
        return model.x0 * np.exp(-model.q * instrument.T) * norm.pdf(d[0]) * np.sqrt(instrument.T)

    @staticmethod
    def calculate_rho(d:np.ndarray, instrument:EquityOption, model:Model):
        return (instrument.T * instrument.K * np.exp(-model.r * instrument.T) * norm.cdf(d[1]) if instrument.call
                else -instrument.T * instrument.K * np.exp(-model.r * instrument.T) * norm.cdf(-d[1]))

    @staticmethod
    def calculate_theta(d:np.ndarray, instrument:EquityOption, model:Model):
        if instrument.call:
            return (-norm.pdf(d[0]) * (model.x0 * model.vol * np.exp(-model.q * instrument.T)) / (2 * np.sqrt(instrument.T))
                      - model.r * instrument.K * np.exp(-model.r * instrument.T) * norm.cdf(d[1])
                      + model.q * model.x0 * np.exp(-model.q * instrument.T) * norm.cdf(d[0]))
        return (-norm.pdf(d[0]) * (model.x0 * model.vol * np.exp(-model.q * instrument.T)) / (2 * np.sqrt(instrument.T))
                      + model.r * instrument.K * np.exp(-model.r * instrument.T) * norm.cdf(-d[1])
                      - model.q * model.x0 * np.exp(-model.q * instrument.T) * norm.cdf(-d[0]))

    @staticmethod
    def calculate_gamma(d:np.ndarray, instrument:EquityOption, model:Model):
        return (np.exp(-model.q * instrument.T) * norm.pdf(d[0])
         / (model.x0 * model.vol * np.sqrt(instrument.T)))

    @staticmethod
    def calculate_volga(d:np.ndarray, instrument:EquityOption, model:Model):
        return (model.x0 * np.sqrt(instrument.T) * np.exp(-model.q * instrument.T) * norm.pdf(d[0])
                * d[0] * d[1] / model.vol)

    @staticmethod
    def calculate_vanna(d:np.ndarray, instrument:EquityOption, model:Model):
        return -np.exp(-model.q * instrument.T) * norm.pdf(d[0]) * d[1] * (1 / model.vol)

class HestonAnalyticalEngine(Engine):
    def __init__(self):
        self.b = None
        self.u = None
        self.char_func1 = None
        self.char_func2 = None
        self.phi_bounds = None
        self.p1 = None
        self.p2 = None

    def setup_heston_params(self, instrument:EquityOption, model:Model):
        self.b = [model.reversion_speed - model.covariance[0, 1] * model.sigma, model.reversion_speed]
        self.u = [0.5, -0.5]
        self.char_func1 = lambda phi: (self.psi(phi=phi, t=instrument.T, r=model.r, q=model.q,
                                                rho=model.covariance[0, 1],
                                                a=(model.mean_vol ** 2) * model.reversion_speed,
                                                sigma=model.sigma, b=self.b[0], u=self.u[0],
                                                x0=np.log(model.x0), v0=model.vol ** 2))
        self.char_func2 = lambda phi: (self.psi(phi=phi, t=instrument.T, r=model.r, q=model.q,
                                                rho=model.covariance[0, 1],
                                                a=(model.mean_vol ** 2) * model.reversion_speed,
                                                sigma=model.sigma, b=self.b[1], u=self.u[1],
                                                x0=np.log(model.x0), v0=model.vol ** 2))

        self.phi_bounds = [1e-8, 750]

        self.p1 = self.get_big_p(char_func=self.char_func1, k=instrument.K, phi_bounds=self.phi_bounds)
        self.p2 = self.get_big_p(char_func=self.char_func2, k=instrument.K, phi_bounds=self.phi_bounds)

    def get_price(self, instrument:EquityOption, model:Model):
        # just ise BSM if sigma = 0
        if model.sigma < 1e-8:
            bsm_engine, bsm_model = self.retrieve_bsm_engine_from_heston(instrument=instrument, model=model)
            return bsm_engine.get_price(instrument=instrument, model=bsm_model)

        self.setup_heston_params(instrument=instrument, model=model)
        p_call = model.x0 * np.exp(-model.q * instrument.T) * self.p1 - instrument.K * np.exp(-model.r * instrument.T) * self.p2

        return {"value": p_call} if instrument.call \
            else {"value": p_call - model.x0 * np.exp(-model.q * instrument.T) + instrument.K * np.exp(-model.r * instrument.T)}

    def get_greeks(self, instrument:EquityOption, model:Model, greek_type:Union[list, str]):
        # just ise BSM if sigma = 0
        if model.sigma < 1e-8:
            bsm_engine, bsm_model = self.retrieve_bsm_engine_from_heston(instrument=instrument, model=model)
            return bsm_engine.get_greeks(instrument=instrument, model=bsm_model, greek_type=greek_type)

        self.setup_heston_params(instrument=instrument, model=model)
        greeks = {}

        if isinstance(greek_type, str):
            greek_type = [greek_type]

        for i in range(len(greek_type)):
            func_name = f"calculate_{greek_type[i]}"
            func = getattr(self, func_name)
            greeks[greek_type[i]] = func(instrument=instrument, model=model)

        return greeks

    def calculate_delta(self, instrument:EquityOption, model:Model):
        return (np.exp(-model.q * instrument.T) * self.p1 if instrument.call
                else np.exp(-model.q * instrument.T) * (self.p1 - 1))

    def calculate_vega(self, instrument:EquityOption, model:Model):
        _d1 = lambda phi: self.D(phi=phi, t=instrument.T, rho=model.covariance[0, 1], sigma=model.sigma, b=self.b[0], u=self.u[0])
        _d2 = lambda phi: self.D(phi=phi, t=instrument.T, rho=model.covariance[0, 1], sigma=model.sigma, b=self.b[1], u=self.u[1])

        integrand = lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) / (phi * 1j) * (
                                    model.x0 * np.exp(-model.q * instrument.T) * self.char_func1(phi) * _d1(phi)
                                - instrument.K * np.exp(-model.r * instrument.T) * self.char_func2(phi) *_d2(phi)
                                )).real

        bound_up = self.phi_bounds[0]
        bound_dn = self.phi_bounds[1]
        return 2 * model.vol * quad(integrand, bound_up, bound_dn)[0] / np.pi

    def calculate_rho(self, instrument:EquityOption, model:Model):
        return (instrument.K * instrument.T * np.exp(-model.r * instrument.T) * self.p2 if instrument.call
                else -instrument.K * instrument.T * np.exp(-model.r * instrument.T) * (1 - self.p2))

    def calculate_theta(self, instrument:EquityOption, model:Model):
        term1 = (model.q * model.x0 * np.exp(-model.q * instrument.T) * self.p1
                - model.r * instrument.K * np.exp(-model.r * instrument.T) * self.p2)

        _d1 = lambda phi: self.d(phi=phi, rho=model.covariance[0, 1], b=self.b[0], u=self.u[0], sigma=model.sigma)
        _g1 = lambda phi: self.g(phi=phi, rho=model.covariance[0, 1], b=self.b[0], u=self.u[0], sigma=model.sigma)
        _d2 = lambda phi: self.d(phi=phi, rho=model.covariance[0, 1], b=self.b[1], u=self.u[1], sigma=model.sigma)
        _g2 = lambda phi: self.g(phi=phi, rho=model.covariance[0, 1], b=self.b[1], u=self.u[1], sigma=model.sigma)

        _dC1 = lambda phi: (1j * (model.r - model.q) * phi
                            + (model.mean_vol**2 * model.reversion_speed / model.sigma**2)
                            * ((self.b[0] - model.covariance[0, 1] * model.sigma * 1j * phi + _d1(phi))
                            + (2 * _g1(phi) * _d1(phi) * np.exp(instrument.T * _d1(phi)))
                               / (1 - _g1(phi) * np.exp(instrument.T * _d1(phi)))))
        _dC2 = lambda phi: (1j * (model.r - model.q) * phi
                            + (model.mean_vol**2 * model.reversion_speed / model.sigma**2)
                            * ((self.b[1] - model.covariance[0, 1] * model.sigma * 1j * phi + _d2(phi))
                            + (2 * _g2(phi) * _d2(phi) * np.exp(instrument.T * _d2(phi)))
                               / (1 - _g2(phi) * np.exp(instrument.T * _d2(phi)))))
        _dD1 = lambda phi: (
                ((self.b[0] - model.covariance[0, 1] * model.sigma * 1j * phi + _d1(phi)) / model.sigma**2)
                * ((-_d1(phi) * np.exp(instrument.T * _d1(phi)) * (1 - _g1(phi)))
                    / ((1 - _g1(phi) * np.exp(instrument.T * _d1(phi))) ** 2))
        )
        _dD2 = lambda phi: (
                ((self.b[1] - model.covariance[0, 1] * model.sigma * 1j * phi + _d2(phi)) / model.sigma ** 2)
                * ((-_d2(phi) * np.exp(instrument.T * _d2(phi)) * (1 - _g2(phi)))
                   / ((1 - _g2(phi) * np.exp(instrument.T * _d2(phi))) ** 2))
        )

        _diff1 = lambda phi: _dC1(phi) + (model.vol ** 2) * _dD1(phi)
        _diff2 = lambda phi: _dC2(phi) + (model.vol ** 2) * _dD2(phi)

        integrand1 = (lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) / (phi * 1j) *
                                  self.char_func1(phi) * _diff1(phi)).real)
        integrand2 = (lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) / (phi * 1j) *
                                  self.char_func2(phi) * _diff2(phi)).real)

        integral1 = quad(integrand1, self.phi_bounds[0], self.phi_bounds[1])[0]
        integral2 = quad(integrand2, self.phi_bounds[0], self.phi_bounds[1])[0]

        term2 = (model.x0 * np.exp(-model.q * instrument.T) * integral1 / np.pi
               - instrument.K * np.exp(-model.r * instrument.T) * integral2 / np.pi)

        return term1 - term2 if instrument.call \
            else term1 - term2 + model.r * instrument.K * np.exp(-model.r * instrument.T) - model.q * model.x0 * np.exp(-model.q * instrument.T)

    def calculate_gamma(self, instrument:EquityOption, model:Model):
        integrand = lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) *
                                 self.char_func1(phi)).real
        scale = np.exp(-model.q * instrument.T) / (model.x0 * np.pi)
        return scale * quad(integrand, self.phi_bounds[0], self.phi_bounds[1])[0]

    def calculate_volga(self, instrument:EquityOption, model:Model):
        _d1 = lambda phi: self.D(phi=phi, t=instrument.T, rho=model.covariance[0, 1], sigma=model.sigma, b=self.b[0], u=self.u[0])
        _d2 = lambda phi: self.D(phi=phi, t=instrument.T, rho=model.covariance[0, 1], sigma=model.sigma, b=self.b[1], u=self.u[1])

        integrand = lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) / (phi * 1j) * (
                                 model.x0 * np.exp(-model.q * instrument.T) * self.char_func1(phi) * (_d1(phi) + 2 * (model.vol *  _d1(phi)) ** 2)
                                  - instrument.K * np.exp(-model.r * instrument.T) * self.char_func2(phi) * (_d2(phi) + 2 * (model.vol * _d2(phi))** 2)
                                )).real

        return (2 / np.pi) * quad(integrand, self.phi_bounds[0], self.phi_bounds[1])[0]


    def calculate_vanna(self, instrument:EquityOption, model:Model):
        _d1 = lambda phi: self.D(phi=phi, t=instrument.T, rho=model.covariance[0, 1], sigma=model.sigma, b=self.b[0], u=self.u[0])
        integrand = lambda phi: (np.exp(-phi * np.log(instrument.K) * 1j) / (1j * phi) * self.char_func1(phi) * _d1(phi)).real
        term = np.exp(-model.q * instrument.T) * quad(integrand, self.phi_bounds[0], self.phi_bounds[1])[0]
        return (2 * model.vol / np.pi) * term

    @staticmethod
    def get_big_p(char_func, k:float, phi_bounds:Union[list, np.ndarray]):
        func_to_pass = lambda phi: (np.exp(-phi * np.log(k) * 1j) * char_func(phi) / (1j * phi)).real
        return 0.5 + quad(func_to_pass, phi_bounds[0], phi_bounds[1])[0] / np.pi

    def psi(self, phi, t:float, r:float, q:float, rho:float, a:float, sigma:float, b:float, u:float, x0:float, v0:float):
        return np.exp(self.C(phi=phi, t=t, r=r, q=q, rho=rho, a=a, sigma=sigma, b=b, u=u) + v0 * self.D(phi=phi, t=t, rho=rho, sigma=sigma, b=b, u=u) + 1j * x0 * phi)

    def D(self, phi, t:float, rho:float, sigma:float, b:float, u:float):
        _d = self.d(phi=phi, rho=rho, b=b, u=u, sigma=sigma)
        _g = self.g(phi=phi, rho=rho, b=b, u=u, sigma=sigma)

        return ((b - rho * sigma * 1j * phi + _d) / (sigma ** 2)) * (
                (1 - np.exp(_d * t)) / (1 - _g * np.exp(_d * t))
                )

    def C(self, phi, t:float, r:float, q:float, rho:float, a:float, sigma:float, b:float, u:float):
        _d = self.d(phi=phi, rho=rho, b=b, u=u, sigma=sigma)
        _g = self.g(phi=phi, rho=rho, b=b, u=u, sigma=sigma)

        return (1j * (r - q) * phi * t
                + (a / sigma ** 2) * (
                        (b - rho * sigma * 1j * phi + _d) * t
                        - 2 * np.log((1 - _g * np.exp(_d * t)) / (1 - _g))
                        )
                )

    def g(self, phi, rho:float, b:float, u:float, sigma:float):
        _d = self.d(phi=phi, rho=rho, b=b, u=u, sigma=sigma)
        return (b - 1j * rho * sigma * phi + _d) / (b - 1j * rho * sigma * phi - _d)

    @staticmethod
    def d(phi, rho:float, b:float, u:float, sigma:float):
        z = (1j * rho * sigma * phi - b) ** 2 - sigma ** 2 * (2j * u * phi - phi ** 2)
        return -np.sqrt(z)

    def retrieve_bsm_engine_from_heston(self, instrument:EquityOption, model:Model):
        if abs(model.vol ** 2 - model.mean_vol ** 2) < 1e-8 or model.reversion_speed < 1e-8:
            sigma = model.vol
        else:
            # Mean reversion adjustment for variance
            sigma = np.sqrt(model.mean_vol ** 2 + (model.vol ** 2 - model.mean_vol ** 2) * (
                        1 - np.exp(-model.reversion_speed * instrument.T)) / (model.reversion_speed * instrument.T))
        bsm_engine = BSMAnalyticalEngine(quiet=self.quiet)
        bsm_model = BSM(x0=model.x0, r=model.r, q=model.q, vol=sigma)
        return bsm_engine, bsm_model
