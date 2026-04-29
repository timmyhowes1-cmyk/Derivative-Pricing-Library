from scipy.spatial.distance import correlation
from engines.equity import Engine
from models.equity import Model
from instruments.equity import EquityOption
from utils.math_utils import *
import copy
import numpy as np
from typing import Union

class MonteCarloEngine(Engine):
    def __init__(self, iterations:float=1e4, timestep:float=1/252, greek_bump_size:Union[float, np.ndarray]=0.01, antithetic_variates:bool=False):
        self.iterations = iterations
        self.timestep = timestep
        self.greek_bump_size = greek_bump_size
        self.antithetic_variates = antithetic_variates

    def get_price(self, instrument:EquityOption, model:Model):
        paths = model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=None, antithetic_variates=self.antithetic_variates)
        if instrument.european:
            samples = np.exp(-model.r * instrument.T) * instrument.payoff(paths)
        else:
            samples = self.get_ls_american_values(instrument=instrument, model=model, paths=paths, deg=2)

        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def get_greeks(self, instrument:EquityOption, model:Model, greek_type:Union[list, str]):
        dw = generate_wiener_increments(n=self.iterations, dt=self.timestep, expiry=instrument.T, cov=getattr(model, 'covariance', None), antithetic_variates=self.antithetic_variates)
        paths = model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=dw, antithetic_variates=self.antithetic_variates)

        greeks = {}
        if isinstance(greek_type, str):
            greek_type = [greek_type]
        if isinstance(self.greek_bump_size, int) or isinstance(self.greek_bump_size, float):
            bump_size = [self.greek_bump_size]
        else:
            bump_size = self.greek_bump_size
        assert len(greek_type) == len(bump_size)

        for i in range(len(greek_type)):
            func_name = f"calculate_{greek_type[i]}"
            func = getattr(self, func_name)
            greeks[greek_type[i]] = func(instrument=instrument, model=model, bump_size=bump_size[i], dw=dw, paths=paths)
        return greeks

    def get_ls_american_values(self, instrument:EquityOption, model:Model, paths:np.ndarray, deg:int=2):
        n_timestep = paths.shape[-1] - 1
        timestep = instrument.T / n_timestep
        stop_time_idx = np.array([n_timestep-1] * self.iterations)
        exercise_payoffs = instrument.payoff(paths)

        for i in range(n_timestep - 2, 0, -1):
            itm = exercise_payoffs[:, i] > 0
            if itm.sum() >= deg + 1:
                current_t = i * timestep
                future_cf = exercise_payoffs[np.arange(self.iterations), stop_time_idx]
                discounted_cf = np.exp(-model.r * (stop_time_idx * timestep - current_t)) * future_cf

                x = np.log(paths[itm, i] / instrument.K)
                y = discounted_cf[itm]

                good = np.isfinite(x) & np.isfinite(y)
                x = x[good]
                y = y[good]

                predict = fit_continuation_lstsq(x, y, deg=deg, ridge=1e-8)
                if predict is not None:
                    continuation_values = predict(x)

                    early_exercise = exercise_payoffs[itm, i][good] > continuation_values

                    itm_idx = np.where(itm)[0][good]
                    stop_time_idx[itm_idx[early_exercise]] = i
        return np.exp(-model.r * stop_time_idx * timestep) * exercise_payoffs[np.arange(self.iterations), stop_time_idx]

    def _generic_first_order_greek(self, instrument:EquityOption, model:Model, attribute:str, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        dw, paths = self._ensure_paths(instrument=instrument, model=model, dw=dw, paths=paths)

        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)

            bumped_paths = new_model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=dw)
            if instrument.european:
                return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)
            else:
                return self.get_ls_american_values(instrument=instrument, model=new_model, paths=bumped_paths)

        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        samples = (p_up - p_down) / (2 * bump_size)

        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def _generic_second_order_greek(self, instrument:EquityOption, model:Model, attribute:str, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        dw, paths = self._ensure_paths(instrument, model, dw=dw, paths=paths)

        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)

            bumped_paths = new_model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=dw)
            if instrument.european:
                return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)
            else:
                return self.get_ls_american_values(instrument=instrument, model=new_model, paths=bumped_paths)

        p_0 = np.exp(-model.r * instrument.T) * instrument.payoff(paths) if instrument.european \
            else self.get_ls_american_values(instrument=instrument, model=model, paths=paths)
        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        samples = (p_up - 2 * p_0 + p_down) / (bump_size ** 2)
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def calculate_delta(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        return self._generic_first_order_greek(instrument=instrument, model=model, attribute="x0", bump_size=bump_size, dw=dw, paths=paths)

    def calculate_vega(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        return self._generic_first_order_greek(instrument=instrument, model=model, attribute="vol", bump_size=bump_size, dw=dw, paths=paths)

    def calculate_rho(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        return self._generic_first_order_greek(instrument=instrument, model=model, attribute="r", bump_size=bump_size, dw=dw, paths=paths)

    def calculate_theta(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        timestep = bump_size if bump_size is not None else self.timestep
        if abs(timestep - self.timestep) > 1e-10:
            dw = generate_wiener_increments(n=self.iterations, dt=timestep, expiry=instrument.T, cov=getattr(model, 'covariance', None), antithetic_variates=self.antithetic_variates)
            paths = model.generate_paths(iterations=self.iterations, timestep=timestep, expiry=instrument.T, dw=dw)
        else:
            dw, paths = self._ensure_paths(instrument=instrument, model=model, dw=dw, paths=paths)

        new_instrument = copy.deepcopy(instrument)
        new_instrument.T += -timestep
        new_model = copy.deepcopy(model)
        if np.ndim(dw) <= 2:
            bumped_paths = new_model.generate_paths(iterations=self.iterations, timestep=timestep,
                                                    expiry=new_instrument.T, dw=dw[..., :-1])
        else:
            bumped_paths = new_model.generate_paths(iterations=self.iterations, timestep=timestep,
                                                    expiry=new_instrument.T, dw=dw[:, :-1, :])

        if instrument.european:
            p0 = np.exp(-model.r * instrument.T) * instrument.payoff(paths)
            p1 = new_instrument.payoff(np.array([[model.s0]] * self.iterations)) if bumped_paths.shape[-1] == 0 \
                else np.exp(-model.r * new_instrument.T) * new_instrument.payoff(bumped_paths)
        else:
            p0 =  self.get_ls_american_values(instrument=instrument, model=model, paths=paths)
            p1 = self.get_ls_american_values(instrument=new_instrument, model=new_model, paths=bumped_paths)

        samples = (p1 - p0) / timestep
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(samples.shape[0])}

    def calculate_gamma(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        return self._generic_second_order_greek(instrument=instrument, model=model, attribute="x0", bump_size=bump_size, dw=dw, paths=paths)

    def calculate_volga(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        return self._generic_second_order_greek(instrument=instrument, model=model, attribute="vol", bump_size=bump_size, dw=dw, paths=paths)

    def calculate_vanna(self, instrument:EquityOption, model:Model, bump_size:float, dw:np.ndarray=None, paths:np.ndarray=None):
        assert len(bump_size) == 2
        dw, _ = self._ensure_paths(instrument=instrument, model=model, dw=dw, paths=paths)

        def get_bumped_p(x0_adj, vol_adj):
            new_model = copy.deepcopy(model)
            new_model.x0 += x0_adj
            new_model.vol += vol_adj

            bumped_paths = new_model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=dw)
            if instrument.european:
                return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)
            else:
                return self.get_ls_american_values(instrument=instrument, model=new_model, paths=bumped_paths)

        p_uu = get_bumped_p(bump_size[0], bump_size[1])
        p_ud = get_bumped_p(bump_size[0], -bump_size[1])
        p_du = get_bumped_p(-bump_size[0], bump_size[1])
        p_dd = get_bumped_p(-bump_size[0], -bump_size[1])

        samples = (p_uu - p_ud - p_du + p_dd) / (4 * bump_size[0] * bump_size[1])
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(samples.shape[0])}

    def _ensure_paths(self, instrument:EquityOption, model:Model, dw:np.ndarray=None, paths:np.ndarray=None):
        """
        Ensures that both Wiener increments and paths are available.
        Returns: (dw, paths)
        """
        if dw is None:
            dw = generate_wiener_increments(n=self.iterations, dt=self.timestep, expiry=instrument.T, cov=getattr(model, 'covariance', None), antithetic_variates=self.antithetic_variates)

        if paths is None:
            paths = model.generate_paths(iterations=self.iterations, timestep=self.timestep, expiry=instrument.T, dw=dw)

        return dw, paths
