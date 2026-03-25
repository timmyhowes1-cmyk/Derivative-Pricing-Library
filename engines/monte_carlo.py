from scipy.spatial.distance import correlation

from engines.base import Engine
from utils.math_utils import *
import copy
import numpy as np

class MonteCarloEngine(Engine):
    def __init__(self, iterations=10000, timestep=0.5, greek_bump_size=None, quiet=False, use_antithetic_variates=False):
        super().__init__(quiet)
        self.iterations = iterations
        self.timestep = timestep
        self.greek_bump_size = greek_bump_size
        self.antithetic_variates = use_antithetic_variates

    def get_price(self, instrument, model):
        print(f"*** {model.__class__.__name__.upper()} MC MODEL ***\nCALCULATING PRICE...\n") if self.quiet is False else None
        paths = model.generate_paths(self.iterations, self.timestep, instrument.T)
        samples = np.exp(-model.r * instrument.T) * instrument.payoff(paths)

        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def get_greeks(self, instrument, model, greek_type="delta"):
        print(f"*** {model.__class__.__name__.upper()} MC MODEL ***\nCALCULATING GREEKS...\n") if self.quiet is False else None
        corr = getattr(model, 'correlation', None)
        dw = generate_wiener_increments(self.iterations, self.timestep, instrument.T, correlation=corr)
        paths = model.generate_paths(self.iterations, self.timestep, instrument.T, dw=dw)

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
            greeks[greek_type[i]] = func(instrument, model, bump_size[i], dw, paths)
        print("\n") if self.quiet is False else None
        return greeks

    def _generic_first_order_greek(self, instrument, model, attribute, bump_size, dw=None, paths=None):
        dw, paths = self._ensure_paths(instrument, model, dw=dw, paths=paths)

        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)

            bumped_paths = new_model.generate_paths(self.iterations, self.timestep, instrument.T, dw=dw)
            return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)

        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        samples = (p_up - p_down) / (2 * bump_size)

        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def _generic_second_order_greek(self, instrument, model, attribute, bump_size, dw=None, paths=None):
        dw, paths = self._ensure_paths(instrument, model, dw=dw, paths=paths)

        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)

            bumped_paths = new_model.generate_paths(self.iterations, self.timestep, instrument.T, dw=dw)
            return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)

        p_0 = np.exp(-model.r * instrument.T) * instrument.payoff(paths)
        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        samples = (p_up - 2 * p_0 + p_down) / (bump_size ** 2)
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(self.iterations)}

    def calculate_delta(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING DELTA... ") if self.quiet is False else None
        return self._generic_first_order_greek(instrument, model, "x0", bump_size, dw, paths)

    def calculate_vega(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING VEGA... ") if self.quiet is False else None
        return self._generic_first_order_greek(instrument, model, "vol", bump_size, dw, paths)

    def calculate_rho(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING RHO... ") if self.quiet is False else None
        return self._generic_first_order_greek(instrument, model, "r", bump_size, dw, paths)

    def calculate_theta(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING THETA... ") if self.quiet is False else None
        timestep = bump_size if bump_size is not None else self.timestep
        if abs(timestep - self.timestep) > 1e-10:
            print("BUMP-SIZE DIFFERENT TO TIMESTEP => GENERATING NEW  WIENER INCREMENTS FOR THETA...") if self.quiet is False else None
            dw = generate_wiener_increments(self.iterations, timestep, instrument.T, correlation=getattr(model, 'correlation', None), antithetic_variates=self.antithetic_variates)
            paths = model.generate_paths(self.iterations, timestep, instrument.T, dw=dw)
        else:
            dw, paths = self._ensure_paths(instrument, model, dw=dw, paths=paths)

        p0 = np.exp(-model.r * instrument.T) * instrument.payoff(paths)
        new_instrument = copy.deepcopy(instrument)
        new_instrument.T += -timestep
        new_model = copy.deepcopy(model)
        bumped_paths = new_model.generate_paths(self.iterations, timestep, new_instrument.T, dw=dw[..., :-1])

        if bumped_paths.shape[1] == 0:
            p1 = new_instrument.payoff(np.array([[model.s0]] * self.iterations))
        else:
            p1 = np.exp(-model.r * new_instrument.T) * new_instrument.payoff(bumped_paths)
        samples = (p1 - p0) / timestep
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(samples.shape[0])}

    def calculate_gamma(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING GAMMA... ") if self.quiet is False else None
        return self._generic_second_order_greek(instrument, model, "x0", bump_size, dw=dw, paths=paths)

    def calculate_volga(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING VOLGA... ") if self.quiet is False else None
        return self._generic_second_order_greek(instrument, model, "vol", bump_size, dw=dw, paths=paths)

    def calculate_vanna(self, instrument, model, bump_size, dw=None, paths=None):
        print("CALCULATING VANNA... ") if self.quiet is False else None
        assert len(bump_size) == 2
        dw, _ = self._ensure_paths(instrument, model, dw=dw, paths=paths)

        def get_bumped_p(x0_adj, vol_adj):
            new_model = copy.deepcopy(model)
            new_model.x0 += x0_adj
            new_model.vol += vol_adj

            bumped_paths = new_model.generate_paths(self.iterations, self.timestep, instrument.T, dw=dw)
            return np.exp(-new_model.r * instrument.T) * instrument.payoff(bumped_paths)

        p_uu = get_bumped_p(bump_size[0], bump_size[1])
        p_ud = get_bumped_p(bump_size[0], -bump_size[1])
        p_du = get_bumped_p(-bump_size[0], bump_size[1])
        p_dd = get_bumped_p(-bump_size[0], -bump_size[1])

        samples = (p_uu - p_ud - p_du + p_dd) / (4 * bump_size[0] * bump_size[1])
        return {"value":np.mean(samples), "std_error":np.std(samples, ddof=1) / np.sqrt(samples.shape[0])}

    def _ensure_paths(self, instrument, model, dw=None, paths=None):
        """
        Ensures that both Wiener increments and paths are available.
        Returns: (dw, paths)
        """
        if dw is None:
            print("GENERATING NEW WIENER PROCESS\n")
            dw = generate_wiener_increments(self.iterations, self.timestep, instrument.T, correlation=model.correlation, antithetic_variates=self.antithetic_variates)

        if paths is None:
            print("GENERATING NEW WIENER PROCESS\n")
            paths = model.generate_paths(self.iterations, self.timestep, instrument.T, dw=dw)

        return dw, paths
