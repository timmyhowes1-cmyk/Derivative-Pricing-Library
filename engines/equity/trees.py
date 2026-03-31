from engines.equity import Engine
from engines.equity import Model
from instruments.equity import EquityOption
import numpy as np
import copy
from typing import Union

# only compatible with BSM model and non pathwise dependent payoffs
class BinomialTree(Engine):
    def __init__(self, greek_bump_size:Union[float, np.ndarray]=0.01, timestep:float=1/252, quiet:bool=False):
        self.greek_bump_size = greek_bump_size
        self.timestep = timestep
        self.quiet=quiet

    def get_risk_neutral_prob(self, model:Model, up:float, down:float):
        p1 = (np.exp(self.timestep * (model.r - model.q)) - down) / (up - down)
        p2 = 1 - p1
        return p1, p2

    def get_price(self, instrument:EquityOption, model:Model):
        assert instrument.__class__.__name__ in ["Vanilla", "Digital"], "Tree pricing only for non path-dependent payoffs"
        n = int(np.round(instrument.T / self.timestep))
        j = np.arange(n + 1)
        up, down = model.get_tree_factors(self.timestep)
        p1, p2 = self.get_risk_neutral_prob(model, up, down)

        terminal_x = model.x0 * (up ** j) * (down ** (n - j))
        price = np.atleast_1d(instrument.payoff(terminal_x))
        discount_factor = np.exp(-model.r * self.timestep)

        for i in range(n - 1, -1, -1):
            assert np.ndim(price) == 1
            if instrument.european:
                price = discount_factor * (p1 * price[1:(i + 2)] + p2 * price[0:(i + 1)])
            else:
                j = np.arange(i + 1)
                x = model.x0 * (up ** j) * (down ** (i - j))
                price = np.maximum(discount_factor * (p1 * price[1:(i + 2)] + p2 * price[0:(i + 1)]), instrument.payoff(x))

        return {"value": price[0]}

    def get_greeks(self, instrument:EquityOption, model:Model, greek_type:Union[list, str]):
        print(f"*** {model.__class__.__name__.upper()} MC MODEL ***\nCALCULATING GREEKS...\n") if self.quiet is False else None

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
            greeks[greek_type[i]] = func(instrument, model, bump_size=bump_size[i])
        print("\n") if self.quiet is False else None
        return greeks

    def _generic_first_order_greek(self, instrument:EquityOption, model:Model, attribute:str, bump_size:float):
        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)
            return self.get_price(instrument, new_model)["value"]

        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        return (p_up - p_down) / (2 * bump_size)

    def _generic_second_order_greek(self, instrument:EquityOption, model:Model, attribute:str, bump_size:float):
        p_0 = self.get_price(instrument, model)["value"]
        def get_p(param_adj):
            new_model = copy.deepcopy(model)
            current_val = getattr(new_model, attribute)
            setattr(new_model, attribute, current_val + param_adj)
            return self.get_price(instrument, new_model)["value"]

        p_up = get_p(bump_size)
        p_down = get_p(-bump_size)

        return (p_up - 2 * p_0 + p_down) / (bump_size ** 2)

    def calculate_delta(self, instrument:EquityOption, model:Model, bump_size:float):
        return self._generic_first_order_greek(instrument, model, attribute="x0", bump_size=bump_size)

    def calculate_vega(self, instrument:EquityOption, model:Model, bump_size:float):
        return self._generic_first_order_greek(instrument, model, attribute="vol", bump_size=bump_size)

    def calculate_rho(self, instrument:EquityOption, model:Model, bump_size:float):
        return self._generic_first_order_greek(instrument, model, attribute="r", bump_size=bump_size)

    def calculate_gamma(self, instrument:EquityOption, model:Model, bump_size:float):
        return self._generic_second_order_greek(instrument, model, attribute="x0", bump_size=bump_size)

    def calculate_volga(self, instrument:EquityOption, model:Model, bump_size:float):
        return self._generic_second_order_greek(instrument, model, attribute="vol", bump_size=bump_size)

    def calculate_theta(self, instrument:EquityOption, model:Model, **kwargs):
        new_instrument = copy.deepcopy(instrument)
        p_0 = self.get_price(instrument, model)["value"]
        new_instrument.T += -self.timestep
        p_up = self.get_price(new_instrument, model)["value"]

        return (p_up - p_0) / self.timestep

    def calculate_vanna(self, instrument:EquityOption, model:Model, bump_size:float):
        def get_p(x0_adj, vol_adj):
            new_model = copy.deepcopy(model)
            new_model.x0 += x0_adj
            new_model.vol += vol_adj
            return self.get_price(instrument, new_model)["value"]

        p_uu = get_p(bump_size[0], bump_size[1])
        p_ud = get_p(bump_size[0], -bump_size[1])
        p_du = get_p(-bump_size[0], bump_size[1])
        p_dd = get_p(-bump_size[0], -bump_size[1])

        return (p_uu - p_ud - p_du + p_dd) / (4 * bump_size[0] * bump_size[1])

