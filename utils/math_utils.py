import numpy as np
from numerical_schemes import *

def simpsons_rule(f, a, b, n=5000):
    if n % 2 != 0:
        raise ValueError("Number of intervals 'n' must be even.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])
    integral += 2 * np.sum(y[2:-1:2])

    return (h / 3) * integral

def generate_standard_normal(size):
    rng = np.random.default_rng()
    return rng.standard_normal(size)

def generate_wiener_increments(n, dt, expiry, correlation=None, antithetic_variates=False):
    gen_func = get_antithetic_stdnormal if antithetic_variates else generate_standard_normal
    dw = np.sqrt(dt) * gen_func((n, int(round(expiry / dt))))

    if correlation is not None:
        shocks = gen_func((n, int(round(expiry / dt))))
        dw2 = correlation * dw + np.sqrt(1 - correlation ** 2) * np.sqrt(dt) * shocks
        return np.array([dw, dw2])
    return dw


def get_antithetic_stdnormal(shape):
    split = int(shape[0] // 2)
    shocks = generate_standard_normal((split, shape[1]))
    shocks = np.concatenate([shocks, -shocks], axis=0)
    return shocks if shape[0] % 2 == 0 \
        else np.concatenate([shocks, generate_standard_normal((1, shape[1]))], axis=0)

def itself(c, x, t):
    return c

def standard_drift_vol(c, x, t):
    return c * x

def heston_vol(sigma, x, t):
    return np.sqrt(sigma) * x

def cir_drift(theta, k, x):
    return theta * (k - x)

def cir_vol(sigma, x):
    return sigma * np.sqrt(x)

def cir_vol_derivative(sigma, x):
    return 0.5 * sigma / np.sqrt(x)

def create_scheme(scheme_name, *args, **kwargs):
    # 1. Get all classes that inherit from NumericalScheme
    subclasses = NumericalScheme.__subclasses__()

    # 2. Find the one matching the string
    for cls in subclasses:
        if cls.__name__ == scheme_name:
            # 3. Instantiate and return
            return cls(*args, **kwargs)

    # 4. Fallback if not found
    raise ValueError(f"Scheme '{scheme_name}' not found. Options are: {[c.__name__ for c in subclasses]}")

def format_for_scheme(param, shape):
    if isinstance(param, float) or isinstance(param, int):
        return param * np.ones(shape=shape)
    return param


