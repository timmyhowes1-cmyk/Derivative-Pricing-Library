import numpy as np

def expanding_mean_axis1(x, arithmetic_mean=True):
    n = np.arange(1, x.shape[1] + 1)
    return np.cumsum(x, axis=1) / n if arithmetic_mean else np.exp(np.cumsum(np.log(x), axis=1) / n)

def fit_continuation_lstsq(x, y, deg=2, ridge=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < deg + 1:
        return None

    x_mean = x.mean()
    x_std = x.std()

    if x_std < 1e-12:
        return None

    z = (x - x_mean) / x_std
    X = np.column_stack([z**k for k in range(deg + 1)])

    if ridge > 0:
        A = X.T @ X + ridge * np.eye(X.shape[1])
        b = X.T @ y
        beta = np.linalg.solve(A, b)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=float)
        z_new = (x_new - x_mean) / x_std
        X_new = np.column_stack([z_new**k for k in range(deg + 1)])
        return X_new @ beta

    return predict

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

def format_for_scheme(param, shape):
    if isinstance(param, float) or isinstance(param, int):
        return param * np.ones(shape=shape)
    return param




