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

def generate_multi_stdnormal(cov, size):
    rng = np.random.default_rng()
    n_variables = cov.shape[0]
    assert np.all(np.diagonal(cov) == 1), "Covariance matrix must have diagonal entries equal to 1"
    return rng.multivariate_normal(np.zeros(n_variables), cov, size=size)

def generate_wiener_increments(n, dt, expiry, cov=None, antithetic_variates=False):
    gen_func = get_antithetic_stdnormal if antithetic_variates else generate_multi_normal
    if cov is not None:
        assert np.all(np.diagonal(cov) == 1), "Covariance matrix must have diagonal entries equal to 1"
    size = (n, int(round(expiry / dt)))
    dw = np.sqrt(dt) * gen_func(cov, size=size)
    return dw

def get_antithetic_stdnormal(cov, size):
    split = int(size[-2] // 2)
    new_size = list(size[:])
    new_size[-2] = split
    new_size = tuple(new_size)
    shocks = generate_standard_normal(new_size) if cov is None else generate_multi_stdnormal(cov, new_size)
    shocks = np.concatenate([shocks, -shocks], axis=-2)

    if size[0] % 2 == 0:
        return shocks
    elif cov is not None:
        return np.concatenate([shocks, generate_multi_stdnormal(cov, size=(1, size[-1]))], axis=0)
    else:
        return np.concatenate([shocks, generate_standard_normal(size=(1, size[-1]))], axis=0)

def format_for_scheme(param, shape):
    if isinstance(param, float) or isinstance(param, int):
        return param * np.ones(shape=shape)
    return param




