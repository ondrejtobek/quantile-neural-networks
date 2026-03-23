import pandas as pd
import numpy as np
import arch

from numba import njit, prange


taus = (
    [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.075]
    + [0.925, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99995]
    + [i / 20 for i in range(1, 20)]
)
taus.sort()


def GARCH(r, ModelName="sGARCH", p=1, q=1, dist="t", EstimateMean=True):
    try:
        if EstimateMean:
            model = arch.univariate.ConstantMean(r)
        else:
            model = arch.univariate.ZeroMean(r)
        # select volatility process
        if ModelName == "sGARCH":
            model.volatility = arch.univariate.GARCH(p=p, q=q)
        elif ModelName == "GJRGARCH":
            model.volatility = arch.univariate.GARCH(p=p, q=q, o=p)
        elif ModelName == "EGARCH":
            model.volatility = arch.univariate.EGARCH(p=p, q=q, o=p)
        # select distribution
        if dist == "norm":
            model.distribution = arch.univariate.Normal()
        elif dist == "skewt":
            model.distribution = arch.univariate.SkewStudent()
        elif dist == "t":
            model.distribution = arch.univariate.StudentsT()
        # estimate
        fit = model.fit(disp="off", show_warning=False)
        # extract parameters
        res = fit.params.to_dict()
        res["vol"] = fit._volatility[-1]
        res["vol2"] = fit._volatility[-2]
        res["inov"] = fit.resid[-1] / fit._volatility[1]
        res["inov2"] = fit.resid[-2] / fit._volatility[2]
        # get convergence flag
        res["Converged"] = fit.convergence_flag == 0
        pars = fit.params[[i for i in fit.params.index if i[:5] in ["gamma", "alpha"]]]
        res["Converged2"] = True
        if np.any(np.abs(pars) > 0.9):
            res["Converged2"] = False
        return pd.DataFrame(res, index=[0])
    except:
        return None


# bootstrap simulations
@njit(parallel=True)
def vol_bootstrap_worker(mu, omega, alpha, beta, vol_init, inov_init, h, draws, taus):
    r = np.empty(draws)
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.normal()
            vol[i, j] = omega + alpha * (inov[i - 1, j] ** 2 * vol[i - 1, j]) + beta * vol[i - 1, j]
        r[j] = (1 + (mu + inov[1:, j] * np.sqrt(vol[1:, j])) / 100).prod() - 1
    return np.append(np.quantile(r, taus), np.std(r))


def vol_bootstrap(df, h=21, draws=100000):
    omega = df["omega"].iloc[0]
    alpha = df["alpha[1]"].iloc[0]
    beta = df["beta[1]"].iloc[0]
    vol_init = df["vol"].iloc[0] ** 2
    inov_init = df["inov"].iloc[0]
    mu = df["mu"].iloc[0]
    pred = vol_bootstrap_worker(mu, omega, alpha, beta, vol_init, inov_init, h, draws, taus)
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "vol": pred[-1]}
        | {f"Q{tau}": pred[i] for i, tau in enumerate(taus)},
        index=[0],
    )
    return out


@njit(parallel=True)
def vol_bootstrap_worker_t(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, taus):
    r = np.empty(draws)
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = omega + alpha * (inov[i - 1, j] ** 2 * vol[i - 1, j]) + beta * vol[i - 1, j]
        r[j] = (1 + (mu + inov[1:, j] * np.sqrt(vol[1:, j])) / 100).prod() - 1
    return np.append(np.quantile(r, taus), np.std(r))


def vol_bootstrap_t(df, h=21, draws=100000, mu=0):
    omega = df["omega"].iloc[0]
    alpha = df["alpha[1]"].iloc[0]
    beta = df["beta[1]"].iloc[0]
    nu = df["nu"].iloc[0]
    vol_init = df["vol"].iloc[0] ** 2
    inov_init = df["inov"].iloc[0]
    mu = df["mu"].iloc[0]
    pred = vol_bootstrap_worker_t(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, taus)
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "vol": pred[-1]}
        | {f"Q{tau}": pred[i] for i, tau in enumerate(taus)},
        index=[0],
    )
    return out


@njit(parallel=True)
def vol_bootstrap2_worker_t(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, taus):
    r = np.empty(draws)
    inov = np.empty((h + 2, draws))
    vol = np.empty((h + 2, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    p = len(alpha)
    q = len(beta)
    inov[0, :] = inov_init[0]
    inov[1, :] = inov_init[1]
    vol[0, :] = vol_init[0]
    vol[1, :] = vol_init[1]
    for j in prange(draws):
        for i in range(2, h + 2):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = omega
            for k in range(p):
                vol[i, j] += alpha[k] * np.square(inov[i - 1 - k, j]) * vol[i - 1 - k, j]
            for l in range(q):
                vol[i, j] += beta[l] * vol[i - 1 - l, j]
        r[j] = (1 + (mu + inov[2:, j] * np.sqrt(vol[2:, j])) / 100).prod() - 1
    return np.append(np.quantile(r, taus), np.std(r))


def vol_bootstrap2_t(df, h=21, draws=100000):
    omega = df["omega"].iloc[0]
    alpha = [df["alpha[1]"].iloc[0]]
    if "alpha[2]" in df.columns:
        alpha += [df["alpha[2]"].iloc[0]]
    beta = [df["beta[1]"].iloc[0]]
    if "beta[2]" in df.columns:
        beta += [df["beta[2]"].iloc[0]]
    nu = df["nu"].iloc[0]
    vol_init = [df["vol2"].iloc[0] ** 2, df["vol"].iloc[0] ** 2]
    inov_init = [df["inov2"].iloc[0], df["inov"].iloc[0]]
    mu = df["mu"].iloc[0]
    pred = vol_bootstrap2_worker_t(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, taus)
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "vol": pred[-1]}
        | {f"Q{tau}": pred[i] for i, tau in enumerate(taus)},
        index=[0],
    )
    return out


@njit(parallel=True)
def vol_bootstrap_worker_gjr_t(mu, omega, alpha, beta, gamma, vol_init, inov_init, nu, h, draws, taus):
    r = np.empty(draws)
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = (
                omega
                + (alpha + (inov[i - 1, j] < 0) * gamma) * np.square(inov[i - 1, j]) * vol[i - 1, j]
                + beta * vol[i - 1, j]
            )
        r[j] = (1 + (mu + inov[1:, j] * np.sqrt(vol[1:, j]) / 100)).prod() - 1
    return np.append(np.quantile(r, taus), np.std(r))


def vol_bootstrap_gjr_t(df, h=21, draws=100000):
    omega = df["omega"].iloc[0]
    alpha = df["alpha[1]"].iloc[0]
    beta = df["beta[1]"].iloc[0]
    gamma = df["gamma[1]"].iloc[0]
    nu = df["nu"].iloc[0]
    vol_init = df["vol"].iloc[0] ** 2
    inov_init = df["inov"].iloc[0]
    mu = df["mu"].iloc[0]
    pred = vol_bootstrap_worker_gjr_t(mu, omega, alpha, beta, gamma, vol_init, inov_init, nu, h, draws, taus)
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "vol": pred[-1]}
        | {f"Q{tau}": pred[i] for i, tau in enumerate(taus)},
        index=[0],
    )
    return out


@njit(parallel=True)
def vol_bootstrap_worker_egarch_t(mu, omega, alpha, beta, gamma, vol_init, inov_init, nu, h, draws, taus):
    r = np.empty(draws)
    inov = np.empty((h + 1, draws))
    logvar = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    mean_z = 0.7978845608028654
    for j in prange(draws):
        inov[0, j] = inov_init
        logvar[0, j] = 2 * np.log(vol_init)
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            logvar[i, j] = (
                omega
                + alpha * (np.abs(inov[i - 1, j]) - mean_z)
                + gamma * inov[i - 1, j]
                + beta * logvar[i - 1, j]
            )
        r[j] = (1 + (mu + inov[1:, j] * np.sqrt(np.exp(logvar[1:, j] / 2))) / 100).prod() - 1
    return np.append(np.quantile(r, taus), np.std(r))


def vol_bootstrap_egarch_t(df, h=21, draws=100000):
    omega = df["omega"].iloc[0]
    alpha = df["alpha[1]"].iloc[0]
    beta = df["beta[1]"].iloc[0]
    gamma = df["gamma[1]"].iloc[0]
    nu = df["nu"].iloc[0]
    vol_init = df["vol"].iloc[0] ** 2
    inov_init = df["inov"].iloc[0]
    mu = df["mu"].iloc[0]
    pred = vol_bootstrap_worker_egarch_t(mu, omega, alpha, beta, gamma, vol_init, inov_init, nu, h, draws, taus)
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "vol": pred[-1]}
        | {f"Q{tau}": pred[i] for i, tau in enumerate(taus)},
        index=[0],
    )
    return out


@njit(parallel=True)
def vol_bootstrap_worker_t_scoring(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, y_true):
    r = np.empty(draws)
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = omega + alpha * (inov[i - 1, j] ** 2 * vol[i - 1, j]) + beta * vol[i - 1, j]
        r[j] = (1 + (mu + inov[1:, j] * np.sqrt(vol[1:, j])) / 100).prod() - 1
    return r


def vol_bootstrap_t_scoring(df, h=21, draws=100000, mu=0):
    omega = df["omega"].iloc[0]
    alpha = df["alpha[1]"].iloc[0]
    beta = df["beta[1]"].iloc[0]
    nu = df["nu"].iloc[0]
    vol_init = df["vol"].iloc[0] ** 2
    inov_init = df["inov"].iloc[0]
    mu = df["mu"].iloc[0]
    y_true = df["r"].iloc[0]
    r = vol_bootstrap_worker_t_scoring(mu, omega, alpha, beta, vol_init, inov_init, nu, h, draws, y_true)
    r = r.clip(-1, 50)
    n = 5000
    probs = np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
    Qs = np.quantile(r, probs)
    score = np.sum(probs * (y_true - Qs).clip(0) + (1 - probs) * (Qs - y_true).clip(0)) / n * 2
    out = pd.DataFrame(
        {"date": df["date"].iloc[0], "DTID": df["DTID"].iloc[0], "score": score},
        index=[0],
    )
    return out
