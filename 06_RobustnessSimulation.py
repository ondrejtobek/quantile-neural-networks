import pandas as pd
import numpy as np
import os
import seaborn
from tqdm import tqdm
import matplotlib.pyplot as plt

from EstimationFunctions.SimulationFunctions import *
from EstimationFunctions.NN_Functions import (
    train_loop,
    mean_quantile_loss,
    ToLaTeX,
)

# path to data
sPath = "./Data"

# define variables to be used generally
taus = (
    [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.075]
    + [0.925, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99995]
    + [i / 20 for i in range(1, 20)]
)
taus.sort()

# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")

# set seed
np.random.seed(123)

## setings
# -> simulate 60 years of data for 5000 stocks plus one year of burnin
# n = [500, 5000]
# T = [240, 480]
# eval on 30 years of data
# 22/264 days
# -> simulate 60 years of data for 5000 stocks
# -> do 1 year of burn in
n_year = 61
days_m = 22
stock_n = 7500
sample_l = n_year * days_m * 12
burnin = days_m * 12
P_jump = 0.01
std_jump = 0.25
df_jump = 3

# variables for ML
vol_vars = (
    [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMAVolD{i}" for i in [20, 10, 6]]
    + ["TV3M", "TV6M", "TV12M"]
)
signal_vars = ["BAB", "Beta", "IdioRisk", "Max", "TV"]
mkt_mean_vars = ["MktAvg10_EW", "MktAvg6_EW", "MktAvg4_EW", "MktAvg1_EW", "MktAvg0.1_EW"]


##### get distribution of data
## distribution for beta
betas = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals.gzip"), columns=["Beta", "DTID", "date"])
betas = betas.loc[(betas["Beta"] < 4) & (betas["Beta"] > -0.5)].copy()
betas["Beta"].hist(bins=100)
betas["Beta"].std()

## look at distribution of GARCH parameters estimated on the actual data
res = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
res = res.loc[res["Spec"] == "GJRGARCH t"].copy()
res = res.loc[res["Converged"] & res["Converged2"]].copy()
res = res.loc[(res["omega"] > res["omega"].quantile(0.05)) & (res["omega"] < res["omega"].quantile(0.95))].copy()
res = res.loc[
    (res["alpha[1]"] > 0.01)
    & (res["beta[1]"] > 0.01)
    & (res["alpha[1]"] + res["beta[1]"] > 0.5)
    & (res["nu"] < 20)
].copy()
res["omega"].hist(bins=100)
res["alpha[1]"].hist(bins=100)
res["beta[1]"].hist(bins=100)
(res["alpha[1]"] + res["beta[1]"]).hist(bins=100)
res["nu"].hist(bins=100)
res = res.merge(betas, on=["DTID", "date"], how="inner")
res = res[["Beta", "omega", "alpha[1]", "beta[1]", "gamma[1]", "nu", "vol"]].copy()
res.rename({"Beta": "MktBeta", "alpha[1]": "alpha", "beta[1]": "beta", "gamma[1]": "gamma"}, axis=1, inplace=True)
res.to_parquet(os.path.join(sPath, "Simulation", "StartingDist.gzip"), compression="gzip")


##### Simulate returns
## simulate market returns
ret_, vola_, inov_ = simulate_r_t(omega=0.0025, alpha=0.06, beta=0.94, vol_init=10, nu=5, h=sample_l)
mkt_ret = pd.DataFrame(
    {"date": range(sample_l - burnin), "MktRet": ret_[burnin:], "vola": vola_[burnin:], "inov": inov_[burnin:]}
)
mkt_ret["MktRet"].std()
# SP500 is about 0.15 / np.sqrt(252)
mkt_ret["MktRet"].hist(bins=100)

## simulate individual stock returns
res = pd.read_parquet(os.path.join(sPath, "Simulation", "StartingDist.gzip"))
res = res.iloc[np.random.randint(0, len(res), stock_n)].copy()
res["DTID"] = range(len(res))
res.reset_index(drop=True, inplace=True)
ret = []
for i, row in tqdm(res.iterrows()):
    ret_, vola_, inov_ = simulate_gjr_r_t(
        omega=row["omega"],
        alpha=row["alpha"],
        beta=row["beta"],
        gamma=row["gamma"],
        vol_init=row["vol"] ** 2,
        nu=row["nu"],
        h=sample_l,
    )
    ret_ = pd.DataFrame(
        {
            "date": range(sample_l - burnin),
            "r": ret_[burnin:],
            "vola": vola_[burnin:],
            "inov": inov_[burnin:],
            "DTID": row["DTID"],
        }
    )
    df_ = ret_.merge(mkt_ret[["date", "MktRet"]], how="inner", on="date")
    df_["jump"] = simulate_jump_t(nu=df_jump, P=P_jump, std=std_jump, h=len(df_))
    df_["r"] = df_["r"] + df_["MktRet"] * row["MktBeta"] + df_["jump"]
    df_.loc[df_["r"] < -0.9, "r"] = -0.9
    df_.loc[df_["r"] > 10, "r"] = 10
    del df_["MktRet"]
    ret += [df_]
ret = pd.concat(ret)
ret["DTID"] = ret["DTID"].astype(int)
ret.loc[ret["r"] < 1, "r"].hist(bins=100)
# save for simulation
ret.to_parquet(os.path.join(sPath, "Simulation", "Ret.gzip"), compression="gzip")
res.to_parquet(os.path.join(sPath, "Simulation", "Dist.gzip"), compression="gzip")
mkt_ret.to_parquet(os.path.join(sPath, "Simulation", "MktRet.gzip"), compression="gzip")

##### simulate actual quantiles
ret = pd.read_parquet(os.path.join(sPath, "Simulation", "Ret.gzip"))
dist = pd.read_parquet(os.path.join(sPath, "Simulation", "Dist.gzip"))
mkt_ret = pd.read_parquet(os.path.join(sPath, "Simulation", "MktRet.gzip"))
month_start_dates = np.arange(days_m * 12 * 30 + 1, days_m * 12 * 60 - days_m + 2, days_m)
res = []
for i in tqdm(month_start_dates):
    # simulate market return
    mkt_ret_init = mkt_ret.loc[mkt_ret["date"] == i - 1]
    rm = vol_bootstrap_worker_t(
        omega=0.0025,
        alpha=0.06,
        beta=0.94,
        vol_init=mkt_ret_init["vola"].values[0],
        inov_init=mkt_ret_init["inov"].values[0],
        nu=5,
        h=days_m,
    )

    # simulate idiosyncratic return
    ret_init = ret.loc[ret["date"] == i - 1]
    for _, row in dist.iterrows():
        omega = row["omega"]
        alpha = row["alpha"]
        beta = row["beta"]
        gamma = row["gamma"]
        nu = row["nu"]
        MktBeta = row["MktBeta"]
        ret_init_i = ret_init.loc[ret_init["DTID"] == row["DTID"]]
        vol_init = ret_init_i["vola"].values[0] ** 2
        inov_init = ret_init_i["inov"].values[0]
        r_i = vol_bootstrap_worker_gjr_t(omega, alpha, beta, gamma, vol_init, inov_init, nu, h=days_m)
        r_i += rm * MktBeta
        jumps = simulate_jump_t(nu=df_jump, P=P_jump, std=std_jump, h=r_i.shape[0] * r_i.shape[1]).reshape(
            r_i.shape
        )
        r_i += jumps
        r_i[r_i < -0.9] = -0.9
        r_i[r_i > 10] = 10
        r = (1 + r_i).prod(axis=0) - 1
        stat = np.append(np.quantile(r, taus), np.std(r))
        res_ = pd.DataFrame(
            {"date": i, "DTID": row["DTID"], "vol": stat[-1]} | {f"Q{tau}": stat[i] for i, tau in enumerate(taus)},
            index=[0],
        )
        for col_ in [f"Q{tau}" for tau in taus]:
            res_.loc[res_[col_] < -1, col_] = -1
            res_.loc[res_[col_] > 20, col_] = 20
        res += [res_]
res = pd.concat(res)
res.to_parquet(os.path.join(sPath, "Simulation", "Q_actual.gzip"), compression="gzip")


##### Estimate GARCH baseline with 3 years of past returns
ret = pd.read_parquet(os.path.join(sPath, "Simulation", "Ret.gzip"))
month_start_dates = np.arange(days_m * 12 * 30 + 1, days_m * 12 * 60 - days_m + 2, days_m)
res = []
for i in tqdm(month_start_dates):
    # simulate market return
    dt = ret.loc[(ret["date"] < i) & (ret["date"] >= i - days_m * 12 * 3)].copy()
    dt = ret.loc[(ret["date"] < i)].copy()
    dt["r"] = dt["r"] * 100
    dist = dt.groupby("DTID")["r"].apply(lambda x: GARCH(x.values, p=1, q=1, dist="t", EstimateMean=False))
    dist = dist.reset_index(drop=True, level=1)

    # simulate idiosyncratic return
    for DTID, row in dist.iterrows():
        omega = row["omega"]
        alpha = row["alpha[1]"]
        beta = row["beta[1]"]
        nu = row["nu"]
        vol_init = row["vol"] ** 2
        inov_init = row["inov"]
        r_i = vol_bootstrap_worker_t(omega, alpha, beta, vol_init, inov_init, nu, h=days_m)
        r_i[r_i < -1] = -1
        r = (1 + r_i).prod(axis=0) - 1
        stat = np.append(np.quantile(r, taus), np.std(r))
        res_ = pd.DataFrame(
            {"date": i, "DTID": DTID, "vol": stat[-1]} | {f"Q{tau}": stat[i] for i, tau in enumerate(taus)},
            index=[0],
        )
        for col_ in [f"Q{tau}" for tau in taus]:
            res_.loc[res_[col_] < -1, col_] = -1
            res_.loc[res_[col_] > 20, col_] = 20
        res += [res_]
res = pd.concat(res)
res.to_parquet(os.path.join(sPath, "Simulation", "Q_baseline.gzip"), compression="gzip")


##### ML data preparation
# Simulation/dt.gzip
ret = pd.read_parquet(os.path.join(sPath, "Simulation", "Ret.gzip"))
ret["MC"] = 1
ret["r_1"] = ret["r"] + 1
ret["RI"] = ret.groupby("DTID")["r_1"].cumprod()
ret = ret[["date", "DTID", "RI", "r", "MC"]]
ret = ret.reset_index(drop=True)
ret.to_parquet(os.path.join(sPath, "Simulation", "dt.gzip"), compression="gzip")

## monthly sampling
# generate signals
Signals = CreateSignals(
    DateSequence=np.arange(days_m * 12 * 5 + 1, days_m * 12 * 60 - days_m + 2, days_m),
    sPath=sPath,
    Signals=vol_vars + signal_vars,
)
Signals.to_parquet(os.path.join(sPath, "Simulation", "Signals.gzip"), compression="gzip")

# generate returns
Ret = CreateReturns(
    DateSequenceStart=np.arange(days_m * 12 * 5 + 1, days_m * 12 * 60 - days_m + 2, days_m),
    DateSequenceEnd=np.arange(days_m * 12 * 5 + days_m + 1, days_m * 12 * 60 + 2, days_m),
    sPath=sPath,
)
Ret.to_parquet(os.path.join(sPath, "Simulation", "Returns.gzip"), compression="gzip")

# join together
Signals = pd.read_parquet(os.path.join(sPath, "Simulation", "Signals.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Simulation", "Returns.gzip"))
data = MLdata(Ret, Signals, exc_normalize=vol_vars)
data.to_parquet(os.path.join(sPath, "Simulation", "MLdata.gzip"), compression="gzip")

## equally-weighted market returns with different moving average filter
ret = MktMeanRet(sPath)
ret.to_parquet(os.path.join(sPath, "Simulation", "Mkt_mean.gzip"), compression="gzip")


##### NN estimation
# settings
sample_split = pd.DataFrame(
    {
        "train_start": days_m * 12 * 5,  # 5 years burnin for estimation
        "train_end": days_m * 12 * 30 - days_m,  # use the first 30 years minus last month
        "valid_start": days_m * 12 * 30,  # validation is the last 30 years
        "valid_end": days_m * 12 * 60,
    },
    index=[0],
)
inputs1 = signal_vars + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [128, 128, 4, 128, 128],
    "hidden_sizes2": [8],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 10,
}
# get estimation data
data_m_full = get_data(sPath, "MLdata.gzip", vol_vars, mkt_mean_vars)

# get data for comparison
Ret = pd.read_parquet(os.path.join(sPath, "Simulation", "Returns.gzip"))
actual = pd.read_parquet(os.path.join(sPath, "Simulation", "Q_actual.gzip"))
actual.rename({"vol": "vol_oos"}, axis=1, inplace=True)
baseline = pd.read_parquet(os.path.join(sPath, "Simulation", "Q_baseline.gzip"))
baseline.rename({"vol": "vol_GARCH"}, axis=1, inplace=True)
Q_cols = [f"Q{tau}" for tau in taus]
actual = actual.loc[actual["vol_oos"] < 1].copy()
act = actual.melt(id_vars=["DTID", "date"], value_vars=Q_cols, value_name="Actual")
base = baseline.melt(id_vars=["DTID", "date"], value_vars=Q_cols, value_name="GARCH")
res_full = act.merge(base, on=["DTID", "date", "variable"])
actual = actual.merge(Ret, on=["date", "DTID"])
baseline = baseline.merge(Ret, on=["date", "DTID"])
baseline = baseline.merge(actual[["date", "DTID"]], on=["date", "DTID"], how="inner")  # discard outliers in actual

## estimate for different sample size
n_runs = 20
DTID = data_m_full["DTID"].unique()
result = []
Q_loss = []
for n_stock in [100, 500, 1000, 2500, 5000]:
    for i in range(n_runs):
        print(f"Running for {n_stock} stocks, run {i}")
        DTID_sel = np.random.choice(DTID, n_stock, replace=False)
        data_m = data_m_full.loc[data_m_full["DTID"].isin(DTID_sel)].copy()
        _, pred_m = train_loop(data_m, data_m, sample_split, param, inputs1, inputs2)
        pred_m.rename({f"pred_raw_{tau}": f"Q{tau}" for tau in taus}, inplace=True, axis=1)

        # get average quantile loss based on monthly return
        act = actual.loc[actual["DTID"].isin(DTID_sel)]
        base = baseline.loc[baseline["DTID"].isin(DTID_sel)]
        pred_m = pred_m.merge(
            actual[["date", "DTID"]], on=["date", "DTID"], how="inner"
        )  # discard outliers in actual
        Q_loss_ = pd.DataFrame(
            {
                "n_stock": n_stock,
                "run": i,
                "Q": taus,
                "NN": [mean_quantile_loss(pred_m["r_raw"], pred_m[f"Q{tau}"], alpha=tau) for tau in taus],
                "Actual": [mean_quantile_loss(act["r"], act[f"Q{tau}"], alpha=tau) for tau in taus],
                "GARCH": [mean_quantile_loss(base["r"], base[f"Q{tau}"], alpha=tau) for tau in taus],
            }
        )
        Q_loss += [Q_loss_]

        # get losses based on actual quantile
        pred_m = pred_m.melt(id_vars=["DTID", "date"], value_vars=Q_cols, value_name="NN")
        res = res_full.merge(pred_m, on=["DTID", "date", "variable"], how="inner")
        result_ = pd.DataFrame(
            {
                "n_stock": n_stock,
                "run": i,
                "GARCH RMSE": RMSE_df_sim(res, "GARCH", "Actual", GrpVar="variable"),
                "NN RMSE": RMSE_df_sim(res, "NN", "Actual", GrpVar="variable"),
                "Std Actual": res.groupby("variable")["Actual"].std(),
                "Mean Actual": res.groupby("variable")["Actual"].mean(),
                "Mean GARCH": res.groupby("variable")["GARCH"].mean(),
                "Mean NN": res.groupby("variable")["NN"].mean(),
            }
        )
        result += [result_]
result = pd.concat(result)
Q_loss = pd.concat(Q_loss)
result.to_parquet(os.path.join(sPath, "Simulation", "NN_Q_RMSE.gzip"), compression="gzip")
Q_loss.to_parquet(os.path.join(sPath, "Simulation", "NN_Q_loss.gzip"), compression="gzip")

## create table
result = pd.read_parquet(os.path.join(sPath, "Simulation", "NN_Q_RMSE.gzip"))
result = result.loc[result["n_stock"] <= 2500].copy()
result = result.reset_index().rename({"variable": "Q"}, axis=1)
result.loc[result["Q"] == "Q5e-05", "Q"] = "Q0.00005"
result["Q"] = result["Q"].str.slice(1).astype(float)
res_m = result.groupby(["Q", "n_stock"])[["NN RMSE"]].mean().reset_index()
res_add = (
    result.loc[result["n_stock"] == 2500]
    .groupby(["Q"])[["Mean Actual", "Std Actual", "GARCH RMSE"]]
    .mean()
    .reset_index()
)
res_add_std = (
    result.loc[result["n_stock"] == 2500].groupby(["Q"])[["Std Actual", "GARCH RMSE"]].std().reset_index()
)
res_add_std.rename({"Std Actual": "Actual_s", "GARCH RMSE": "GARCH_s"}, axis=1, inplace=True)
res_add = res_add.merge(res_add_std, on="Q")
res_add["Actuals"] = (100 * res_add["Std Actual"]).transform(lambda x: f"{x:0.2f}") + (
    100 * res_add["Actual_s"]
).transform(lambda x: f" ({x:0.2f})")
res_add["GARCH"] = res_add["GARCH RMSE"].transform(lambda x: f"{100 * x:0.2f}") + res_add["GARCH_s"].transform(
    lambda x: f" ({100 * x:0.2f})"
)
res_add["Mean Actual"] = res_add["Mean Actual"].transform(lambda x: f"{100 * x:0.2f}")
res_add["Empty"] = ""
res_add["Empty2"] = ""
res_add["Empty3"] = ""
res_add = res_add[["Q", "Mean Actual", "Empty", "Actuals", "Empty2", "GARCH", "Empty3"]]
res_std = result.groupby(["Q", "n_stock"])["NN RMSE"].std().reset_index().rename({"NN RMSE": "std"}, axis=1)
res = res_m[["Q", "n_stock", "NN RMSE"]].rename({"NN RMSE": "mean"}, axis=1).merge(res_std, on=["Q", "n_stock"])
res["NN"] = (100 * res["mean"]).transform(lambda x: f"{x:0.2f}") + (100 * res["std"]).transform(
    lambda x: f" ({x:0.2f})"
)
res = res.set_index(["Q", "n_stock"])["NN"].unstack(-1).reset_index()
res = res_add.merge(res, on="Q")
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "Simulations_RMSE.txt"), index=False)


## histogram plots
dt = pd.read_parquet(os.path.join(sPath, "Simulation", "StartingDist.gzip"))
dt.rename({"omega": "Omega", "alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "nu": "Nu"}, axis=1, inplace=True)
fig = plt.figure(figsize=(15, 9))
for i, Var in enumerate(["MktBeta", "Omega", "Alpha", "Beta", "Gamma", "Nu"]):
    ax = plt.subplot(2, 3, i + 1)
    dt[Var].hist(density=True, bins=100, ax=ax, label=Var, alpha=0.7)
    ax.set_ylabel("Density")
    ax.set_xlabel("")
    ax.set_title(Var)
fig.tight_layout()
fig.savefig(os.path.join(sPath, "Output", "Simulation_GARCH_param_hist.png"))
plt.close()
