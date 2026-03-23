import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from EstimationFunctions.NN_Functions import *
from DataModules.ProcessData import *
from Signals.CreateSignals import *
from EstimationFunctions.GARCH_Functions import *


# path to data
sPath = "./Data"

# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")
np.random.seed(123)


### fetch daily returns with the whole history
# DataSource = "WRDS"
# DataSource = "DST"
for DataSource in ["WRDS", "DST"]:
    if DataSource == "WRDS":
        DM = DataManager(
            sPath,
            Date="2025-01-01",
            source="WRDS",
            Inputs={"dt": {"start": -1000, "end": 0, "items": ["r"]}},
        )
    elif DataSource == "DST":
        DM = DataManager(
            sPath,
            Date="2025-01-01",
            source="DST",
            Inputs={"dt": {"start": -1000, "end": 0, "items": ["r"]}},
        )
    FetchedData = DM.fetch({"dt": {"start": -1000, "end": 0, "items": ["r"]}})
    dt = FetchedData["dt"].copy()
    dt = dt.dropna(subset="r")
    dt["r"] = dt["r"] * 100
    dt["r2"] = dt["r"] ** 2
    dt.loc[dt["r2"] > 2500, "r2"] = 2500
    dt.reset_index(inplace=True)

    # get risk-free rate
    rf = pd.read_csv(os.path.join(sPath, "Inputs", "FF3.CSV"))
    rf["date"] = rf["date"].astype(str)
    rf["date"] = pd.to_datetime(rf["date"].str.slice(0, 4) + "-" + rf["date"].str.slice(4, 6) + "-01")
    rf["RF"] = rf["RF"] / 21
    rf = rf[["date", "RF"]].copy()

    # filter DTIDs in the given month
    pred_dates = pd.read_parquet(
        os.path.join(sPath, "Features", "MLdata_full.gzip"), columns=["date", "DTID"]
    ).drop_duplicates()

    result = []
    for date in tqdm(TimeSequence(start="1995-01-01", end="2023-12-01")):
        dt_ = dt.loc[(dt["date"] >= pd.to_datetime(date) + relativedelta(months=-36)) & (dt["date"] < date)].copy()
        dt_ = dt_.loc[dt_["DTID"].isin(pred_dates.loc[pred_dates["date"] == date, "DTID"])].copy()

        if len(dt_) == 0:
            break

        if DataSource == "WRDS":
            # sGARCH with estimated mean
            res = dt_.groupby("DTID")["r"].apply(lambda x: GARCH(x.values, p=1, q=1, dist="t"))
            res = res.reset_index(drop=True, level=1)
            res["Spec"] = "GARCH mu t"
            res["date"] = date
            res["dist"] = "t"
            result += [res.copy()]

            # sGARCH with const mean
            dt_2 = dt_.copy()
            dt_2["r"] = dt_2["r"] - 10 / 252
            res = dt_2.groupby("DTID")["r"].apply(
                lambda x: GARCH(x.values, p=1, q=1, dist="t", EstimateMean=False)
            )
            res = res.reset_index(drop=True, level=1)
            res["mu"] = 10 / 252
            res["Spec"] = "GARCH 10mu t"
            res["date"] = date
            res["dist"] = "t"
            result += [res.copy()]

        # the rest is with mean based on risk free rate
        mu = rf.loc[rf["date"] == date, "RF"].values[0] + 5 / 252
        if np.isnan(mu):
            mu = 10 / 252
        dt_ = dt_.copy()
        dt_["r"] = dt_["r"] - mu

        # EWMA RiskMetrics model
        x = dt_.groupby("DTID")["r2"].ewm(alpha=0.06, adjust=False, min_periods=1).mean()
        x = np.sqrt(x.reset_index(drop=True, level=1)).groupby("DTID").last()
        result += [
            pd.DataFrame(
                {
                    "vol": x,
                    "omega": 0,
                    "Spec": "EWMAVol",
                    "mu": mu,
                    "date": date,
                    "alpha[1]": 0.06,
                    "beta[1]": 0.94,
                    "dist": "t",
                    "nu": 4,
                }
            )
        ]

        # with t dist
        res = dt_.groupby("DTID")["r"].apply(lambda x: GARCH(x.values, p=1, q=1, dist="t", EstimateMean=False))
        res = res.reset_index(drop=True, level=1)
        res["Spec"] = "GARCH t"
        res["mu"] = mu
        res["date"] = date
        res["dist"] = "t"
        result += [res.copy()]

        if DataSource == "WRDS":
            # with normal dist
            res = dt_.groupby("DTID")["r"].apply(
                lambda x: GARCH(x.values, p=1, q=1, dist="norm", EstimateMean=False)
            )
            res = res.reset_index(drop=True, level=1)
            res["Spec"] = "GARCH normal"
            res["mu"] = mu
            res["date"] = date
            res["dist"] = "normal"
            result += [res.copy()]

            # eGARCH
            res = dt_.groupby("DTID")["r"].apply(
                lambda x: GARCH(x.values, p=1, q=1, dist="t", ModelName="EGARCH", EstimateMean=False)
            )
            res = res.reset_index(drop=True, level=1)
            res["Spec"] = "EGARCH t"
            res["mu"] = mu
            res["date"] = date
            res["dist"] = "t"
            result += [res.copy()]

            # GJR GARCH
            res = dt_.groupby("DTID")["r"].apply(
                lambda x: GARCH(x.values, p=1, q=1, dist="t", ModelName="GJRGARCH", EstimateMean=False)
            )
            res = res.reset_index(drop=True, level=1)
            res["Spec"] = "GJRGARCH t"
            res["mu"] = mu
            res["date"] = date
            res["dist"] = "t"
            result += [res.copy()]

            # sGARCH order 2
            res = dt_.groupby("DTID")["r"].apply(lambda x: GARCH(x.values, p=2, q=2, dist="t", EstimateMean=False))
            res = res.reset_index(drop=True, level=1)
            res["Spec"] = "GARCH2 10mu t"
            res["mu"] = mu
            res["date"] = date
            res["dist"] = "t"
            result += [res.copy()]

    result = pd.concat(result)
    result.reset_index(inplace=True)
    result["date"] = pd.to_datetime(result["date"])

    if DataSource == "WRDS":
        result.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"), compression="gzip")
    elif DataSource == "DST":
        result.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST.gzip"), compression="gzip")

### run bootstrap
result = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
res1 = (
    result.loc[result["Spec"] == "GARCH normal"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap)
)
res1["Spec"] = "bootstrap_GARCH_normal"
res2 = (
    result.loc[result["Spec"] == "GARCH mu t"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
)
res2["Spec"] = "bootstrap_GARCH_mu_t"
res3 = (
    result.loc[result["Spec"] == "GARCH 10mu t"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
)
res3["Spec"] = "bootstrap_GARCH_10mu_t"
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 5
res4 = result_.loc[result_["Spec"] == "EWMAVol"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res4["Spec"] = "bootstrap_RISKMETRICS_t5"
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 4
res5 = result_.loc[result_["Spec"] == "EWMAVol"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res5["Spec"] = "bootstrap_RISKMETRICS_t4"
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 3
res6 = result_.loc[result_["Spec"] == "EWMAVol"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res6["Spec"] = "bootstrap_RISKMETRICS_t3"
result_ = result.loc[result["Spec"] == "GARCH2 10mu t"].copy()
res7 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap2_t)
res7["Spec"] = "bootstrap_GARCH2_t"
result_ = result.loc[result["Spec"] == "EGARCH t"].copy()
res8 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_egarch_t)
res8["Spec"] = "bootstrap_EGARCH_t"
result_ = result.loc[result["Spec"] == "GJRGARCH t"].copy()
res9 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_gjr_t)
res9["Spec"] = "bootstrap_GJRGARCH_t"
res10 = result.loc[result["Spec"] == "GARCH t"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res10["Spec"] = "bootstrap_GARCH_t"
res = pd.concat([res1, res2, res3, res4, res5, res6, res7, res8, res9, res10])
res.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q.gzip"), compression="gzip")

## international sample
result = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST.gzip"))
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 4
res1 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res1["Spec"] = "bootstrap_RISKMETRICS_t4"
res2 = result.loc[result["Spec"] == "GARCH t"].groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t)
res2["Spec"] = "bootstrap_GARCH_t"
res = pd.concat([res1, res2])
res.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST_Q.gzip"), compression="gzip")

## connect together internaltional and US
res = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q.gzip"))
res_sub = res.loc[res["Spec"] == "bootstrap_RISKMETRICS_t4"].copy()
res = res.loc[res["Spec"] == "bootstrap_GARCH_t"].copy()
res_ = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST_Q.gzip"))
res_sub_ = res_.loc[res_["Spec"] == "bootstrap_RISKMETRICS_t4"].copy()
res_ = res_.loc[res_["Spec"] == "bootstrap_GARCH_t"].copy()
res = pd.concat([res, res_])
res_sub = pd.concat([res_sub, res_sub_])

# substitute where there was problematic covergence during estimation
fit = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
fit_ = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST.gzip"))
fit = pd.concat([fit, fit_])
fit = fit.loc[(fit["Spec"] == "GARCH t")].copy()
fit = fit.loc[~fit["Converged2"].astype(bool) | ~fit["Converged"].astype(bool)].copy()
fit = fit[["DTID", "date", "Converged"]].copy()
fit["date"] = pd.to_datetime(fit["date"])
res = res.merge(fit, on=["DTID", "date"], how="left")
res = res.loc[res["Converged"].isnull()].copy()
res_sub = res_sub.merge(fit, on=["DTID", "date"], how="inner")
res = pd.concat([res, res_sub])

# truncate large predictions
for col_ in [f"Q{tau}" for tau in taus]:
    res.loc[res[col_] < -1, col_] = -1
    res.loc[res[col_] > 50, col_] = 50
res.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_GARCH_t.gzip"), compression="gzip")


###### scoring rules
### run bootstrap
result = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
data = pd.read_parquet(os.path.join(sPath, "Features", "MLdata_full.gzip"), columns=["date", "DTID", "r"])
result = result.merge(data, on=["date", "DTID"], how="left")
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 4
res1 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t_scoring)
res1["Spec"] = "bootstrap_RISKMETRICS_t4"
res2 = (
    result.loc[result["Spec"] == "GARCH t"]
    .groupby(["DTID", "date"], group_keys=False)
    .apply(vol_bootstrap_t_scoring)
)
res2["Spec"] = "bootstrap_GARCH_t"
res = pd.concat([res1, res2])
res.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_scoring.gzip"), compression="gzip")

## international sample
result = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST.gzip"))
result = result.merge(data, on=["date", "DTID"], how="left")
result_ = result.loc[result["Spec"] == "EWMAVol"].copy()
result_["nu"] = 4
res1 = result_.groupby(["DTID", "date"], group_keys=False).apply(vol_bootstrap_t_scoring)
res1["Spec"] = "bootstrap_RISKMETRICS_t4"
res2 = (
    result.loc[result["Spec"] == "GARCH t"]
    .groupby(["DTID", "date"], group_keys=False)
    .apply(vol_bootstrap_t_scoring)
)
res2["Spec"] = "bootstrap_GARCH_t"
res = pd.concat([res1, res2])
res.to_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST_Q_scoring.gzip"), compression="gzip")

## connect together internaltional and US
res = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_scoring.gzip"))
res_sub = res.loc[res["Spec"] == "bootstrap_RISKMETRICS_t4"].copy()
res = res.loc[res["Spec"] == "bootstrap_GARCH_t"].copy()
res_ = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST_Q_scoring.gzip"))
res_sub_ = res_.loc[res_["Spec"] == "bootstrap_RISKMETRICS_t4"].copy()
res_ = res_.loc[res_["Spec"] == "bootstrap_GARCH_t"].copy()
res = pd.concat([res, res_])
res_sub = pd.concat([res_sub, res_sub_])

# substitute where there was problematic covergence during estimation
fit = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
fit_ = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_DST.gzip"))
fit = pd.concat([fit, fit_])
fit = fit.loc[(fit["Spec"] == "GARCH t")].copy()
fit = fit.loc[~fit["Converged2"].astype(bool) | ~fit["Converged"].astype(bool)].copy()
fit = fit[["DTID", "date", "Converged"]].copy()
fit["date"] = pd.to_datetime(fit["date"])
res = res.merge(fit, on=["DTID", "date"], how="left")
res = res.loc[res["Converged"].isnull()].copy()
res_sub = res_sub.merge(fit, on=["DTID", "date"], how="inner")
res = pd.concat([res, res_sub])
res.to_parquet(os.path.join(sPath, "Predict", "ScoringGARCH.gzip"), compression="gzip")
