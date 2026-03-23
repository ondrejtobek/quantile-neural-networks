import pandas as pd
import numpy as np
import os

from lightgbm import LGBMRegressor, reset_parameter
from tqdm import tqdm
from EstimationFunctions.NN_Functions import *

# path to data
sPath = "./Data"

# file names
files = {
    "W_file_22d": "MLdata_W_22d.gzip",
    "W_file_22d_full": "MLdata_W_22d_full.gzip",
    "M_file": "MLdata.gzip",
    "M_file_full": "MLdata_full.gzip",
}


# define variables to be used generally
anomalies = get_anomalies_list(sPath)
vol_vars = (
    [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMARange{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMAVolD{i}" for i in [20, 10, 6]]
    + ["TV3M", "TV6M", "TV12M"]
)
mkt_mean_vars = ["MktAvg10_EW", "MktAvg6_EW", "MktAvg4_EW", "MktAvg1_EW", "MktAvg0.1_EW"]
taus = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
taus.sort()


def train_loop_RF(
    data,
    data_m,
    sample_split,
    param,
    inputs1,
    output=["r_raw"],
    taus=taus,
):
    input_vars = inputs1
    extra_cols = [i for i in ["r_raw", "r_scale"] if i in data.columns]
    preds_m = []
    for i in tqdm(range(len(sample_split))):
        split = sample_split.iloc[i].to_dict()

        # training sample
        train_df = data.loc[(data["date"] >= split["train_start"]) & (data["date"] <= split["train_end"]), :]

        # predict sample monthly
        test_df_m = data_m.loc[
            (data_m["date"] >= split["valid_start"]) & (data_m["date"] <= split["valid_end"]), :
        ].reset_index(drop=True)

        pred_cols = []
        for tau in taus:
            param["alpha"] = tau
            # estimate model
            model = LGBMRegressor(**param)
            if param["boosting_type"] == "gbdt":
                model.fit(
                    train_df[input_vars],
                    train_df[output],
                    callbacks=[reset_parameter(learning_rate=lambda iter: 0.25 if iter < 50 else 0.01)],
                )
            else:
                model.fit(train_df[input_vars], train_df[output])

            # predict
            pred_col = "Q" + str(param["alpha"])
            test_df_m[pred_col] = model.predict(test_df_m[input_vars])
            pred_cols += [pred_col]
        preds_m += [test_df_m[["DTID", "date"] + extra_cols + pred_cols]]

    if len(preds_m) > 0:
        preds_m = pd.concat(preds_m).reset_index(drop=True)

    return preds_m


sample_split = validation_logic()
inputs1 = anomalies + [i + "_raw" for i in vol_vars] + mkt_mean_vars


### full sample
data = get_data(
    sPath,
    files["W_file_22d_full"],
    vol_vars,
    mkt_mean_vars,
    regions=["USA"],
    rescale_mean=False,
)
data_m = get_data(sPath, files["M_file_full"], vol_vars, mkt_mean_vars, rescale_mean=False)

## GBRT
param = {
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": 16,
    "objective": "quantile",
    "alpha": "0.5",
    "verbosity": -1,
}
pred_m = train_loop_RF(data, data_m, sample_split, param, inputs1, output="r_raw", taus=taus)
pred_m.to_parquet(os.path.join(sPath, "Predict", "GBRT1_full_m.gzip"), compression="gzip")
del pred_m

## RF
param = {
    "boosting_type": "rf",
    "n_estimators": 100,
    "num_leaves": 20,
    "bagging_fraction": 0.1,
    "bagging_freq": 5,
    "n_jobs": 16,
    "objective": "quantile",
    "alpha": "0.5",
    "verbosity": -1,
}
pred_m = train_loop_RF(data, data_m, sample_split, param, inputs1, output="r_raw", taus=taus)
pred_m.to_parquet(os.path.join(sPath, "Predict", "RF1_full_m.gzip"), compression="gzip")
del data, data_m, pred_m

### liquid
data = get_data(sPath, files["W_file_22d"], vol_vars, mkt_mean_vars, regions=["USA"], rescale_mean=False)
data_m = get_data(sPath, files["M_file"], vol_vars, mkt_mean_vars, rescale_mean=False)

## GBRT
param = {
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": 16,
    "objective": "quantile",
    "alpha": "0.5",
    "verbosity": -1,
}
pred_m = train_loop_RF(data, data_m, sample_split, param, inputs1, output="r_raw", taus=taus)
pred_m.to_parquet(os.path.join(sPath, "Predict", "GBRT1_m.gzip"), compression="gzip")

## RF
param = {
    "boosting_type": "rf",
    "n_estimators": 100,
    "num_leaves": 20,
    "bagging_fraction": 0.1,
    "bagging_freq": 5,
    "n_jobs": 16,
    "objective": "quantile",
    "alpha": "0.5",
    "verbosity": -1,
}
pred_m = train_loop_RF(data, data_m, sample_split, param, inputs1, output="r_raw", taus=taus)
pred_m.to_parquet(os.path.join(sPath, "Predict", "RF1_m.gzip"), compression="gzip")


### quantile loss function fit
Settings = {
    "GARCH": "VolaGARCH_Q_GARCH_t.gzip",
    "Two Stage": "NN_clean_M_full_m.gzip",
    "Linear": "NN_clean_onestage_NN0_full_M_m.gzip",
    "RF": "RF1_full_m.gzip",
    "GBRT": "GBRT1_full_m.gzip",
}
res = []
for Type in ["Full", "Liquid"]:
    for Spec, File in Settings.items():
        if Spec == "GARCH":
            pred2 = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_GARCH_t.gzip"))
            if Type == "Liquid":
                pred = GetPredictions(sPath, files, pred_file="NN_clean_M_m.gzip", keep_r=True)
            else:
                pred = GetPredictions(sPath, files, pred_file="NN_clean_M_full_m.gzip", keep_r=True)
            pred = pred.merge(pred2, on=["date", "DTID"], how="inner")
        else:
            if Type == "Liquid":
                File = File.replace("_full", "")
            pred = GetPredictions(sPath, files, pred_file=File, keep_r=True)
        if "Q0.5" in pred.columns:
            col = "Q"
        elif "pred_raw_0.5" in pred.columns:
            col = "pred_raw_"
        else:
            col = "pred_"
        pred = pred.loc[pred["date"] <= "2018-12-31"].copy()
        for reg in ["USA", "Europe", "Japan", "Asia Pacific"]:
            pred_ = pred.loc[pred["region"] == reg]
            loss = pred_.groupby("date").apply(
                lambda x: np.mean([mean_quantile_loss(x["r_raw"], x[f"{col}{tau}"], alpha=tau) for tau in taus])
            )
            res += [pd.DataFrame({"Spec": Spec, "region": reg, "type": Type, "loss": loss}).reset_index()]
res = pd.concat(res)
res_bench = res.loc[res["Spec"] == "Two Stage"].copy()
del res_bench["Spec"]
res_bench.rename({"loss": "loss_bench"}, axis=1, inplace=True)
res = res.merge(res_bench, on=["date", "region", "type"])
res["diff"] = res["loss"] - res["loss_bench"]
Agg = res.groupby(["type", "Spec", "region"])["diff"].agg(["mean", "count"])
Agg["Avg Loss"] = res.groupby(["type", "Spec", "region"])["loss"].mean()
Agg["Error"] = res.groupby(["type", "Spec", "region"])["diff"].apply(lambda x: NW_std(x.values))
Agg["t-stat"] = Agg["mean"] / Agg["Error"] * np.sqrt(Agg["count"])
Agg = Agg[["Avg Loss", "t-stat"]].copy()
Agg["Avg Loss"] = Agg["Avg Loss"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.3f}")
Agg["t-stat"] = Agg["t-stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
Agg = Agg.reset_index().melt(id_vars=["type", "Spec", "region"]).set_index(["type", "region", "Spec", "variable"])
Agg_ = Agg["value"].unstack([0, 1]).reset_index()
Agg_.loc[Agg_["variable"] == "t-stat", "Spec"] = " "
Agg_["variable"] = Agg_["variable"].map({"Avg Loss": r"$L_{avg}$", "t-stat": "(t-stat)"})
regions = ["USA", "Europe", "Japan", "Asia Pacific"]
Agg_["Empty"] = ""
Agg_ = pd.concat(
    [Agg_[["Spec", "variable"]], Agg_["Full"][regions], Agg_["Empty"], Agg_["Liquid"][regions]], axis=1
)
out = ToLaTeX(Agg_)
out.to_csv(os.path.join(sPath, "Output", "QuantileLoss_RF.txt"), index=False)


##### MSE profitability vs median vs mean
### NN
res = []
for Type in ["Full", "Liquid"]:
    ## Mean computed from quantiles
    File = "NN_clean_skewness_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    # dt = dt.loc[dt["date"] <= "2018-12-31"].copy()
    dt = CreatePredSignal(dt, PredVar="m1")
    ret_base = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret_base)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "Quantiles - Mean"
    metrics["type"] = Type
    res += [metrics]

    ## RF with vol and normalized predicted returns
    File = "RF1_full_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = dt.loc[dt["date"] <= "2018-12-31"].copy()
    dt = CreatePredSignal(dt, PredVar="Q0.5")
    ret = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "RF"
    metrics["type"] = Type
    # add t-stat wrt MSE with vol
    diff = ret_base[2] - ret[2]
    tstat = diff.groupby("region").mean() / diff.groupby("region").apply(lambda x: NW_std(x.values))
    tstat = tstat * np.sqrt(diff.groupby("region").count())
    metrics = metrics.merge(pd.DataFrame({"t-stat": tstat}), on="region")
    res += [metrics]

    ## RF with vol and normalized predicted returns
    File = "GBRT1_full_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = dt.loc[dt["date"] <= "2018-12-31"].copy()
    dt = CreatePredSignal(dt, PredVar="Q0.5")
    ret = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "GBRT"
    metrics["type"] = Type
    # add t-stat wrt MSE with vol
    diff = ret_base[2] - ret[2]
    tstat = diff.groupby("region").mean() / diff.groupby("region").apply(lambda x: NW_std(x.values))
    tstat = tstat * np.sqrt(diff.groupby("region").count())
    metrics = metrics.merge(pd.DataFrame({"t-stat": tstat}), on="region")
    res += [metrics]
res = pd.concat(res)
res.rename({"ls_mean": "Avg Ret", "ls_SR": "SR"}, axis=1, inplace=True)
res["Avg Ret"] = res["Avg Ret"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
res["SR"] = res["SR"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
res["t-stat"] = res["t-stat"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
res = res.melt(id_vars=["type", "Spec", "region"]).set_index(["type", "region", "Spec", "variable"])
res_ = res["value"].unstack([0, 1]).reset_index()
regions = ["USA", "Europe", "Japan", "Asia Pacific", "Global"]
res_["Empty"] = ""
res_ = pd.concat(
    [res_[["Spec", "variable"]], res_["Full"][regions], res_["Empty"], res_["Liquid"][regions]], axis=1
)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "MSEvsQuantile1_RF.txt"), index=False)
