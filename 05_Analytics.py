import pandas as pd
import numpy as np
import os
import seaborn

from EstimationFunctions.NN_Functions import *
import matplotlib.pyplot as plt


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
taus = (
    [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.075]
    + [0.925, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99995]
    + [i / 20 for i in range(1, 20)]
)
taus.sort()


##### number of observations
res = []
for Type in ["Full", "Liquid"]:
    File = "NN_clean_M_full_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = (
        dt.groupby(["region", "date"])["DTID"]
        .count()
        .groupby("region")
        .agg(["min", "mean", "max"])
        .round(0)
        .astype(int)
    )
    dt = dt.reset_index()
    dt["Type"] = Type
    res.append(dt)
res = pd.concat(res)
regions = ["USA", "Europe", "Japan", "Asia Pacific"]
Agg = res.melt(id_vars=["Type", "region"]).set_index(["Type", "region", "variable"])
Agg = Agg["value"].unstack([0, 1]).reset_index()
regions = ["USA", "Europe", "Japan", "Asia Pacific"]
Agg["Empty"] = ""
Agg = pd.concat([Agg[["variable"]], Agg["Full"][regions], Agg["Empty"], Agg["Liquid"][regions]], axis=1)
out = ToLaTeX(Agg)
out.to_csv(os.path.join(sPath, "Output", "Observation_counts.txt"), index=False)


##### hyperparameter search
# parameters search for one architecture
Settings = [
    [0.01, 0.0],
    [0.01, 0.2],
    [0.01, 0.4],
    [0.001, 0.0],
    [0.001, 0.2],
    [0.001, 0.4],
    [0.0001, 0.0],
    [0.0001, 0.2],
    [0.0001, 0.4],
    [0.0003, 0.2],
    [0.003, 0.2],
    [0.0003, 0.3],
    [0.0003, 0.1],
    [0.001, 0.1],
    [0.001, 0.3],
]
res = []
for LR, DR in Settings:
    DR_, LR_ = str(DR).replace("0.", ""), str(LR).replace("0.", "")
    try:
        pred = pd.read_parquet(
            os.path.join(sPath, "Predict", f"NN_clean_M_full_parameter_search_LR{LR_}_DR{DR_}.gzip"),
        )
        loss = np.mean([mean_quantile_loss(pred["r_raw"], pred[f"pred_raw_{tau}"], alpha=tau) for tau in taus])
    except:
        loss = np.nan
    try:
        pred = pd.read_parquet(
            os.path.join(sPath, "Predict", f"NN_clean_M_parameter_search_LR{LR_}_DR{DR_}.gzip"),
        )
        loss2 = np.mean([mean_quantile_loss(pred["r_raw"], pred[f"pred_raw_{tau}"], alpha=tau) for tau in taus])
    except:
        loss2 = np.nan
    res += [
        pd.DataFrame(
            {
                "Learning Rate": str(LR),
                "Dropout Rate": str(DR),
                "Loss Full": "" if np.isnan(loss) else f"{loss:.6f}",
                # "Loss Liquid": "" if np.isnan(loss2) else f"{loss2:.6f}",
            },
            index=[0],
        )
    ]
res = pd.concat(res)
ToLaTeX_list(res.columns)
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "HyperparameterSearch1.txt"), index=False)

# architecture search
Settings = [
    ["s1_2x128_s2_0", [128, 128, 4, 128, 128], []],
    ["s1_2x128_s2_16", [128, 128, 4, 128, 128], [16]],
    ["s1_2x64_s2_8", [64, 64, 4, 64, 64], [8]],
    ["s1_2x256_s2_8", [256, 256, 4, 256, 256], [8]],
    ["s1_3x128_s2_8", [128, 128, 128, 4, 128, 128, 128], [8]],
    ["s1_1x128_s2_8", [128, 4, 128], [8]],
    ["s1_4x64_s2_8", [64, 64, 64, 64, 4, 64, 64, 64, 64], [8]],
]
res = []
for Spec, hs1, hs2 in Settings:
    pred = pd.read_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_full_parameter_search_{Spec}.gzip"),
    )
    pred2 = pd.read_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_parameter_search_{Spec}.gzip"),
    )
    SpecL = Spec.split("_")
    loss = np.mean([mean_quantile_loss(pred["r_raw"], pred[f"pred_raw_{tau}"], alpha=tau) for tau in taus])
    loss2 = np.mean([mean_quantile_loss(pred2["r_raw"], pred2[f"pred_raw_{tau}"], alpha=tau) for tau in taus])
    res += [
        pd.DataFrame(
            {
                "Stage1 Block": SpecL[1],
                "Stage2": SpecL[3] + "x1",
                "Loss Full": f"{loss:.6f}",
                # "Loss Liquid": f"{loss2:.6f}",
            },
            index=[0],
        )
    ]
res = pd.concat(res)
ToLaTeX_list(res.columns)
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "HyperparameterSearch2.txt"), index=False)

##### motivation for the two stage approach
Settings = {
    "GARCH": "VolaGARCH_Q_GARCH_t.gzip",
    "Two Stage": "NN_clean_M_full_m.gzip",
    "Linear": "NN_clean_onestage_NN0_full_M_m.gzip",
    "One Layer 32": "NN_clean_onestage_NN1_full_M_m.gzip",
    "Two Layers 128x128": "NN_clean_onestage_NN2_full_M_m.gzip",
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
out.to_csv(os.path.join(sPath, "Output", "TwoStageMotivation1.txt"), index=False)


##### compare different variant of GARCH
Settings = [
    ["GARCH(1, 1), normal dist", "bootstrap_GARCH_normal", "GARCH normal"],
    ["GARCH(1, 1), t dist", "bootstrap_GARCH_t", "GARCH t"],
    ["GARCH(1, 1) fixed, t dist 5 df", "bootstrap_RISKMETRICS_t5", "EWMAVol"],
    ["GARCH(1, 1) fixed, t dist 4 df", "bootstrap_RISKMETRICS_t4", "EWMAVol"],
    ["GARCH(1, 1) fixed, t dist 3 df", "bootstrap_RISKMETRICS_t3", "EWMAVol"],
    ["GARCH(2, 2), t dist", "bootstrap_GARCH2_t", "GARCH2 10mu t"],
    ["EGARCH(1, 1, 1), t dist", "bootstrap_EGARCH_t", "EGARCH t"],
    ["GARCH(1, 1), t dist, mu estim", "bootstrap_GARCH_mu_t", "GARCH mu t"],
]
pred_Q = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q.gzip"))
fit_full = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH.gzip"))
res = []
for Type in ["Full", "Liquid"]:
    File = "NN_clean_M_full_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    pred_full = GetPredictions(sPath, files, pred_file=File, keep_r=True)
    pred_full = pred_full.loc[pred_full["region"] == "USA"].copy()
    for Desc, Spec, Spec2 in Settings:
        pred2 = pred_Q.loc[pred_Q["Spec"] == Spec].copy()
        fit = fit_full.loc[(fit_full["Spec"] == Spec2) & fit_full["Converged"].notnull()].copy()
        fit = fit.loc[~fit["Converged2"].astype(bool) | ~fit["Converged"].astype(bool)].copy()
        fit["date"] = pd.to_datetime(fit["date"])
        fit = fit[["DTID", "date", "Converged"]].copy()
        pred2 = pred2.merge(fit, on=["DTID", "date"], how="left")
        pred2 = pred2.loc[pred2["Converged"].isnull()].copy()
        pred3 = pred_Q.loc[pred_Q["Spec"] == "bootstrap_RISKMETRICS_t4"].copy()
        pred3 = pred3.merge(fit, on=["DTID", "date"], how="inner")
        pred2 = pd.concat([pred2, pred3])
        pred = pred_full.merge(pred2, on=["date", "DTID"], how="inner")
        for col_ in [f"Q{tau}" for tau in taus]:
            pred.loc[pred[col_] < -1, col_] = -1
            pred.loc[pred[col_] > 50, col_] = 50
        loss = pred.groupby("date").apply(
            lambda x: np.mean([mean_quantile_loss(x["r_raw"], x[f"Q{tau}"], alpha=tau) for tau in taus])
        )
        res += [
            pd.DataFrame(
                {
                    "Spec": Desc,
                    "type": Type,
                    "loss": loss,
                }
            ).reset_index()
        ]
res = pd.concat(res)
res_bench = res.loc[(res["Spec"] == "GARCH(1, 1), t dist")].copy()
res_bench.drop(["Spec"], axis=1, inplace=True)
res_bench.rename({"loss": "loss_bench"}, axis=1, inplace=True)
res = res.merge(res_bench, on=["date", "type"])
res["diff"] = res["loss"] - res["loss_bench"]
Agg = res.groupby(["type", "Spec"])["diff"].agg(["mean", "count"])
Agg["Avg Loss"] = res.groupby(["type", "Spec"])["loss"].mean()
Agg["Error"] = res.groupby(["type", "Spec"])["diff"].apply(lambda x: NW_std(x.values))
Agg["t-stat"] = Agg["mean"] / Agg["Error"] * np.sqrt(Agg["count"])
Agg = Agg[["Avg Loss", "t-stat"]].copy()
Agg["Avg Loss"] = Agg["Avg Loss"].transform(lambda x: "" if np.isnan(x) else f"{x:.5f}")
Agg["t-stat"] = Agg["t-stat"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
Agg = Agg.reset_index().melt(id_vars=["type", "Spec"]).set_index(["type", "Spec", "variable"])
Agg_ = Agg["value"].unstack([0]).reset_index()
ToLaTeX_list(Agg_.columns)
out = ToLaTeX(Agg_)
out.to_csv(os.path.join(sPath, "Output", "GARCH_estim1.txt"), index=False)


##### portfolio single sorts
def CreateSingleSorts(Var, sort_n=10, wgt_type="EW", File="NN_clean_skewness_full.gzip"):
    res = []
    for Type in ["Full", "Liquid"]:
        if Type == "Liquid":
            dt = GetPredictions(sPath, files, horizon="M", full_sample=False)
            dt2 = pd.read_parquet(os.path.join(sPath, "Predict", File.replace("_full", "")))
        else:
            dt = GetPredictions(sPath, files, horizon="M", full_sample=True)
            dt2 = pd.read_parquet(os.path.join(sPath, "Predict", File))
        # dt2 = dt2.loc[~dt2['LinearFlag']].copy()
        dt = dt.merge(dt2, on=["DTID", "date", "region"])
        dt = AdjustMoments(dt)
        pred = CreatePredSignalSorts(dt, Var1=Var, sort_n=sort_n)
        port = ConstructPortfolios(pred, sPath, ret_type="M", wgt_type=wgt_type, port_type="sorts")
        res_ = PortfolioMetricsSorts(port, sort_n=sort_n)
        res_["type"] = Type
        res += [res_]
    res = pd.concat(res)
    res_ = res.set_index(["region", "Var1_num", "Var1", "type"])["out"].unstack([3, 0]).fillna("").reset_index()
    regions = ["USA", "Europe", "Japan", "Asia Pacific"]
    res_["Empty"] = ""
    res_ = pd.concat([res_[["Var1"]], res_["Full"][regions], res_["Empty"], res_["Liquid"][regions]], axis=1)
    return res_


### EW
## mean
res = CreateSingleSorts(Var="m1")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts1.txt"), index=False)

## median
res = CreateSingleSorts(Var="pred_raw_0.5")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts2.txt"), index=False)

## volatility
res = CreateSingleSorts(Var="var_adj")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts3.txt"), index=False)

## skewness
res = CreateSingleSorts(Var="skew_adj")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts4.txt"), index=False)

## kurtosis
res = CreateSingleSorts(Var="kurtosis_adj")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts5.txt"), index=False)

### VW
## mean
res = CreateSingleSorts(Var="m1", wgt_type="VW")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts1_VW.txt"), index=False)

## median
res = CreateSingleSorts(Var="pred_raw_0.5", wgt_type="VW")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts2_VW.txt"), index=False)

## volatility
res = CreateSingleSorts(Var="var_adj", wgt_type="VW")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts3_VW.txt"), index=False)

## skewness
res = CreateSingleSorts(Var="skew_adj", wgt_type="VW")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts4_VW.txt"), index=False)

## kurtosis
res = CreateSingleSorts(Var="kurtosis_adj", wgt_type="VW")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts5_VW.txt"), index=False)

### EW for one stage NN2
## mean
res = CreateSingleSorts(Var="m1", File="NN_clean_skewness_NN2_full.gzip")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts1_NN2.txt"), index=False)

## volatility
res = CreateSingleSorts(Var="var_adj", File="NN_clean_skewness_NN2_full.gzip")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts3_NN2.txt"), index=False)

## skewness
res = CreateSingleSorts(Var="skew_adj", File="NN_clean_skewness_NN2_full.gzip")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts4_NN2.txt"), index=False)

## kurtosis
res = CreateSingleSorts(Var="kurtosis_adj", File="NN_clean_skewness_NN2_full.gzip")
res.to_latex().replace("SpaceHolder", "")  # for headers
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "SingleSorts5_NN2.txt"), index=False)


##### L, S, LS for individual quantiles
Qs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
res = []
res_add = []
for Type in ["Full", "Liquid"]:
    if Type == "Liquid":
        dt = GetPredictions(sPath, files, horizon="M", full_sample=False)
    else:
        dt = GetPredictions(sPath, files, horizon="M", full_sample=True)
    ret_join = []
    for Q in Qs:
        pred = CreatePredSignal(dt, PredVar=f"pred_raw_{Q}")
        ret = ConstructPortfolios(pred, sPath, ret_type="M", wgt_type="EW")
        metrics = PortfolioMetricsClean(*ret)
        metrics["type"] = Type
        metrics["Q"] = Q
        res += [metrics]
        ret_join += [ret]
    for i in range(len(Qs)):
        long = ret_join[i][0]
        short = ret_join[len(Qs) - i - 1][1]
        ls = long - short
        metrics = PortfolioMetricsCleanLS(ls)[["region", "ls", "ls_SR"]]
        metrics.rename({"ls": "ls_cross", "ls_SR": "ls_cross_SR"}, axis=1, inplace=True)
        metrics["type"] = Type
        metrics["Q"] = Qs[i]
        res_add += [metrics]
res = pd.concat(res)
res_add = pd.concat(res_add)
res = res.merge(res_add, on=["type", "Q", "region"])
for Var in ["long_SR", "short_SR", "ls_SR"]:
    res[Var] = res[Var].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
Vars = ["long", "long_SR", "short", "short_SR", "ls", "ls_SR"]  # , "ls_cross", "ls_cross_SR"
for region in ["USA", "Europe", "Japan", "Asia Pacific"]:
    res1 = res.loc[(res["region"] == region) & (res["type"] == "Full"), ["Q"] + Vars].reset_index(drop=True)
    res2 = res.loc[(res["region"] == region) & (res["type"] == "Liquid"), Vars].reset_index(drop=True)
    res1["Empty"] = ""
    res_ = pd.concat([res1, res2], axis=1)
    out = ToLaTeX(res_)
    out.to_csv(os.path.join(sPath, "Output", f"QuantilePortfolios{region.replace(' ', '')}.txt"), index=False)
    # QuantilePortfoliosUSA.txt, QuantilePortfoliosEurope.txt, QuantilePortfoliosJapan.txt, QuantilePortfoliosAsiaPacific.txt


##### L, S, LS for individual quantiles benchmark model
Qs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
res = []
res_add = []
for Type in ["Full", "Liquid"]:
    if Type == "Liquid":
        dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN2_M_m.gzip")
    else:
        dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN2_full_M_m.gzip")
    ret_join = []
    for Q in Qs:
        pred = CreatePredSignal(dt, PredVar=f"pred_{Q}")
        ret = ConstructPortfolios(pred, sPath, ret_type="M", wgt_type="EW")
        metrics = PortfolioMetricsClean(*ret)
        metrics["type"] = Type
        metrics["Q"] = Q
        res += [metrics]
        ret_join += [ret]
    for i in range(len(Qs)):
        long = ret_join[i][0]
        short = ret_join[len(Qs) - i - 1][1]
        ls = long - short
        metrics = PortfolioMetricsCleanLS(ls)[["region", "ls", "ls_SR"]]
        metrics.rename({"ls": "ls_cross", "ls_SR": "ls_cross_SR"}, axis=1, inplace=True)
        metrics["type"] = Type
        metrics["Q"] = Qs[i]
        res_add += [metrics]
res = pd.concat(res)
res_add = pd.concat(res_add)
res = res.merge(res_add, on=["type", "Q", "region"])
for Var in ["long_SR", "short_SR", "ls_SR"]:
    res[Var] = res[Var].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
Vars = ["long", "long_SR", "short", "short_SR", "ls", "ls_SR"]  # , "ls_cross", "ls_cross_SR"
for region in ["USA"]:  # , "Europe", "Japan", "Asia Pacific"
    res1 = res.loc[(res["region"] == region) & (res["type"] == "Full"), ["Q"] + Vars].reset_index(drop=True)
    res2 = res.loc[(res["region"] == region) & (res["type"] == "Liquid"), Vars].reset_index(drop=True)
    res1["Empty"] = ""
    res_ = pd.concat([res1, res2], axis=1)
    out = ToLaTeX(res_)
    out.to_csv(
        os.path.join(sPath, "Output", f"QuantilePortfoliosBenchmark{region.replace(' ', '')}.txt"), index=False
    )
    # QuantilePortfoliosBenchmarkUSA.txt


##### MSE R-squared vs median vs mean
# get raw returns that are being predicted
Cols = ["DTID", "date", "r", "region", "MC"]
ret = pd.read_parquet(os.path.join(sPath, "Features", "Monthly_ret.gzip"), columns=Cols)

# get risk-free rate
rf = pd.read_csv(os.path.join(sPath, "Inputs", "FF3.CSV"))
rf["date"] = rf["date"].astype(str)
rf["date"] = pd.to_datetime(rf["date"].str.slice(0, 4) + "-" + rf["date"].str.slice(4, 6) + "-01")
rf["RF"] = rf["RF"] / 100
rf = rf[["date", "RF"]].copy()
ret = ret.merge(rf, on="date")
ret["r"] = ret["r"] - ret["RF"]

res = []
for Type in ["Full", "Liquid"]:
    ## Mean computed from quantiles
    File = "NN_clean_skewness_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["m1"] = dt["m1"] - dt["RF"]
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="m1", actual="r"),
            "Spec": "Two stage NN - Mean",
            "type": Type,
        }
    )
    res += [metrics]
    dt_bench = dt[["DTID", "date", "m1"]].copy()

    ## Mean computed from quantiles from one stage NN2
    File = "NN_clean_skewness_NN2_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["m1_alt"] = dt["m1"] - dt["RF"]
    dt["m1_alt"] = dt["m1_alt"].clip(-0.5, 0.5)
    del dt["m1"]
    dt = dt.merge(dt_bench, on=["DTID", "date"])
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="m1_alt", actual="r"),
            "Spec": "2hNN Quantile - Mean",
            "DM stat": DieboldMariano_df(dt, pred1="m1_alt", pred2="m1", actual="r", AddGlobal=True),
            "type": Type,
        }
    )
    res += [metrics]

    ## Median
    dt = GetPredictions(sPath, files, horizon="M", full_sample=False if Type == "Liquid" else True)
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["pred_raw_0.5"] = dt["pred_raw_0.5"] - dt["RF"]
    dt = dt.merge(dt_bench, on=["DTID", "date"])
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="pred_raw_0.5", actual="r"),
            "DM stat": DieboldMariano_df(dt, pred1="pred_raw_0.5", pred2="m1", actual="r", AddGlobal=True),
            "Spec": "Two stage NN - Median",
            "type": Type,
        }
    )
    res += [metrics]

    ## MSE with vol and normalized predicted returns
    File = "NN_clean_MSE_NN2_raw_full_M_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["pred"] = dt["pred"] - dt["RF"]
    dt = dt.merge(dt_bench, on=["DTID", "date"])
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="pred", actual="r"),
            "DM stat": DieboldMariano_df(dt, pred1="pred", pred2="m1", actual="r", AddGlobal=True),
            "Spec": "2hNN MSE - Mean",
            "type": Type,
        }
    )
    res += [metrics]
res = pd.concat(res)
res["R-sqrd"] = res["R-sqrd"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["DM stat"] = res["DM stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
res = res.reset_index().melt(id_vars=["type", "region", "Spec"], var_name="Measure")
res = res.set_index(["type", "region", "Spec", "Measure"])["value"].unstack([0, 1]).reset_index()
regions = ["USA", "Europe", "Japan", "Asia Pacific", "Global"]
res["Empty"] = ""
res_ = pd.concat(
    [res[["Spec"]], res[["Measure"]], res["Full"][regions], res["Empty"], res["Liquid"][regions]], axis=1
)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "MSEvsQuantile2.txt"), index=False)


##### volatility forecasting vs GARCH
res = []
for Type in ["Full", "Liquid"]:
    pred = pd.read_parquet(os.path.join(sPath, "Predict", "Volatility_m.gzip"))
    pred["date"] = pd.to_datetime(pred["date"])
    pred.rename({"var": "var_oos"}, axis=1, inplace=True)
    pred2 = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_GARCH_t.gzip"))
    pred2.rename({"vol": "std_GARCH"}, axis=1, inplace=True)
    File = "NN_clean_skewness_full.gzip"
    File2 = "NN_clean_skewness_NN2_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
        File2 = File2.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt2 = pd.read_parquet(os.path.join(sPath, "Predict", File2))
    dt2.rename({"std": "std_NN2"}, axis=1, inplace=True)
    dt = dt.merge(pred2[["DTID", "date", "std_GARCH"]], on=["DTID", "date"], how="left")
    dt = dt.merge(pred, on=["DTID", "date"], how="left")
    dt = dt.merge(dt2[["DTID", "date", "std_NN2"]], on=["DTID", "date"], how="left")
    dt = dt.loc[(dt["std_GARCH"] < 1) & (dt["vol"] < 1)].copy()
    metrics = pd.DataFrame(
        {
            "RMSE": RMSE_df(dt, pred="std", actual="vol"),
            "MAD": MAD_df(dt, pred="std", actual="vol"),
            "Spec": "NN Quantiles",
            "type": Type,
        }
    )
    res += [metrics]
    metrics = pd.DataFrame(
        {
            "RMSE": RMSE_df(dt, pred="std_GARCH", actual="vol"),
            "MAD": MAD_df(dt, pred="std_GARCH", actual="vol"),
            "DM stat": DieboldMariano_df(dt, pred1="std", pred2="std_GARCH", actual="vol", AddGlobal=True),
            "Spec": "GARCH",
            "type": Type,
        }
    )
    res += [metrics]
    metrics = pd.DataFrame(
        {
            "RMSE": RMSE_df(dt, pred="std_NN2", actual="vol"),
            "MAD": MAD_df(dt, pred="std_NN2", actual="vol"),
            "DM stat": DieboldMariano_df(dt, pred1="std", pred2="std_NN2", actual="vol", AddGlobal=True),
            "Spec": "2hNN",
            "type": Type,
        }
    )
    res += [metrics]
res = pd.concat(res)
res["RMSE"] = res["RMSE"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["MAD"] = res["MAD"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["DM stat"] = res["DM stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
res = res.reset_index().melt(id_vars=["type", "region", "Spec"], var_name="Measure")
res = res.set_index(["type", "region", "Spec", "Measure"])["value"].unstack([0, 1]).reset_index()
regions = ["USA", "Europe", "Japan", "Asia Pacific", "Global"]
res["Empty"] = ""
res_ = pd.concat(
    [res[["Spec"]], res[["Measure"]], res["Full"][regions], res["Empty"], res["Liquid"][regions]], axis=1
)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "VolaPredict.txt"), index=False)


##### motivation for the two stage approach -> breakdown of quantiles
Settings = {
    "GARCH": "VolaGARCH_Q_GARCH_t.gzip",
    "Linear": "NN_clean_onestage_NN0_full_M_m.gzip",
    "One Layer 32": "NN_clean_onestage_NN1_full_M_m.gzip",
    "Two Layers 128x128": "NN_clean_onestage_NN2_full_M_m.gzip",
    "Two Stage": "NN_clean_M_full_m.gzip",
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
        pred = pred.loc[pred["region"] == "USA"].copy()
        pred = pred.loc[pred["date"] <= "2018-12-31"].copy()
        pred.rename({f"{col}{tau}": f"Q{tau}" for tau in taus}, axis=1, inplace=True)
        pred_ = pred.melt(id_vars=["date", "DTID"], value_vars=[f"Q{tau}" for tau in taus])
        pred_["tau"] = pred_["variable"].map({f"Q{tau}": tau for tau in taus})
        pred_ = pred_.merge(pred[["date", "DTID", "r_raw"]], on=["date", "DTID"])
        loss = pred_.groupby(["date", "variable"]).apply(
            lambda x: mean_quantile_loss(x["r_raw"], x["value"], alpha=x["tau"].values[0])
        )
        res += [pd.DataFrame({"Spec": Spec, "type": Type, "loss": loss}).reset_index()]
res = pd.concat(res)
res_bench = res.loc[res["Spec"] == "Two Stage"].copy()
del res_bench["Spec"]
res_bench.rename({"loss": "loss_bench"}, axis=1, inplace=True)
res = res.merge(res_bench, on=["date", "type", "variable"])
res["variable"] = res["variable"].map({f"Q{tau}": tau for tau in taus})
res["diff"] = res["loss"] - res["loss_bench"]
Agg = res.groupby(["type", "Spec", "variable"])["diff"].agg(["mean", "count"])
Agg["Avg Loss"] = res.groupby(["type", "Spec", "variable"])["loss"].mean()
Agg["Error"] = res.groupby(["type", "Spec", "variable"])["diff"].apply(lambda x: NW_std(x.values))
Agg["t-stat"] = Agg["mean"] / Agg["Error"] * np.sqrt(Agg["count"])
Agg = Agg[["Avg Loss", "t-stat"]].copy()
Agg["Avg Loss"] = Agg["Avg Loss"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.3f}")
Agg["t-stat"] = Agg["t-stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
Agg.reset_index(inplace=True)
Agg.rename({"variable": "Q"}, axis=1, inplace=True)
Agg = Agg.melt(id_vars=["type", "Spec", "Q"]).set_index(["type", "Spec", "Q", "variable"])
Agg_ = Agg["value"].unstack([0, 1]).reset_index()
Agg_ = Agg_.loc[Agg_["variable"] == "Avg Loss"]
Agg_["Empty"] = ""
Vars = Settings.keys()
Agg_ = pd.concat([Agg_[["Q"]], Agg_["Full"][Vars], Agg_["Empty"], Agg_["Liquid"][Vars]], axis=1)
out = ToLaTeX(Agg_)
out.to_csv(os.path.join(sPath, "Output", "TwoStageMotivation1_Qs.txt"), index=False)

##### Scoring rule
## NN two stage
ret = pd.read_parquet(os.path.join(sPath, "Features", "Monthly_ret.gzip"), columns=["DTID", "date", "r"])
Settings = {
    "Linear": "NN_clean_onestage_NN0_full_M_m.gzip",
    "OneLayer32": "NN_clean_onestage_NN1_full_M_m.gzip",
    "TwoLayers128": "NN_clean_onestage_NN2_full_M_m.gzip",
    "TwoStage": "NN_clean_M_full_m.gzip",
}
for Type in ["Full", "Liquid"]:
    for Spec, File in Settings.items():
        if Type == "Liquid":
            File = File.replace("_full", "")
        dt = GetPredictions(sPath, files, pred_file=File)
        dt = dt.merge(ret, on=["DTID", "date"])
        dt = ComputeScoring(dt, taus)
        file_type = "" if Type == "Liquid" else "_full"
        dt.to_parquet(os.path.join(sPath, "Predict", f"NN_{Spec}_scoring{file_type}.gzip"), compression="gzip")

Settings = {
    "GARCH": "ScoringGARCH.gzip",
    "Two Stage": "NN_TwoStage_scoring_full.gzip",
    "One Layer 32": "NN_OneLayer32_scoring_full.gzip",
    "Two Layers 128x128": "NN_TwoLayers128_scoring_full.gzip",
}
res = []
# there are problems with missing data for some of the models so need to only use predictions that are available for all of them
for Type in ["Full", "Liquid"]:
    dt = []
    for Spec, File in Settings.items():
        if Spec == "GARCH":
            pred2 = pd.read_parquet(os.path.join(sPath, "Predict", "ScoringGARCH.gzip"))
        else:
            if Type == "Liquid":
                File = File.replace("_full", "")
            pred2 = pd.read_parquet(os.path.join(sPath, "Predict", File))
        pred2 = pred2[["date", "DTID", "score"]].rename({"score": Spec}, axis=1)
        if len(dt) == 0:
            dt = pred2.copy()
        else:
            dt = dt.merge(pred2, on=["date", "DTID"], how="inner")
    # add region
    if Type == "Liquid":
        pred = GetPredictions(sPath, files, pred_file="NN_clean_M_m.gzip")
    else:
        pred = GetPredictions(sPath, files, pred_file="NN_clean_M_full_m.gzip")
    pred = pred[["date", "DTID", "region"]].merge(dt, on=["date", "DTID"], how="inner")
    pred = pred.loc[pred["date"] <= "2018-12-31"].copy()
    pred = pred.dropna().copy()
    # mean score
    for Spec, File in Settings.items():
        for reg in ["USA", "Europe", "Japan", "Asia Pacific"]:
            pred_ = pred.loc[pred["region"] == reg]
            loss = pred_.groupby("date")[Spec].mean()
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
Agg.rename({"Avg Loss": "Avg CRPS", "t-stat": "(t-stat)"}, axis=1, inplace=True)
Agg = Agg.reset_index().melt(id_vars=["type", "Spec", "region"]).set_index(["type", "region", "Spec", "variable"])
Agg_ = Agg["value"].unstack([0, 1]).reset_index()
Agg_.loc[Agg_["variable"] == "(t-stat)", "Spec"] = " "
regions = ["USA", "Europe", "Japan", "Asia Pacific"]
Agg_["Empty"] = ""
Agg_ = pd.concat(
    [Agg_[["Spec", "variable"]], Agg_["Full"][regions], Agg_["Empty"], Agg_["Liquid"][regions]], axis=1
)
out = ToLaTeX(Agg_)
out.to_csv(os.path.join(sPath, "Output", "TwoStageScoring1.txt"), index=False)


##############################################################################################################
################################################## Plots #####################################################
##### example plot for one ISIN
## plot of quantiles for individual ISINs
# need to remove the last digit
"02079K10"  # google -> 14542
"03783310"  # apple -> 14593
"59491810"  # microsoft -> 10107

VarMap = {f"pred_raw_{tau}": f"{tau} Quantile" for tau in [0.01, 0.1, 0.5, 0.9, 0.99]}
VarMap["r_raw"] = "22d-ahead Return"

dt = GetPredictions(sPath, files, horizon="M", full_sample=False, region=["USA"], keep_r=True)
dt = dt.rename(VarMap, axis=1).set_index("date")
dt_ = dt.loc[dt["DTID"] == "14593"].copy()  # apple
# dt_ = dt.loc[dt["DTID"] == "14542"].copy()  # google
# dt_ = dt.loc[dt["DTID"] == "10107"].copy()  # microsoft

fig = plt.figure(figsize=(10, 7))
ax = plt.subplot(1, 1, 1)
ax.plot(dt_["0.01 Quantile"], label="0.01 Quantile", linestyle="--", color="blue")
ax.plot(dt_["0.1 Quantile"], label="0.1 Quantile", linestyle="--", color="lightblue")
ax.plot(dt_["0.5 Quantile"], label="0.5 Quantile", color="red")
ax.plot(dt_["0.9 Quantile"], label="0.9 Quantile", linestyle="--", color="lightblue")
ax.plot(dt_["0.99 Quantile"], label="0.99 Quantile", linestyle="--", color="blue")
ax.plot(dt_["22d-ahead Return"], label="22d-ahead Return", color="black")
ax.set_ylabel("22-day Return")
ax.set_xlabel("")
ax.set_title("Apple: CUSIP 03783310")
ax.legend()
fig.savefig(os.path.join(sPath, "Output", "Apple.png"))
plt.close()

##### evolution of density
### Microsoft over 2008
dt = GetPredictions(sPath, files, horizon="M", full_sample=False, region=["USA"], keep_r=True)
dt = dt.loc[(dt["date"] >= "2008-07-01") & (dt["date"] <= "2008-12-31") & (dt["DTID"] == "10107")].copy()
y = np.array(taus)
cols = [f"pred_raw_{tau}" for tau in taus]
res = []
for i, date in enumerate(dt["date"]):
    x = dt.iloc[i][cols].values.astype(float)
    res_ = DensityIntegrationPlots(x, y)
    res_["date"] = date
    res += [res_]
res = pd.concat(res)
res.loc[np.abs(res["x"]) < 0.3].set_index("x").groupby("date")["density"].plot(legend=True)

### Microsoft beginning of Oct 2008
dt = dt.loc[(dt["date"] == "2008-10-01") & (dt["DTID"] == "10107")].copy()
y = np.array(taus)
cols = [f"pred_raw_{tau}" for tau in taus]
x = dt.iloc[0][cols].values.astype(float)
res = DensityIntegrationPlots(x, y)

fig = plt.figure(figsize=(14, 6))
seaborn.set_theme()
ax = plt.subplot(1, 2, 1)
seaborn.lineplot(data=res, ax=ax, x="x", y="cdf")
seaborn.scatterplot(ax=ax, x=x, y=y, s=50, color="black")
ax.set_xlabel("22-day Return")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Cumulative Density Function")
ax = plt.subplot(1, 2, 2)
seaborn.lineplot(data=res, ax=ax, x="x", y="density")
ax.set_xlabel("22-day Return")
ax.set_ylabel("Probability Density")
ax.set_title("Density Function")
fig.suptitle("Microsoft (CUSIP 59491810) October 2008")
fig.savefig(os.path.join(sPath, "Output", "Microsoft2008Oct.png"))
plt.close()

### Microsoft: cdf-pdf
plt.style.use("default")
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
pd.Series(taus, index=x).plot(marker="x", color="black", markerfacecolor="gray", ax=axs[0])
axs[0].set_xlabel(xlabel=r"22-day return", fontsize=14)
axs[0].set_ylabel(ylabel=r"Cumulative probability", fontsize=14)

res.set_index("x")["density"].plot(
    markerfacecolor="gray",
    color="black",
    ylabel="Probability density",
    xlabel=r"22-day return",
    linestyle="-",
    ax=axs[1],
)
axs[1].set_xlabel(xlabel=r"22-day return", fontsize=14)
axs[1].set_ylabel(ylabel=r"Cumulative probability", fontsize=14)

plt.tight_layout()
plt.savefig(sPath, "Output", "cdf_pdf_msft_corrected.png")
plt.savefig(sPath, "Output", "cdf_pdf_msft_corrected.pdf")

### Peaked distribution
dt = GetPredictions(sPath, files, horizon="M", full_sample=True, keep_r=True)
dt = dt.loc[(dt["date"] == "2014-08-01") & (dt["DTID"] == "90280")].copy()  # quite nice, CUSIP 45166R20
# IDENIX PHARMACEUTICA - acquired by Merck with the transaction closing in 2014Q3
# https://www.merck.com/news/merck-to-acquire-idenix/
y = np.array(taus)
cols = [f"pred_raw_{tau}" for tau in taus]
x = dt.iloc[0][cols].values.astype(float)
res = DensityIntegrationPlots(x, y)

fig = plt.figure(figsize=(14, 6))
seaborn.set_theme()
ax = plt.subplot(1, 2, 1)
seaborn.lineplot(data=res, ax=ax, x="x", y="cdf")
seaborn.scatterplot(ax=ax, x=x, y=y, s=50, color="black")
ax.set_xlabel("22-day Return")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Cumulative Density Function")
ax = plt.subplot(1, 2, 2)
seaborn.lineplot(data=res, ax=ax, x="x", y="density")
ax.set_xlabel("22-day Return")
ax.set_ylabel("Probability Density")
ax.set_title("Density Function")
fig.suptitle("Idenix Pharmaceuticals, Inc. (CUSIP 45166R20) August 2014")
fig.savefig(os.path.join(sPath, "Output", "DirichletDensity.png"))
plt.close()


##### create plot with r_scale and its interaction with the second stage
## liquid
# get r_scale covering even estimation sample
data = pd.read_parquet(
    os.path.join(sPath, "Features", files["W_file_22d"]), columns=["region", "EWMAVol6", "date"]
)
data = data.loc[data["region"].isin(["USA"])].copy()
data.reset_index(drop=True, inplace=True)
rscale = data.groupby(["date"])[["EWMAVol6"]].mean() / 0.022 * 0.11
rscale.rename({"EWMAVol6": "r_scale_full"}, axis=1, inplace=True)

# r_scale adjusted with second stage NN
dt = pd.read_parquet(
    os.path.join(sPath, "Predict", "NN_clean_M.gzip"), columns=["pred_raw_0.99", "pred_0.99", "r_scale", "date"]
)
dt["stage2"] = dt["pred_raw_0.99"] / dt["pred_0.99"] / dt["r_scale"]
df = dt.groupby("date")[["stage2", "r_scale"]].median()
df["r_scale_adj"] = df["stage2"] * df["r_scale"]
df = df.merge(rscale, on="date", how="outer")
df["r_scale"] = df["r_scale"].fillna(df["r_scale_full"])
df_liq = df.loc[df.index <= "2018-12-31", ["r_scale", "r_scale_adj"]].copy()

## full
# get r_scale covering even estimation sample
data = pd.read_parquet(
    os.path.join(sPath, "Features", files["W_file_22d_full"]), columns=["region", "EWMAVol6", "date"]
)
data = data.loc[data["region"].isin(["USA"])].copy()
data.reset_index(drop=True, inplace=True)
rscale = data.groupby(["date"])[["EWMAVol6"]].mean() / 0.022 * 0.11
rscale.rename({"EWMAVol6": "r_scale_full"}, axis=1, inplace=True)

# r_scale adjusted with second stage NN
dt = pd.read_parquet(
    os.path.join(sPath, "Predict", "NN_clean_M_full.gzip"),
    columns=["pred_raw_0.99", "pred_0.99", "r_scale", "date"],
)
dt["stage2"] = dt["pred_raw_0.99"] / dt["pred_0.99"] / dt["r_scale"]
df = dt.groupby("date")[["stage2", "r_scale"]].median()
df["r_scale_adj"] = df["stage2"] * df["r_scale"]
df = df.merge(rscale, on="date", how="outer")
df["r_scale"] = df["r_scale"].fillna(df["r_scale_full"])
df_full = df.loc[df.index <= "2018-12-31", ["r_scale", "r_scale_adj"]].copy()

## create the plot
color1 = "#9E3232"
seaborn.set_theme(style="whitegrid")
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(2, 1, 1)
ax.plot(df_liq["r_scale"], label=r"$\overline{\sigma}$", color="black")
ax.plot(
    df_liq["r_scale_adj"], label=r"$\overline{\sigma} \times \widehat{\sigma}_t^M$", linestyle="--", color=color1
)
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_title("Liquid Sample")
ax.legend()
ax = plt.subplot(2, 1, 2)
ax.plot(df_full["r_scale"], label=r"$\overline{\sigma}$", color="black")
ax.plot(
    df_full["r_scale_adj"], label=r"$\overline{\sigma} \times \widehat{\sigma}_t^M$", linestyle="--", color=color1
)
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_title("Full Sample")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(sPath, "Output", "r_scale_adj.png"))
plt.close()


##############################################################################################################
########################################### US data 2019-2023 update #########################################
### quantile loss function fit
Settings = {
    "GARCH": "VolaGARCH_Q_GARCH_t.gzip",
    "Two Stage": "NN_clean_M_full_m.gzip",
    "Linear": "NN_clean_onestage_NN0_full_M_m.gzip",
    "One Layer 32": "NN_clean_onestage_NN1_full_M_m.gzip",
    "Two Layers 128x128": "NN_clean_onestage_NN2_full_M_m.gzip",
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
        pred = pred.loc[pred["date"] > "2018-12-31"].copy()
        if "Q0.5" in pred.columns:
            col = "Q"
        elif "pred_raw_0.5" in pred.columns:
            col = "pred_raw_"
        else:
            col = "pred_"
        loss = pred.groupby("date").apply(
            lambda x: np.mean([mean_quantile_loss(x["r_raw"], x[f"{col}{tau}"], alpha=tau) for tau in taus])
        )
        res += [pd.DataFrame({"Spec": Spec, "type": Type, "loss": loss}).reset_index()]
res = pd.concat(res)
res_bench = res.loc[res["Spec"] == "Two Stage"].copy()
del res_bench["Spec"]
res_bench.rename({"loss": "loss_bench"}, axis=1, inplace=True)
res = res.merge(res_bench, on=["date", "type"])
res["diff"] = res["loss"] - res["loss_bench"]
Agg = res.groupby(["type", "Spec"])["diff"].agg(["mean", "count"])
Agg["Avg Loss"] = res.groupby(["type", "Spec"])["loss"].mean()
Agg["Error"] = res.groupby(["type", "Spec"])["diff"].apply(lambda x: NW_std(x.values))
Agg["t-stat"] = Agg["mean"] / Agg["Error"] * np.sqrt(Agg["count"])
Agg = Agg[["Avg Loss", "t-stat"]].copy()
Agg["Avg Loss"] = Agg["Avg Loss"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.3f}")
Agg["t-stat"] = Agg["t-stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
Agg = Agg.reset_index().melt(id_vars=["type", "Spec"]).set_index(["type", "Spec", "variable"])
Agg_ = Agg["value"].unstack([0]).reset_index()
Agg_.loc[Agg_["variable"] == "t-stat", "Spec"] = " "
Agg_["variable"] = Agg_["variable"].map({"Avg Loss": r"$L_{avg}$", "t-stat": "(t-stat)"})
Agg_["Empty"] = ""
Agg_ = pd.concat([Agg_[["Spec", "variable", "Full"]], Agg_["Empty"], Agg_["Liquid"]], axis=1)
out = ToLaTeX(Agg_)
out.to_csv(os.path.join(sPath, "Output", "TwoStageMotivationUpdate1.txt"), index=False)


##### volatility forecasting vs GARCH
res = []
for Type in ["Full", "Liquid"]:
    pred = pd.read_parquet(os.path.join(sPath, "Predict", "Volatility_m.gzip"))
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.loc[pred["date"] > "2018-12-31"].copy()
    pred.rename({"var": "var_oos"}, axis=1, inplace=True)
    pred2 = pd.read_parquet(os.path.join(sPath, "Predict", "VolaGARCH_Q_GARCH_t.gzip"))
    pred2.rename({"vol": "std_GARCH"}, axis=1, inplace=True)
    File = "NN_clean_skewness_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt = dt.merge(pred2[["DTID", "date", "std_GARCH"]], on=["DTID", "date"], how="left")
    dt = dt.merge(pred, on=["DTID", "date"], how="left")
    dt = dt.loc[(dt["std_GARCH"] < 1) & (dt["vol"] < 1)].copy()
    metrics = pd.DataFrame(
        {
            "RMSE": RMSE_df(dt, pred="std", actual="vol"),
            "MAD": MAD_df(dt, pred="std", actual="vol"),
            "Spec": "NN Quantiles",
            "type": Type,
        }
    )
    res += [metrics]
    metrics = pd.DataFrame(
        {
            "RMSE": RMSE_df(dt, pred="std_GARCH", actual="vol"),
            "MAD": MAD_df(dt, pred="std_GARCH", actual="vol"),
            "DM stat": DieboldMariano_df(dt, pred1="std", pred2="std_GARCH", actual="vol", AddGlobal=True),
            "Spec": "GARCH",
            "type": Type,
        }
    )
    res += [metrics]
res = pd.concat(res)
res = res.loc[res.index == "USA"].copy()
res["RMSE"] = res["RMSE"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["MAD"] = res["MAD"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["DM stat"] = res["DM stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
res = res.melt(id_vars=["type", "Spec"], var_name="Measure")
res = res.reset_index(drop=True).set_index(["type", "Spec", "Measure"])["value"].unstack([0, 1]).reset_index()
res["Empty"] = ""
res_ = pd.concat([res[["Measure"]], res["Full"], res["Empty"], res["Liquid"]], axis=1)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "VolaPredictUpdate.txt"), index=False)


## LS portfolio metrics
res = []
for Type in ["Full", "Liquid"]:
    ## Mean computed from quantiles
    File = "NN_clean_skewness_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt = CreatePredSignal(dt, PredVar="m1")
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    ret_base = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret_base)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "Two stage NN - Mean"
    metrics["type"] = Type
    res += [metrics]

    ## Median
    dt = GetPredictions(sPath, files, horizon="M", full_sample=False if Type == "Liquid" else True)
    dt = CreatePredSignal(dt, PredVar="pred_raw_0.5")
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    ret = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "Two stage NN - Median"
    metrics["type"] = Type
    # add t-stat wrt MSE with vol
    diff = ret_base[2] - ret[2]
    tstat = diff.groupby("region").mean() / diff.groupby("region").apply(lambda x: NW_std(x.values))
    tstat = tstat * np.sqrt(diff.groupby("region").count())
    metrics = metrics.merge(pd.DataFrame({"t-stat": tstat}), on="region")
    res += [metrics]

    ## MSE with vol and normalized predicted returns
    File = "NN_clean_MSE_NN2_full_vol_M_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = CreatePredSignal(dt, PredVar="pred")
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    ret = ConstructPortfolios(dt, sPath, ret_type="M", wgt_type="EW")
    metrics = PortfolioMetrics(*ret)[["region", "ls_mean", "ls_SR"]]
    metrics["Spec"] = "2hNN MSE - Mean"
    metrics["type"] = Type
    # add t-stat wrt MSE with vol
    diff = ret_base[2] - ret[2]
    tstat = diff.groupby("region").mean() / diff.groupby("region").apply(lambda x: NW_std(x.values))
    tstat = tstat * np.sqrt(diff.groupby("region").count())
    metrics = metrics.merge(pd.DataFrame({"t-stat": tstat}), on="region")
    res += [metrics]
res = pd.concat(res)
res = res.loc[res["region"] == "USA"].copy()
del res["region"]
res.rename({"ls_mean": "Avg Ret", "ls_SR": "SR"}, axis=1, inplace=True)
res["Avg Ret"] = res["Avg Ret"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
res["SR"] = res["SR"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
res["t-stat"] = res["t-stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
res = res.melt(id_vars=["type", "Spec"]).set_index(["type", "Spec", "variable"])
res_ = res["value"].unstack([0, 1]).reset_index()
res_["Empty"] = ""
res_ = pd.concat([res_[["variable"]], res_["Full"], res_["Empty"], res_["Liquid"]], axis=1)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "MSEvsQuantileUpdate1.txt"), index=False)


##### MSE R-squared vs median vs mean
# get raw returns that are being predicted
Cols = ["DTID", "date", "r", "region", "MC"]
ret = pd.read_parquet(os.path.join(sPath, "Features", "Monthly_ret.gzip"), columns=Cols)

# get risk-free rate
rf = pd.read_csv(os.path.join(sPath, "Inputs", "FF3.CSV"))
rf["date"] = rf["date"].astype(str)
rf["date"] = pd.to_datetime(rf["date"].str.slice(0, 4) + "-" + rf["date"].str.slice(4, 6) + "-01")
rf["RF"] = rf["RF"] / 100
rf = rf[["date", "RF"]].copy()
ret = ret.merge(rf, on="date")
ret["r"] = ret["r"] - ret["RF"]

res = []
for Type in ["Full", "Liquid"]:
    ## Mean computed from quantiles
    File = "NN_clean_skewness_full.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = pd.read_parquet(os.path.join(sPath, "Predict", File))
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["m1"] = dt["m1"] - dt["RF"]
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="m1", actual="r"),
            "Spec": "Two stage NN - Mean",
            "type": Type,
        }
    )
    res += [metrics]
    dt_bench = dt[["DTID", "date", "m1"]].copy()

    ## Median
    dt = GetPredictions(sPath, files, horizon="M", full_sample=False if Type == "Liquid" else True)
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    dt["pred_raw_0.5"] = dt["pred_raw_0.5"] - dt["RF"]
    dt = dt.merge(dt_bench, on=["DTID", "date"])
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="pred_raw_0.5", actual="r"),
            "DM stat": DieboldMariano_df(dt, pred1="pred_raw_0.5", pred2="m1", actual="r", AddGlobal=True),
            "Spec": "Two stage NN - Median",
            "type": Type,
        }
    )
    res += [metrics]

    ## MSE with vol and normalized predicted returns
    File = "NN_clean_MSE_NN2_raw_full_M_m.gzip"
    if Type == "Liquid":
        File = File.replace("_full", "")
    dt = GetPredictions(sPath, files, pred_file=File)
    dt = dt.loc[dt["date"] > "2018-12-31"].copy()
    dt = dt.merge(ret, on=["DTID", "date", "region"])
    dt["pred"] = dt["pred"] - dt["RF"]
    dt = dt.merge(dt_bench, on=["DTID", "date"])
    metrics = pd.DataFrame(
        {
            "R-sqrd": Rsqrd_df(dt, pred="pred", actual="r"),
            "DM stat": DieboldMariano_df(dt, pred1="pred", pred2="m1", actual="r", AddGlobal=True),
            "Spec": "2hNN MSE - Mean",
            "type": Type,
        }
    )
    res += [metrics]
res = pd.concat(res)
res = res.loc[res.index == "USA"].copy()
res["R-sqrd"] = res["R-sqrd"].transform(lambda x: "" if np.isnan(x) else f"{100*x:.2f}")
res["DM stat"] = res["DM stat"].transform(lambda x: "" if np.isnan(x) else f"({x:.2f})")
res = res.reset_index(drop=True).melt(id_vars=["type", "Spec"], var_name="Measure")
res = res.set_index(["type", "Spec", "Measure"])["value"].unstack([0, 1]).reset_index()
res["Empty"] = ""
res_ = pd.concat([res[["Measure"]], res["Full"], res["Empty"], res["Liquid"]], axis=1)
out = ToLaTeX(res_)
out.to_csv(os.path.join(sPath, "Output", "MSEvsQuantileUpdate2.txt"), index=False)
