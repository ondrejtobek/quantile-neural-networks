import pandas as pd
import numpy as np
import os
import seaborn

from EstimationFunctions.NN_Functions import *
from tqdm import tqdm
from scipy.stats import norm, t, skewnorm, nct
import matplotlib.pyplot as plt
import statsmodels.api as sm


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


##### adjustments for integrated moments to better fit actual distribution
### run simpulation
n = 1000000
nc_full = np.random.uniform(low=-0.5, high=5.0, size=n)
df_out = np.array(list(range(5, 21)) + list(range(30, 101, 10)) + [1000, 10000])
df_full = np.random.choice(df_out, size=n, replace=True)
y = np.array(taus)
res = []
for nc, df in tqdm(zip(nc_full, df_full)):
    x = nct.ppf(taus, df, nc, scale=0.1)
    res_ = GetMoments(x, y)
    theo = nct.stats(df, nc, moments="mvsk", scale=0.1)
    res_["mean_t"] = theo[0]
    res_["var_t"] = theo[1]
    res_["skewness_t"] = theo[2]
    res_["kurtosis_t"] = theo[3]
    res += [res_]
res = pd.concat(res)
res.to_parquet(os.path.join(sPath, "Predict", "Simulate_moments.gzip"), compression="gzip")

res = pd.read_parquet(os.path.join(sPath, "Predict", "Simulate_moments.gzip"))
res = res.loc[res["kurtosis_t"] < 17].copy()
res["excess_kurtosis"] = res["kurtosis"] - 3
res["skew_sqr"] = res["skew"] ** 2
res["excess_kurtosis_sqr"] = res["excess_kurtosis"] ** 2

# variance
res["ratio"] = res["var_t"] / res["var"]
y = res["ratio"].values
X = sm.add_constant(res[["skew", "excess_kurtosis"]].values)
model = sm.OLS(y, X)
results = model.fit()
results.summary()
results.params

# skewness
y = (res["skewness_t"] - res["skew"]).values
X = res[["skew", "skew_sqr", "excess_kurtosis"]].values
model = sm.OLS(y, X)
results = model.fit()
results.summary()
results.params

# kurtosis
y = (res["kurtosis_t"] - res["excess_kurtosis"]).values
X = res[["excess_kurtosis", "excess_kurtosis_sqr", "skew"]].values
model = sm.OLS(y, X)
results = model.fit()
results.summary()
results.params


## test the distribution
# note that the theoretical kurtosis is expressed as excess
def theoretical_mom(x, name):
    return pd.DataFrame({"Dist": name, "t1": x[0], "t2": x[1], "t3": x[2], "t4": x[3] + 3}, index=[0])


y = np.array(taus)
Vars = ["m1", "var", "skew", "kurtosis", "var_adj", "skew_adj", "kurtosis_adj"]
res = []
# normal
x = norm.ppf(taus, scale=0.1)
res_ = theoretical_mom(norm.stats(moments="mvsk", scale=0.1), "Normal")
res_ = pd.concat([res_, AdjustMoments(GetMoments(x, y))[Vars]], axis=1)
res += [res_]

# t dist
for df in [5, 6, 10]:
    x = t.ppf(taus, df, scale=0.1)
    res_ = theoretical_mom(t.stats(df, moments="mvsk", scale=0.1), f"t: df={df}")
    res_ = pd.concat([res_, AdjustMoments(GetMoments(x, y))[Vars]], axis=1)
    res += [res_]

# skewed t
for df, nc in [[5, 1], [6, 3], [5, 4]]:
    x = nct.ppf(taus, df, nc, scale=0.1)
    res_ = theoretical_mom(nct.stats(df, nc, moments="mvsk", scale=0.1), f"nct: df={df}, nc={nc}")
    res_ = pd.concat([res_, AdjustMoments(GetMoments(x, y))[Vars]], axis=1)
    res += [res_]

res = pd.concat(res)
res.rename(
    {
        "m1": "m1",
        "var": "m2",
        "skew": "m3",
        "kurtosis": "m4",
        "var_adj": "a2",
        "skew_adj": "a3",
        "kurtosis_adj": "a4",
    },
    axis=1,
    inplace=True,
)
for Var in ["t2", "m2", "a2"]:
    res[Var] = res[Var] * 100
for Var in [Var for Var in res.columns if Var != "Dist"]:
    res[Var] = res[Var].transform(lambda x: "" if np.isnan(x) else f"{x:.3f}")
res = res[["Dist", "t1", "m1", "t2", "m2", "a2", "t3", "m3", "a3", "t4", "m4", "a4"]].copy()
res.to_latex().replace("SpaceHolder", "")  # for headers
res = ToLaTeX(res)
res.to_csv(os.path.join(sPath, "Output", "MomentAdjustment1.txt"), index=False, sep="\t")


##### density estimation - compute moments
## NN
Keep_vars = [
    "date",
    "DTID",
    "region",
    "m0",
    "m1",
    "m2",
    "m3",
    "m4",
    "var",
    "std",
    "skew",
    "kurtosis",
    "LinearFlag",
    "Error",
]
# full sample
dt = GetPredictions(sPath, files, horizon="M", full_sample=True)
dt = ComputeMoments(dt, taus)
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_full.gzip"), compression="gzip")

# liquid sample
dt = GetPredictions(sPath, files, horizon="M", full_sample=False)
dt = ComputeMoments(dt, taus)
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness.gzip"), compression="gzip")

## linear
# full sample
dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN0_full_M_m.gzip")
dt.rename({f"pred_{tau}": f"pred_raw_{tau}" for tau in taus}, axis=1, inplace=True)
dt = ComputeMoments(dt, taus)
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_linear_full.gzip"), compression="gzip")

# liquid sample
dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN0_M_m.gzip")
dt.rename({f"pred_{tau}": f"pred_raw_{tau}" for tau in taus}, axis=1, inplace=True)
dt = ComputeMoments(dt, taus)
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_linear.gzip"), compression="gzip")

## 2hNN
# full sample
dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN2_full_M_m.gzip")
dt = ComputeMoments(dt, taus, pred_col="pred")
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_NN2_full.gzip"), compression="gzip")

# liquid sample
dt = GetPredictions(sPath, files, pred_file="NN_clean_onestage_NN2_M_m.gzip")
dt = ComputeMoments(dt, taus, pred_col="pred")
dt = dt[Keep_vars]
dt.to_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_NN2.gzip"), compression="gzip")


##### create table and plots for moments
KeepVars = ["date", "DTID", "m1", "std", "skew", "kurtosis", "std_adj", "skew_adj", "kurtosis_adj"]
dt1 = pd.read_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness.gzip"))
dt1 = AdjustMoments(dt1)
dt1["std_adj"] = np.sqrt(dt1["var_adj"])
dt1 = dt1.loc[dt1["region"] == "USA", KeepVars].copy()
dt1["Type"] = "Liquid"
dt2 = pd.read_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_full.gzip"))
dt2 = AdjustMoments(dt2)
dt2["std_adj"] = np.sqrt(dt2["var_adj"])
dt2 = dt2.loc[dt2["region"] == "USA", KeepVars].copy()
dt2["Type"] = "Full"
dt = pd.concat([dt1, dt2])
dt.rename(
    {
        "m1": "Mean",
        "std": "Std. Dev.",
        "skew": "Skewness",
        "kurtosis": "Kurtosis",
        "std_adj": "Std. Dev. Adj",
        "skew_adj": "Skewness Adj",
        "kurtosis_adj": "Kurtosis Adj",
    },
    axis=1,
    inplace=True,
)
res = dt.melt(
    id_vars=["date", "DTID", "Type"],
    value_vars=["Mean", "Std. Dev.", "Std. Dev. Adj", "Skewness", "Skewness Adj", "Kurtosis", "Kurtosis Adj"],
    var_name="Variable",
    value_name="Value",
)
res = res.groupby(["Variable", "Type"])["Value"].agg(["mean", "median", "std"]).reset_index()
res["mean"] = res["mean"].transform(lambda x: f"{x:0.4f}")
res["median"] = res["median"].transform(lambda x: f"{x:0.4f}")
res["std"] = res["std"].transform(lambda x: f"{x:0.4f}")
res = res.melt(id_vars=["Variable", "Type"], var_name="Stat")
res = res.sort_values(["Type", "Stat"]).set_index(["Variable", "Stat", "Type"]).unstack([2, 1])
res = res.loc[["Mean", "Std. Dev.", "Std. Dev. Adj", "Skewness", "Skewness Adj", "Kurtosis", "Kurtosis Adj"]]
res["Empty"] = ""
res = res.reset_index()
res = pd.concat([res.iloc[:, :4], res["Empty"], res.iloc[:, 4:7]], axis=1)
out = ToLaTeX(res)
out.to_csv(os.path.join(sPath, "Output", "Moment_summary_stat.txt"), index=False)

### histogram plots
color2 = "#1D91C0"
color1 = "#C7E9B4"
NameDict = {"Mean": "Mean", "Std. Dev.": "Std. Dev.", "Skewness Adj": "Skewness", "Kurtosis Adj": "Kurtosis"}
fig = plt.figure(figsize=(24, 6))
for i, Var in enumerate(["Mean", "Std. Dev.", "Skewness Adj", "Kurtosis Adj"]):
    ax = plt.subplot(1, 4, i + 1)
    # ax.set_aspect('equal')
    for h in dt["Type"].unique():
        dt_ = dt.loc[
            (
                (dt["Type"] == h)
                & (dt[Var] < dt.groupby("Type")[Var].transform("quantile", 0.995))
                & (dt[Var] > dt.groupby("Type")[Var].transform("quantile", 0.005))
            )
        ].copy()
        if h == "Liquid":
            color = color2
        elif h == "Full":
            color = color1
        dt_[Var].hist(density=True, bins=100, ax=ax, label=h, alpha=0.7, color=color)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlabel(NameDict[Var], fontsize=14)
    ax.set_title(NameDict[Var], fontsize=16)
    ax.grid(False)
    if i == 3:
        ax.legend()
fig.tight_layout()
# fig.savefig(os.path.join(sPath, "Output", "Moment_hist_corrected_1_4.png"))
fig.savefig(os.path.join(sPath, "Output", "Moment_hist_corrected_1_4.pdf"))
plt.close()


##### variable correlations
def CreateVariableCorr(regions=["USA"], CorVars=["pred_0.5", "m1", "var_adj", "skew_adj", "kurtosis_adj"]):
    res = []
    for Type in ["Full", "Liquid"]:
        if Type == "Liquid":
            dt = GetPredictions(sPath, files, horizon="M", full_sample=False)
            dt2 = pd.read_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness.gzip"))
        else:
            dt = GetPredictions(sPath, files, horizon="M", full_sample=True)
            dt2 = pd.read_parquet(os.path.join(sPath, "Predict", "NN_clean_skewness_full.gzip"))
        dt = dt.merge(dt2, on=["DTID", "date", "region"])
        dt = AdjustMoments(dt)
        dt = dt.loc[dt["region"].isin(regions)].copy()
        corr = []
        for date in dt["date"].unique():
            corr_ = dt.loc[dt["date"] == date, CorVars].corr(method="spearman").melt(ignore_index=False)
            corr_ = corr_.reset_index().rename({"index": "Var1", "variable": "Var2"}, axis=1)
            corr_["date"] = date
            corr += [corr_]
        corr = pd.concat(corr)
        corr = corr.groupby(["Var1", "Var2"])["value"].mean().reset_index()
        corr["type"] = Type
        res += [corr]
    res = pd.concat(res)
    res["value"] = res["value"].transform(lambda x: "" if np.isnan(x) else f"{x:.2f}")
    res_ = res.set_index(["Var1", "Var2", "type"])["value"].unstack([2, 1]).loc[CorVars].reset_index()
    res_["Var1"] = res_["Var1"].map(
        {
            "pred_0.5": "Median",
            "m1": "Mean",
            "var_adj": "Variance",
            "skew_adj": "Skewness",
            "kurtosis_adj": "Kurtosis",
        }
    )
    res_["Empty"] = ""
    res_ = pd.concat([res_[["Var1"]], res_["Full"][CorVars], res_["Empty"], res_["Liquid"][CorVars]], axis=1)
    return res_


# cross-sectional correlation of variables for the US
out = CreateVariableCorr()
out = ToLaTeX(out)
out.to_csv(os.path.join(sPath, "Output", "VariableCorrelations.txt"), index=False)

# cross-sectional correlation of variables for other regions
out = CreateVariableCorr(regions=["Europe"])
out = ToLaTeX(out)
out.to_csv(os.path.join(sPath, "Output", "VariableCorrelationsEurope.txt"), index=False)

out = CreateVariableCorr(regions=["Asia Pacific"])
out = ToLaTeX(out)
out.to_csv(os.path.join(sPath, "Output", "VariableCorrelationsAsiaPacific.txt"), index=False)

out = CreateVariableCorr(regions=["Japan"])
out = ToLaTeX(out)
out.to_csv(os.path.join(sPath, "Output", "VariableCorrelationsJapan.txt"), index=False)
