### This scipt provides minimum example to run the champion model on liquid universe of stocks
# only in the US using CRSP / Compustat

import os
import numpy as np
import pandas as pd
import seaborn

from tqdm import tqdm
import matplotlib.pyplot as plt

# source data manager and data processing
from DataModules.ProcessData import *

# source function for creation of signals
from Signals.CreateSignals import *

# source functions for estimation of neural nets
from EstimationFunctions.NN_Functions import (
    get_anomalies_list,
    validation_logic,
    get_data,
    train_loop,
    ComputeMoments,
    AdjustMoments,
    DensityIntegrationPlots,
)


# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")

np.random.seed(123)
sPath = "./Data"

####################################################################################################
########################################## Create Data #############################################

##### Raw data processing
## process raw data into Parquet files or SQL database
ProcessData(sPath, RunDB=["CRSP", "Compustat", "IBESsum", "IBESdet"])
# hopefully should run without Datastream data with dummy file for DSTstatic.csv


##### Create signals
## create monthly signals
os.makedirs(os.path.join(sPath, "Features"), exist_ok=True)
# create signals for CRSP + Compustat
Signals = CreateSignals(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-01"),
    sPath=sPath,
    source="WRDS",
)
Signals.to_parquet(os.path.join(sPath, "Features", "WRDS_signals.gzip"), compression="gzip")

## create weekly signals
# create signals for CRSP + Compustat
Signals = CreateSignals(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-31", freq="7D"),
    sPath=sPath,
    source="WRDS",
)
Signals.to_parquet(os.path.join(sPath, "Features", "WRDS_signals_W.gzip"), compression="gzip")


##### Create returns
## create next month returns
# create returns for CRSP + Compustat
Ret = CreateReturns(
    DateSequenceStart=TimeSequence(start="1973-01-01", end="2023-12-01"),
    DateSequenceEnd=TimeSequence(start="1973-02-01", end="2024-01-01"),
    sPath=sPath,
    source="WRDS",
)
Ret.to_parquet(os.path.join(sPath, "Features", "WRDS_returns.gzip"), compression="gzip")

## create 22 business days returns
# create returns for CRSP + Compustat
Ret = CreateReturns(
    DateSequenceStart=TimeSequence(start="1973-01-01", end="2023-12-25", freq="7D"),
    DateSequenceEnd=TimeSequence(start="1973-01-31", end="2024-01-25", freq="7D"),
    sPath=sPath,
    source="WRDS",
)
Ret.to_parquet(os.path.join(sPath, "Features", "WRDS_returns_22d.gzip"), compression="gzip")


##### Create universe filters
### Liquid data
## universe filters for monthly data
# create filters for CRSP + Compustat
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-01"),
    sPath=sPath,
    source="WRDS",
)
Universe.to_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter.gzip"), compression="gzip")

## universe filters for weekly updated data
# create filters for CRSP + Compustat
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-31", freq="7D"),
    sPath=sPath,
    source="WRDS",
)
Universe.to_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W.gzip"), compression="gzip")


##### Connect together returns, features, and subset the universe
# don't rescale volatility variables
vol_vars = (
    [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMARange{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMAVolD{i}" for i in [20, 10, 6]]
    + ["TV3M", "TV6M", "TV12M"]
)
### Liquid sample
## monthly data
# CRSP + Compustat
Signals = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter.gzip"))
dataCRSP = MLdata(Ret, Signals, Universe, source="WRDS", exc_normalize=vol_vars)
dataCRSP.to_parquet(os.path.join(sPath, "Features", "MLdata.gzip"), compression="gzip")

## Weekly data with 22d ahead returns
# just CRSP + Compustat
Signals = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals_W.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns_22d.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W.gzip"))
dataCRSP = MLdata(Ret, Signals, Universe, source="WRDS", exc_normalize=vol_vars)
dataCRSP.to_parquet(os.path.join(sPath, "Features", "MLdata_W_22d.gzip"), compression="gzip")

##### equally-weighted market returns with different moving average filter
ret = MktMeanRet(sPath, regions=["USA"])
ret.to_parquet(os.path.join(sPath, "Features", "Mkt_mean.gzip"), compression="gzip")


####################################################################################################
########################################## Run Estimation ##########################################

# define variables to be used generally
anomalies = get_anomalies_list(sPath)
vol_vars = (
    [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMARange{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMAVolD{i}" for i in [20, 10, 6]]
    + ["TV3M", "TV6M", "TV12M"]
)
mkt_mean_vars = ["MktAvg10_EW", "MktAvg6_EW", "MktAvg4_EW", "MktAvg1_EW", "MktAvg0.1_EW"]
taus = (
    [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.075]
    + [0.925, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99995]
    + [i / 20 for i in range(1, 20)]
)
taus.sort()

### quantile regression two-stage with bottleneck
# set up parameters
sample_split = validation_logic()
inputs1 = anomalies + vol_vars + mkt_mean_vars
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
    "num_networks": 20,
}

# get data
data = get_data(sPath, "MLdata_W_22d.gzip", vol_vars, mkt_mean_vars, regions=["USA"])
data_m = get_data(sPath, "MLdata.gzip", vol_vars, mkt_mean_vars)

# run estimation and create predicitons
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, inputs2)

# save output with monthly forecast and weekly forecasts
pred.to_parquet(os.path.join(sPath, "Predict", "NN_quantile_reg_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_quantile_reg_M_m.gzip"), compression="gzip")


####################################################################################################
########################################## Derive Moments ##########################################
### density estimation - compute moments
# get predictions
dt = pd.read_parquet(os.path.join(sPath, "Predict", "NN_quantile_reg_M_m.gzip"))

# compute moments without adjustment
dt = ComputeMoments(dt, taus)

# adjust the moments for imperfections in tails
dt = AdjustMoments(dt)

# clean up the results
Keep_vars = [
    "date",
    "DTID",
    "m0",  # integrated density - normalizing constant so that the probability is equal to 1
    "m1",  # non-central moments
    "m2",
    "m3",
    "m4",
    "var",  # central moments
    "std",
    "skew",
    "kurtosis",
    "LinearFlag",
    "Error",
]
dt = dt[Keep_vars]

# see summary statistics
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
fig.savefig(os.path.join(sPath, "Output", "Moment_hist_corrected_1_4.png"))
plt.close()

####################################################################################################
############################## Plot derived distribution forecasts #################################

##### example plot for one ISIN
## plot of quantiles for individual ISINs
# need to remove the last digit
"02079K10"  # google -> 14542
"03783310"  # apple -> 14593
"59491810"  # microsoft -> 10107

VarMap = {f"pred_raw_{tau}": f"{tau} Quantile" for tau in [0.01, 0.1, 0.5, 0.9, 0.99]}
VarMap["r_raw"] = "22d-ahead Return"

dt = pd.read_parquet(os.path.join(sPath, "Predict", "NN_quantile_reg_M_m.gzip"))
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
dt = pd.read_parquet(os.path.join(sPath, "Predict", "NN_quantile_reg_M_m.gzip"))
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
plt.savefig("data/Output/cdf_pdf_msft_corrected.png")
