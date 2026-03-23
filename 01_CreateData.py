import os

import fastparquet
import numpy as np
import pandas as pd

# source data manager and data processing
from DataModules.ProcessData import *

# source function for creation of signals
from Signals.CreateSignals import *

# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")

np.random.seed(123)
sPath = "./Data"


##### Raw data processing
## process raw data into Parquet files or SQL database
ProcessData(sPath)


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
# create signals for DST
Signals = CreateSignals(
    DateSequence=TimeSequence(start="1990-01-01", end="2018-12-01"),
    sPath=sPath,
    source="DST",
    region=["Japan", "Europe", "Asia Pacific"],
)
Signals.to_parquet(os.path.join(sPath, "Features", "DST_signals.gzip"), compression="gzip")

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
# create returns for DST
Ret = CreateReturns(
    DateSequenceStart=TimeSequence(start="1990-01-01", end="2018-12-01"),
    DateSequenceEnd=TimeSequence(start="1990-02-01", end="2019-01-01"),
    sPath=sPath,
    source="DST",
    region=["Japan", "Europe", "Asia Pacific"],
)
Ret.to_parquet(os.path.join(sPath, "Features", "DST_returns.gzip"), compression="gzip")

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
# create filters for DST
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1990-01-01", end="2018-12-01"),
    sPath=sPath,
    source="DST",
    region=["Japan", "Europe", "Asia Pacific"],
)
Universe.to_parquet(os.path.join(sPath, "Features", "DST_universe_filter.gzip"), compression="gzip")

## universe filters for weekly updated data
# create filters for CRSP + Compustat
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-31", freq="7D"),
    sPath=sPath,
    source="WRDS",
)
Universe.to_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W.gzip"), compression="gzip")

### Full data
## universe filters for monthly data
# create filters for CRSP + Compustat
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-01"),
    sPath=sPath,
    source="WRDS",
    MClim=0.0,
    VOLlim=0.0,
)
Universe.to_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_full.gzip"), compression="gzip")
# create filters for DST
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1990-01-01", end="2018-12-01"),
    sPath=sPath,
    source="DST",
    region=["Japan", "Europe", "Asia Pacific"],
    MClim=0.0,
    VOLlim=0.0,
)
Universe.to_parquet(os.path.join(sPath, "Features", "DST_universe_filter_full.gzip"), compression="gzip")

## universe filters for weekly updated data
# create filters for CRSP + Compustat
Universe = UniverseFilter(
    DateSequence=TimeSequence(start="1973-01-01", end="2023-12-31", freq="7D"),
    sPath=sPath,
    source="WRDS",
    MClim=0.0,
    VOLlim=0.0,
)
Universe.to_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W_full.gzip"), compression="gzip")


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
# DST
Signals = pd.read_parquet(os.path.join(sPath, "Features", "DST_signals.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "DST_returns.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "DST_universe_filter.gzip"))
dataDST = MLdata(Ret, Signals, Universe, source="DST", exc_normalize=vol_vars)
# connect together
dataDST = dataDST.loc[dataDST["region"] != "North America"].copy()
data = pd.concat([dataCRSP, dataDST])
data.to_parquet(os.path.join(sPath, "Features", "MLdata.gzip"), compression="gzip")

## Weekly data with 22d ahead returns
# just CRSP + Compustat
Signals = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals_W.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns_22d.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W.gzip"))
dataCRSP = MLdata(Ret, Signals, Universe, source="WRDS", exc_normalize=vol_vars)
dataCRSP.to_parquet(os.path.join(sPath, "Features", "MLdata_W_22d.gzip"), compression="gzip")

### full sample
## monthly data
# DST
Signals = pd.read_parquet(os.path.join(sPath, "Features", "DST_signals.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "DST_returns.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "DST_universe_filter_full.gzip"))
dataDST = MLdata(Ret, Signals, Universe, source="DST", exc_normalize=vol_vars)
dataDST = dataDST.loc[dataDST["region"] != "North America"].copy()
# CRSP + Compustat
Signals = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_full.gzip"))
dataCRSP = MLdata(Ret, Signals, Universe, source="WRDS", exc_normalize=vol_vars)
# connect together
data = pd.concat([dataCRSP, dataDST])
data = data.reset_index(drop=True)
data.to_parquet(os.path.join(sPath, "Features", "MLdata_full.gzip"), compression="gzip")

## Weekly data with 22d ahead returns
# just CRSP + Compustat
Signals = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_signals_W.gzip"))
Ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns_22d.gzip"))
Universe = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_W_full.gzip"))
dataCRSP = MLdata(Ret, Signals, Universe, source="WRDS", exc_normalize=vol_vars)
dataCRSP.to_parquet(os.path.join(sPath, "Features", "MLdata_W_22d_full.gzip"), compression="gzip")


##### equally-weighted market returns with different moving average filter
ret = MktMeanRet(sPath, regions=["USA", "Europe", "Japan", "Asia Pacific"])
ret.to_parquet(os.path.join(sPath, "Features", "Mkt_mean.gzip"), compression="gzip")


##### create returns and market cap for portfolio analysis
start_date = "1995-01-01"
end_date = "2023-12-31"
### monthly
# get returns
ret = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_returns.gzip"))
ret_DST = pd.read_parquet(os.path.join(sPath, "Features", "DST_returns.gzip"))
ret = pd.concat([ret, ret_DST])
# get region and MC
reg_cols = ["DTID", "date", "region", "MC"]
reg = pd.read_parquet(os.path.join(sPath, "Features", "WRDS_universe_filter_full.gzip"), columns=reg_cols)
reg_DST = pd.read_parquet(os.path.join(sPath, "Features", "DST_universe_filter_full.gzip"), columns=reg_cols)
reg = pd.concat([reg, reg_DST])
# merge together
ret = ret.merge(reg, on=["DTID", "date"])
ret = ret.loc[(ret["date"] >= start_date) & (ret["date"] <= end_date)].copy()
ret.to_parquet(os.path.join(sPath, "Features", "Monthly_ret.gzip"), compression="gzip")

### daily
ret = GetDailyRetMC(sPath, start_date, end_date, regions=["USA", "Europe", "Japan", "Asia Pacific"])
ret.to_parquet(os.path.join(sPath, "Features", "Daily_ret.gzip"), compression="gzip")


##### create one month ahead volatility
result = GetFutureVola1M(sPath, start_date="1995-01-01", end_date="2023-12-01")
result.to_parquet(os.path.join(sPath, "Predict", "Volatility_m.gzip"), compression="gzip")
