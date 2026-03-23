import warnings
import os
import arch

import numpy as np
import pandas as pd
from tqdm import tqdm

from numba import njit, prange

# source anomalies
from Signals.SignalClass import SignalClass
from Signals.Market import IdiosyncraticRisk, MaxReturn, Volatility
from Signals.Volatility import EWMAVolatility, EWMAVolatilityD, Volatility3M, Volatility6M, Volatility12M

# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class BetaAndCoskewness(SignalClass):
    """
    Coskewness: Harvey and Siddique (2000)
    Beta: Fama and MacBeth (1973)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -61, "end": 0, "items": ["RI", "MC"]}}
        self.Output = ["Coskew", "Beta"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]

        new_index = [DM.Date - mnth * 22 for mnth in range(61)]
        index = pd.MultiIndex.from_product(
            [dt.index.get_level_values("DTID").unique(), new_index], names=["DTID", "date"]
        )
        DT = pd.DataFrame(index=index).reset_index()
        DT.set_index("date", inplace=True)
        DT.sort_index(inplace=True)
        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        DT = pd.merge_asof(DT, dt, on="date", by="DTID")
        DT.set_index(["DTID", "date"], inplace=True)
        DT.sort_index(inplace=True)
        DT["r"] = DT["RI"] / DT["RI"].groupby("DTID").shift(1) - 1
        del DT["RI"]
        DT.dropna(inplace=True)
        DT["ones"] = 1
        mkt = DT.groupby("date").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt})
        DT.reset_index(inplace=True)
        DT = DT.merge(mkt, on="date")

        def GetBetaCoskew(g):
            """
            function that computes loading on market returns
            """
            if g.shape[0] < 36:
                return pd.Series({"Beta": np.nan, "Coskew": np.nan}, index=["Beta", "Coskew"])
            else:
                y = g["r"].values
                X = g[["ones", "rm"]].values
                beta = np.linalg.pinv(X).dot(y)
                resid = y - X.dot(beta)
                resid2 = X[:, 1] - X[:, 1].mean()
                numer = (resid * resid2**2).mean()
                denom = np.sqrt((resid**2).mean()) * (resid2**2).mean()
                if denom != 0:
                    coskew = numer / denom
                else:
                    coskew = np.nan
                return pd.Series({"Beta": beta[1], "Coskew": coskew}, index=["Beta", "Coskew"])

        result = DT.groupby("DTID").apply(GetBetaCoskew)
        return result[Out]


class MarketBetaBAB(SignalClass):
    """
    Betting against Beta: Frazzini and Pedersen (2014)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -60, "end": 0, "items": ["r", "MC"]}}
        self.Output = ["BAB"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]

        dt["FirstYear"] = 0
        dt.loc[dt.index.get_level_values("date") > DM.Date - 12 * 22, "FirstYear"] = 1
        dt.dropna(inplace=True)
        dt["ones"] = 1
        mkt = dt.groupby("date").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt})
        dt.reset_index(inplace=True)
        dt = dt.merge(mkt, on="date")
        dt.set_index(["DTID", "date"], inplace=True)
        dt["r3"] = dt["r"] + dt["r"].groupby("DTID").shift(1) + dt["r"].groupby("DTID").shift(2)
        dt["rm3"] = dt["rm"] + dt["rm"].groupby("DTID").shift(1) + dt["rm"].groupby("DTID").shift(2)

        def GetBeta(g):
            """
            function that computes loading on market returns
            """
            if g.shape[0] < 250:
                return np.nan
            else:
                return g["rm3"].corr(g["r3"])

        BAB = (
            dt.groupby("DTID").apply(GetBeta)
            * dt.loc[dt["FirstYear"] == 1, "r"].groupby("DTID").std()
            / dt.loc[dt["FirstYear"] == 1, "rm"].groupby("DTID").std()
        )
        result = pd.DataFrame({"BAB": BAB})
        return result[Out]


class DataManager:
    """
    Main workhorse to fetch data to be used in the analysis
    """

    def InitDBs(self):
        """
        initialize individual datasets
        """

        # for market data
        if (self.Inputs == None) or ("dt" in self.Inputs.keys()):
            if self.Inputs == None:
                items = None
                # load 20 years of data as default
                start = self.Date + -241 * self.days_m
                end = self.Date
            else:
                items = self.Inputs["dt"]["items"] + ["DTID", "date"]
                start = self.Date + self.Inputs["dt"]["start"] * self.days_m
                end = self.Date + self.Inputs["dt"]["end"] * self.days_m

            dt = pd.read_parquet(os.path.join(self.sPath, "Simulation", self.dtName + ".gzip"), columns=items)
            dt = dt.loc[(dt.date < end) & (dt.date >= start)].copy()
            dt.set_index(["DTID", "date"], inplace=True)
            self.dt = dt
        else:
            self.dt = pd.DataFrame()

    def FetchDBs(self, DBs):
        """
        method to fetch DBs from in memory instance of DataManager so that SQL is not run each time
        """

        DBs.Date = self.Date
        FetchedData = DBs.fetch(self.Inputs)
        if "dt" in FetchedData.keys():
            self.dt = FetchedData["dt"]

    def fetch(self, Input):
        """
        callable function to provide required inputs
        """

        Output = {}
        if "dt" in Input.keys():
            start = self.Date + Input["dt"]["start"] * self.days_m
            end = self.Date + Input["dt"]["end"] * self.days_m
            dt = (
                self.dt.loc[
                    (self.dt.index.get_level_values("date") >= start)
                    & (self.dt.index.get_level_values("date") < end),
                    Input["dt"]["items"],
                ]
            ).copy()
            Output["dt"] = dt

        return Output

    def __init__(self, sPath, Date, Inputs=None, DBs=None):
        self.sPath = sPath
        self.days_m = 22
        self.dtName = "dt"
        if isinstance(Date, int):
            self.Date = Date
        else:
            raise Exception("Please provide date as int object.")
        self.Inputs = Inputs
        if DBs == None:
            self.InitDBs()
        elif isinstance(DBs, DataManager):
            self.FetchDBs(DBs)
        else:
            raise Exception("DBs must be either None or an instance of DataManager.")


def GetSignalClasses():
    """
    Collects all classes with implemented signals
    """

    return [
        "IdiosyncraticRisk",
        "MaxReturn",
        "Volatility",
        "EWMAVolatility",
        "EWMAVolatilityD",
        "Volatility3M",
        "Volatility6M",
        "Volatility12M",
        "MarketBetaBAB",
        "BetaAndCoskewness",
    ]


def ProcessSignalsDict(Signals=None):
    """
    creates SignalsDict, subsets it to the required signals, and provides required inputs
    """

    # get dictionary describing anomaly classes and anomalies within them
    SignalsDict = {Class: eval(Class + "().Output") for Class in GetSignalClasses()}

    # get reduced signal dictionary for the signals that we want
    if Signals == None:
        SignalsDictRed = SignalsDict
    else:
        SignalsDictRed = {
            Class: [Signal for Signal in ClassSignals if Signal in Signals]
            for Class, ClassSignals in SignalsDict.items()
            if any(elem in Signals for elem in ClassSignals)
        }

    # map signal classes that are required because of dependencies
    DependsOn = []
    for Class in SignalsDictRed.keys():
        DependsOn = DependsOn + eval(Class + "().DependsOn")
    SignalsDictRed2 = {key: SignalsDict[key] for key in set(list(SignalsDictRed.keys()) + DependsOn)}

    # get set of required inputs
    Inputs = {}
    for Class in SignalsDictRed2.keys():
        Input = eval(Class + "().Input")
        for db in Input.keys():
            if db in Inputs.keys():
                Inputs[db]["start"] = min(Inputs[db]["start"], Input[db]["start"])
                Inputs[db]["end"] = max(Inputs[db]["end"], Input[db]["end"])
                Inputs[db]["items"] = list(set(Inputs[db]["items"] + Input[db]["items"]))
            else:
                Inputs[db] = Input[db]

    return Inputs, SignalsDictRed


def CreateSignalsWorker(Date, sPath, Signals=None, DBs=None):
    """
    creates signals at the given T0 date and given signals
    """

    # get required inputs
    Inputs, SignalsDict = ProcessSignalsDict(Signals)

    # initiate data manager
    DM = DataManager(sPath, Date=Date, Inputs=Inputs, DBs=DBs)

    # get empty index with DTID to collect the results
    FetchedData = DM.fetch({"dt": {"start": -1, "end": 0, "items": []}})
    dt = FetchedData["dt"]
    dt.reset_index(inplace=True)
    JoinedSignals = dt[["DTID"]].groupby("DTID").last()
    JoinedSignals.reset_index(inplace=True)

    for Class, ClassSignals in SignalsDict.items():
        try:
            Signal = eval(Class + "().CreateSignal(DM, Out=ClassSignals)")
            if Signal.index.names == ["DTID"]:
                JoinedSignals = JoinedSignals.merge(Signal, on="DTID", how="left")
            else:
                raise Exception(Class + " is not correctly implemented. Expecte DTID or FTID index.")
        except:
            # if error add the signals as nan
            for Signal in ClassSignals:
                JoinedSignals[Signal] = np.nan

    JoinedSignals["date"] = DM.Date
    return JoinedSignals


def CreateSignals(DateSequence, sPath, Signals=None, InMemory=True):
    """
    function that processes the defined signals at each date of DateSequence and returns panel data with them
    inputs:
        DateSequence: list of dates at which to compute the signals
        sPath: location to master folder
        Signals: list of signals that should be computed
        InMemmory: will store all the Data in memory and fetch it from there for each time period, needs 64GB RAM
    """

    # preallocate all the data in memory if needed
    if InMemory == True:
        Inputs = {"dt": {"start": -1000000, "end": 0, "items": ["MC", "r", "RI"]}}
        DBs = DataManager(sPath, Date=100000, Inputs=Inputs)
    else:
        DBs = None

    Output = []
    for TimeIndex in tqdm(DateSequence):
        Output += [CreateSignalsWorker(Date=int(TimeIndex), sPath=sPath, Signals=Signals, DBs=DBs)]
    Output = pd.concat(Output)

    return Output


def CreateReturnsWorker(DateStart, DateEnd, DM):
    """
    creates returns at the given T0 date
    """

    DateDiff = np.ceil((DateEnd - DateStart) / 22) + 1
    DM.Date = DateStart
    FetchedData = DM.fetch({"dt": {"start": -1, "end": DateDiff, "items": ["RI"]}})
    dt = FetchedData["dt"]
    dt2 = dt.loc[dt.index.get_level_values("date") < DateEnd].copy()
    dt = dt.loc[dt.index.get_level_values("date") < DateStart].copy()
    dt.rename(columns={"RI": "RI_lag"}, inplace=True)
    dt = dt.groupby("DTID").last()
    dt2 = dt2.groupby("DTID").last()
    dt = dt.join(dt2)
    dt["r"] = dt["RI"] / dt["RI_lag"] - 1
    dt["date"] = DM.Date
    dt.reset_index(inplace=True)
    return dt[["DTID", "date", "r"]]


def CreateReturns(DateSequenceStart, DateSequenceEnd, sPath):
    """
    function that processes the future returns at each date of DateSequence and returns panel data with them
    inputs:
        DateSequenceStart: list of dates at which the return starts
        DateSequenceEnd:  list of dates at which the return ends
        sPath: location to master folder
    """

    # use the auxiliary function CreateReturnsWorker to compute returns at each time index in DateSequence
    if len(DateSequenceStart) != len(DateSequenceEnd):
        raise Exception("DateSequenceStart must be of the same length as DateSequenceEnd.")

    Output = pd.DataFrame()

    # preallocate all the data in memory
    DM = DataManager(
        sPath,
        Date=100000,
        Inputs={"dt": {"start": -1000000, "end": 0, "items": ["RI"]}},
    )

    print(f"Number of periods to be processed: {len(DateSequenceStart)}")
    Output = []
    for TimeStart, TimeEnd in tqdm(zip(DateSequenceStart, DateSequenceEnd)):
        ret = CreateReturnsWorker(DateStart=int(TimeStart), DateEnd=int(TimeEnd), DM=DM)
        Output += [ret]
    Output = pd.concat(Output)
    return Output


def MLdata(Ret, Signals, exc_normalize=[]):
    """
    this function connects returns plus signals and formats everything for ML application
    """

    data = pd.merge(Ret, Signals, on=["date", "DTID"], how="inner")
    del Ret, Signals

    # normalize all the signals to their cross-sectional ranks
    signals = [item for item in data.columns if item not in ["DTID", "date", "r", "MC"] + exc_normalize]
    for signal in signals:
        data[signal] = data.groupby(["date"])[signal].rank(pct=True)
        data[signal] = data[signal].fillna(0.5)
    return data


def MktMeanRet(sPath):

    DM = DataManager(
        sPath,
        Date=100000,
        Inputs={"dt": {"start": -1000000, "end": 0, "items": ["r"]}},
    )

    FetchedData = DM.fetch({"dt": {"start": -1000000, "end": 0, "items": ["r"]}})
    dt = FetchedData["dt"].copy()
    dt.dropna(subset="r", inplace=True)
    mkt = pd.DataFrame({"EW": dt.groupby("date")["r"].mean()})
    VarNames = []
    for decay in [10, 6, 4, 1, 0.1]:
        VarName = f"MktAvg{decay}_EW"
        mkt[VarName] = mkt["EW"].ewm(alpha=decay / 100, adjust=False).mean()
        VarNames += [VarName]
    res = mkt[VarNames]
    res = res.reset_index()
    res["date"] = res["date"].shift(-1)
    res = res.dropna()
    res["date"] = res["date"].astype(int)

    return res


@njit()
def simulate_r_t(omega, alpha, beta, vol_init, nu, h):
    inov = np.empty(h)
    vol = np.empty(h)
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0] = vol_init
    inov[0] = np.random.standard_t(nu) / t_std
    for i in range(1, h):
        inov[i] = np.random.standard_t(nu) / t_std
        vol[i] = min(omega + alpha * (inov[i - 1] ** 2 * vol[i - 1]) + beta * vol[i - 1], 2500)
    r = (inov * np.sqrt(vol)) / 100
    return r, np.sqrt(vol), inov


@njit(parallel=True)
def vol_bootstrap_worker_t(omega, alpha, beta, vol_init, inov_init, nu, h, draws=100000):
    r = np.empty((h, draws))
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = min(omega + alpha * (inov[i - 1, j] ** 2 * vol[i - 1, j]) + beta * vol[i - 1, j], 2500)
        r[:, j] = (inov[1:, j] * np.sqrt(vol[1:, j])) / 100
    return r


@njit()
def simulate_gjr_r_t(omega, alpha, beta, gamma, vol_init, nu, h):
    inov = np.empty(h)
    vol = np.empty(h)
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0] = vol_init
    inov[0] = np.random.standard_t(nu) / t_std
    for i in range(1, h):
        inov[i] = np.random.standard_t(nu) / t_std
        vol[i] = min(
            omega + (alpha + (inov[i - 1] < 0) * gamma) * np.square(inov[i - 1]) * vol[i - 1] + beta * vol[i - 1],
            2500,
        )
    r = (inov * np.sqrt(vol)) / 100
    return r, np.sqrt(vol), inov


@njit(parallel=True)
def vol_bootstrap_worker_gjr_t(omega, alpha, beta, gamma, vol_init, inov_init, nu, h, draws=100000):
    r = np.empty((h, draws))
    inov = np.empty((h + 1, draws))
    vol = np.empty((h + 1, draws))
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    vol[0, :] = vol_init
    inov[0, :] = inov_init
    for j in prange(draws):
        for i in range(1, h + 1):
            inov[i, j] = np.random.standard_t(nu) / t_std
            vol[i, j] = min(
                omega
                + (alpha + (inov[i - 1, j] < 0) * gamma) * np.square(inov[i - 1, j]) * vol[i - 1, j]
                + beta * vol[i - 1, j],
                2500,
            )
        r[:, j] = (inov[1:, j] * np.sqrt(vol[1:, j])) / 100
    return r


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


@njit()
def simulate_jump_t(nu, P, std, h):
    r = np.zeros(h)
    t_std = np.sqrt(nu / (nu - 2))  # standard deviation of t dist
    for i in prange(h):
        if np.random.uniform() < P:
            r[i] = np.random.standard_t(nu) / t_std * std
    # r[r > 0] *= 2  # larger positive jumps
    return r


def get_data(
    sPath,
    feature_file,
    vol_vars,
    mkt_mean_vars,
    rescale_mean=True,
    adjust_r="divide",
    r_scale=0.11,
):
    ## load data with features
    data = pd.read_parquet(os.path.join(sPath, "Simulation", feature_file))

    # rescale volatility variables that are not standardized
    data.reset_index(drop=True, inplace=True)
    for Var in vol_vars:
        data.loc[data[Var].isnull(), Var] = data.groupby(["date"])[Var].transform("mean")
        data[Var + "_raw"] = data[Var] / 0.022
        data[Var] = data[Var] / data.groupby("date")[Var].transform("mean")
    data["r_scale"] = data.groupby(["date"])["EWMAVol6_raw"].transform("mean") * r_scale

    # normalize returns
    data["r_raw"] = data["r"]
    if adjust_r == "divide":
        data["r"] = data["r"] / data["r_scale"]
    elif adjust_r == "standardize":
        data["r"] = data["r"] - data.groupby(["date"])["r"].transform("median")
        data["r_scale"] = data.groupby(["date"])["r"].transform(lambda x: np.abs(x).mean()) / 1.4
        data["r"] = data["r"] / data["r_scale"]
    data = data.copy()

    # create cross-sectional mean volatility variables
    for Var in vol_vars:
        data[Var + "_mean"] = data.groupby(["date"])[Var + "_raw"].transform("mean")

    # add historical average market returns
    MktMean = pd.read_parquet(os.path.join(sPath, "Simulation", "Mkt_mean.gzip"), columns=["date"] + mkt_mean_vars)
    MktMean.sort_values("date", inplace=True)
    data.sort_values("date", inplace=True)
    data = pd.merge_asof(data, MktMean, on="date")
    data.reset_index(drop=True, inplace=True)
    if rescale_mean:
        for Var in mkt_mean_vars:
            data[Var] = data[Var] / data["r_scale"]

    return data


def RMSE_df_sim(dt, pred, actual, GrpVar=None):
    def RMSE(x, y):
        return np.sqrt(((x - y) ** 2).mean())

    if GrpVar is not None:
        out = dt.groupby(GrpVar).apply(lambda x: RMSE(x[actual], x[pred]))
    else:
        out = RMSE(dt[actual], dt[pred])

    return out


def MAD_df_sim(dt, pred, actual, GrpVar=None):
    def MAD(x, y):
        return (np.abs(x - y)).mean()

    if GrpVar is not None:
        out = dt.groupby(GrpVar).apply(lambda x: MAD(x[actual], x[pred]))
    else:
        out = MAD(dt[actual], dt[pred])

    return out
