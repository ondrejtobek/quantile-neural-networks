import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# source data manager and data processing
from DataModules.DataManager import *

# source anomalies
from Signals.Fundamental import *
from Signals.IBES import *
from Signals.Market import *
from Signals.Volatility import *

# suppress numpy warnings - divide by zero, np.log() etc.
np.seterr(divide="ignore", invalid="ignore")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def TimeSequence(start, end, freq="MS"):
    """
    takes the same inputs as pd.date_range() and transforms them in list of YYYY-MM-DD dates
    """

    return [
        Date.strftime("%Y-%m-%d")
        for Date in pd.date_range(start=start, end=end, freq=freq).to_pydatetime().tolist()
    ]


def GetSignalClasses():
    """
    Collects all classes with implemented signals
    """

    return [cls.__name__ for cls in SignalClass.__subclasses__()]


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


def CreateSignalsWorker(Date, sPath, source="DST", region=None, Signals=None, DBs=None):
    """
    creates signals at the given T0 date and given signals
    """

    # get required inputs
    Inputs, SignalsDict = ProcessSignalsDict(Signals)

    # initiate data manager
    DM = DataManager(sPath, Date=Date, source=source, region=region, Inputs=Inputs, DBs=DBs)
    DM.ProcessFT()

    # get empty index with DTID and FTID to collect the results
    FetchedData = DM.fetch(
        {"FT": {"start": -18, "end": 0, "items": ["DTID"]}, "dt": {"start": -1, "end": 0, "items": []}}
    )
    dt = FetchedData["dt"]
    dt.reset_index(inplace=True)
    dt = dt[["DTID"]].groupby("DTID").last()
    FT = FetchedData["FT"]
    FT.reset_index(inplace=True)
    FT = FT[["DTID", "FTID"]].groupby("DTID").last()
    JoinedSignals = dt.merge(FT, on="DTID", how="left")
    JoinedSignals.reset_index(inplace=True)

    for Class, ClassSignals in SignalsDict.items():
        try:
            Signal = eval(Class + "().CreateSignal(DM, Out=ClassSignals)")
            if Signal.index.names == ["DTID"]:
                JoinedSignals = JoinedSignals.merge(Signal, on="DTID", how="left")
            elif Signal.index.names == ["FTID"]:
                JoinedSignals = JoinedSignals.merge(Signal, on="FTID", how="left")
            else:
                raise Exception(Class + " is not correctly implemented. Expecte DTID or FTID index.")
        except:
            # if error add the signals as nan
            for Signal in ClassSignals:
                JoinedSignals[Signal] = np.nan

    JoinedSignals["date"] = DM.Date
    return JoinedSignals


def CreateSignals(DateSequence, sPath, source="DST", region=None, Signals=None, InMemory=True):
    """
    function that processes the defined signals at each date of DateSequence and returns panel data with them
    inputs:
        DateSequence: list of dates at which to compute the signals
        sPath: location to master folder
        source: either DST (DataStream) or WRDS (CRSP + Compustat)
        region: WRDS: [USA], DST: [North America, Japan, Europe, Asia Pacific]
        Signals: list of signals that should be computed
        InMemmory: will store all the Data in memory and fetch it from there for each time period, needs 64GB RAM
    """

    # now use the auxiliary function CreateSignalsWorker to compute signals at each time index in DateSequence
    if type(DateSequence) == str:
        DateSequence = [DateSequence]

    if type(region) == str:
        region = [region]

    if source == "DST":
        if region == None:
            region = ["North America", "Japan", "Europe", "Asia Pacific"]
            print(
                "Will default to ['North America', 'Japan', 'Europe', 'Asia Pacific'] in region argument for DST."
            )
        elif any([reg not in ["North America", "Japan", "Europe", "Asia Pacific"] for reg in region]):
            raise Exception(
                "Region needs to be either None or a subset of"
                "['North America', 'Japan', 'Europe', 'Asia Pacific']."
            )
    elif source == "WRDS":
        region = [None]
        print("Only USA is available as a region for WRDS and the code will default to it now.")
    else:
        raise Exception("source should be either WRDS or DST.")

    Output = []
    for reg in region:
        # preallocate all the data in memory if needed
        if InMemory == True:
            Inputs = ProcessSignalsDict(Signals)[0]
            for key in Inputs.keys():
                Inputs[key]["start"] = -1800  # shift the beginning by 150 years
            DBs = DataManager(sPath, Date="2050-01-01", source=source, region=reg, Inputs=Inputs)
        else:
            DBs = None

        if reg is not None:
            print("Starting " + reg)

        for TimeIndex in tqdm(DateSequence):
            Output += [
                CreateSignalsWorker(
                    Date=TimeIndex, sPath=sPath, source=source, region=reg, Signals=Signals, DBs=DBs
                )
            ]
    Output = pd.concat(Output)
    return Output


def CreateReturnsWorker(DateStart, DateEnd, DM):
    """
    creates returns at the given T0 date
    """

    # convert to datetime format
    DateStart = datetime.strptime(DateStart, "%Y-%m-%d")
    DateEnd = datetime.strptime(DateEnd, "%Y-%m-%d")
    DateDiff = np.ceil((DateEnd - DateStart).days / 30) + 1

    DM.Date = DateStart
    FetchedData = DM.fetch({"dt": {"start": -1, "end": DateDiff, "items": ["RI"]}})
    dt = FetchedData["dt"]
    # limit to firms with prices in the past 7 days
    dt = dt.loc[dt.index.get_level_values("date") >= DateStart - relativedelta(days=7)].copy()
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


def winsor(y, threshold):
    """
    function that performs winsorization on panda series
    """

    x = y.copy()
    LowerQuant = x.quantile(threshold)
    UpperQuant = x.quantile(1 - threshold)
    x[x < LowerQuant] = LowerQuant
    x[x > UpperQuant] = UpperQuant
    return x


def CreateReturns(DateSequenceStart, DateSequenceEnd, sPath, source="DST", region=None):
    """
    function that processes the future returns at each date of DateSequence and returns panel data with them
    inputs:
        DateSequenceStart: list of dates at which the return starts
        DateSequenceEnd:  list of dates at which the return ends
        sPath: location to master folder
        source: either DST (DataStream) or WRDS (CRSP + Compustat)
        region: WRDS: [USA], DST: [North America, Japan, Europe, Asia Pacific]
    """

    # use the auxiliary function CreateReturnsWorker to compute returns at each time index in DateSequence
    if type(DateSequenceStart) == str:
        DateSequenceStart = [DateSequenceStart]

    if type(DateSequenceEnd) == str:
        DateSequenceEnd = [DateSequenceEnd]

    if len(DateSequenceStart) != len(DateSequenceEnd):
        raise Exception("DateSequenceStart must be of the same length as DateSequenceEnd.")

    if type(region) == str:
        region = [region]

    if source == "DST":
        if region == None:
            region = ["North America", "Japan", "Europe", "Asia Pacific"]
            print(
                "Will default to ['North America', 'Japan', 'Europe', 'Asia Pacific'] in region argument for DST."
            )
        elif any([reg not in ["North America", "Japan", "Europe", "Asia Pacific"] for reg in region]):
            raise Exception(
                "Region needs to be either None or a subset of"
                "['North America', 'Japan', 'Europe', 'Asia Pacific']."
            )
    elif source == "WRDS":
        region = ["USA"]
        print("Only 'USA' is available as a region for WRDS and the code will default to it now.")
    else:
        raise Exception("source should be either WRDS or DST.")

    Output = pd.DataFrame()
    for reg in region:
        # preallocate all the data in memory
        DM = DataManager(
            sPath,
            Date="2050-01-01",
            source=source,
            region=reg,
            Inputs={"dt": {"start": -1800, "end": 0, "items": ["RI"]}},
        )

        if reg is not None:
            print("Starting " + reg)

        print(f"Number of periods to be processed: {len(DateSequenceStart)}")
        for TimeStart, TimeEnd in tqdm(zip(DateSequenceStart, DateSequenceEnd)):
            ret = CreateReturnsWorker(DateStart=TimeStart, DateEnd=TimeEnd, DM=DM)

            # winsorize if the source is DST
            if source == "DST":
                try:
                    if datetime.strptime(TimeStart, "%Y-%m-%d") < datetime.strptime("1990-01-01", "%Y-%m-%d"):
                        ret.r = ret.r.transform(lambda x: winsor(x, 0.001))
                    elif datetime.strptime(TimeStart, "%Y-%m-%d") < datetime.strptime("2000-01-01", "%Y-%m-%d"):
                        ret.r = ret.r.transform(lambda x: winsor(x, 0.0001))
                except:
                    pass

            Output = pd.concat([Output, ret])

    if source == "DST":
        try:
            # set returns larger than 20 to missing
            Output.loc[Output["r"] > 20, "r"] = np.nan
            # discard extreme returns that reverse back the next month
            Output["l.r"] = Output.groupby("DTID").r.shift(1)  # lag
            Output["f.r"] = Output.groupby("DTID").r.shift(-1)  # lead
            Output.loc[
                (((Output["r"] > 3) | (Output["l.r"] > 3)) & ((1 + Output["l.r"]) * (1 + Output["r"]) < 1.5)), "r"
            ] = np.nan
            Output.loc[
                (((Output["r"] > 3) | (Output["f.r"] > 3)) & ((1 + Output["f.r"]) * (1 + Output["r"]) < 1.5)), "r"
            ] = np.nan
            del Output["l.r"], Output["f.r"]
        except:
            print("Filter of returns for wrong values failed.")

    return Output


def FetchCharacteristics(Date, DM, COID):
    """
    creates characteristics to filter the universe at the given T0 date
    """

    # convert to datetime format
    DM.Date = datetime.strptime(Date, "%Y-%m-%d")
    FetchedData = DM.fetch({"dt": {"start": -12, "end": 0, "items": ["PRC", "VOL", "MC", "RI", COID]}})
    dt = FetchedData["dt"]
    dt.dropna(subset=["PRC", "MC"], inplace=True)

    # keep only tickers with observations in the past week
    RecentObs = dt.loc[
        (dt.index.get_level_values("date") > DM.Date - relativedelta(days=7))
    ].index.get_level_values("DTID")
    dt = dt.loc[dt.index.get_level_values("DTID").isin(RecentObs)].copy()

    # get market cap and price
    MC = dt.groupby("DTID").MC.last()
    PRC = dt.groupby("DTID").PRC.last()
    RI = dt.groupby("DTID").RI.last()
    COIDval = dt.groupby("DTID")[COID].last()

    # get average daily market trading volume over the past year
    dt["VOL"] = dt["VOL"] * dt["PRC"]
    EnoughObs = dt["VOL"].groupby("DTID").count() >= 50
    dt = dt.loc[dt.index.get_level_values("DTID").isin(EnoughObs[EnoughObs == True].index)].copy()
    vol = dt["VOL"].groupby("DTID").mean() / 1000000

    result = pd.DataFrame({"MC": MC, "VOL": vol, "PRC": PRC, "RI": RI, "COID": COIDval})
    result["date"] = DM.Date
    result.index.rename("DTID", inplace=True)
    result.reset_index(inplace=True)
    return result


def Filter(Universe, MClim=0.05, VOLlim=0.05, PRClim=1, RIlim=0.0003):
    """
    filters the stock universe of stocks according to our setting
    """

    Universe.dropna(subset=["PRC", "MC"], inplace=True)
    for col in ["PRC", "RI", "MC", "VOL"]:
        Universe[col] = Universe[col].fillna(0)

    # drop stocks with small capitalization defined as fraction of total
    Universe.sort_values("MC", inplace=True)
    Universe["MCsort"] = Universe.MC.cumsum() / Universe.MC.sum()

    # drop stocks with small traded volume defined as fraction of total
    Universe.sort_values("VOL", inplace=True)
    Universe["VOLsort"] = Universe.VOL.cumsum() / Universe.VOL.sum()

    # always require that the return index is at least 0.0003 so that rounding does not affect the results
    Universe = Universe.loc[Universe.RI >= RIlim].copy()
    del Universe["RI"]

    # apply the universe filters on MC, trading volume, and PRC
    Universe["Eligible"] = True
    Universe.loc[Universe["MCsort"] <= MClim, "Eligible"] = False
    Universe.loc[
        ((Universe["VOLsort"] <= VOLlim) & ((Universe["VOLsort"] != 0) | (Universe["MCsort"] <= MClim * 2))),
        "Eligible",
    ] = False
    Universe.loc[Universe["PRC"] <= PRClim, "Eligible"] = False
    del Universe["VOLsort"], Universe["MCsort"]
    return Universe


def UniverseFilter(
    DateSequence,
    sPath,
    source="DST",
    region=None,
    MClim=0.05,
    VOLlim=0.05,
    RIlim=0.0003,
    PRCfilter={"USA": 1, "North America": 1, "Japan": 1, "Europe": 1, "Asia Pacific": 0.1},
):
    """
    function that processes volume, market cap, and price filter for universe of stocks
    inputs:
        DateSequence: list of dates at which to compute the filters
        sPath: location to master folder
        source: either DST (DataStream) or WRDS (CRSP + Compustat)
        region: WRDS: [USA], DST: [North America, Japan, Europe, Asia Pacific]
    """

    if type(DateSequence) == str:
        DateSequence = [DateSequence]

    if type(region) == str:
        region = [region]

    if source == "DST":
        COID = "FTID"  # Worldscope identifier in DST
        if region == None:
            region = ["North America", "Japan", "Europe", "Asia Pacific"]
            print(
                "Will default to ['North America', 'Japan', 'Europe', 'Asia Pacific'] in region argument for DST."
            )
        elif any([reg not in ["North America", "Japan", "Europe", "Asia Pacific"] for reg in region]):
            raise Exception(
                "Region needs to be either None or a subset of"
                "['North America', 'Japan', 'Europe', 'Asia Pacific']."
            )
    elif source == "WRDS":
        COID = "PERMCO"  # use PERMCO in CRSP
        region = ["USA"]
        print("Only 'USA' is available as a region for WRDS and the code will default to it now.")
        RIlim = 0.0  # do not discard based on RI as it is not subject to rounding for CRSP
    else:
        raise Exception("source should be either WRDS or DST.")

    Output = pd.DataFrame()
    for reg in region:
        if reg is not None:
            print("Starting " + reg)

        # preallocate all the data in memory
        DM = DataManager(
            sPath,
            Date="2050-01-01",
            source=source,
            region=reg,
            Inputs={"dt": {"start": -1800, "end": 0, "items": ["PRC", "VOL", "MC", "RI", COID]}},
        )

        for TimeIndex in tqdm(DateSequence):
            Universe = FetchCharacteristics(Date=TimeIndex, DM=DM, COID=COID)
            Universe = Filter(Universe, MClim=MClim, VOLlim=VOLlim, PRClim=PRCfilter[reg], RIlim=RIlim)
            Universe["region"] = reg
            Output = pd.concat([Output, Universe])

    return Output


def MLdata(Ret, Signals, Universe, source="WRDS", normalize="Quantile", uviverse_filter=True, exc_normalize=[]):
    """
    this function connects returns, signals, and universe together and formats everything for ML application
    """

    if uviverse_filter:
        Universe = Universe.loc[Universe["Eligible"]].copy()

    data = pd.merge(Universe[["date", "DTID", "region", "MC", "COID"]], Ret, on=["date", "DTID"], how="left")
    data = pd.merge(data, Signals, on=["date", "DTID"], how="left")
    del Universe, Ret, Signals

    # keep only one observation per firm in each time period
    # rank the firm securities with fundamental data linking as 0 and those without as 1
    data["ranking"] = data.FTID.isnull() * 1
    data.sort_values("ranking", inplace=True)
    # drop duplicated firm-date observations
    data = data.loc[~data.duplicated(subset=["COID", "date"], keep="first") | data.COID.isnull()].copy()
    del data["COID"], data["FTID"], data["ranking"]

    # fill in the missing returns with zero
    data["r"] = data["r"].fillna(0)

    # normalize all the signals to their cross-sectional ranks
    signals = [item for item in data.columns if item not in ["DTID", "date", "r", "region", "MC"] + exc_normalize]
    for signal in signals:
        if normalize == "Quantile":
            data[signal] = data.groupby(["date", "region"])[signal].rank(pct=True)
            data[signal] = data[signal].fillna(0.5)
        elif normalize == "Standardize":
            data[signal] = (
                data[signal] - data.groupby(["date", "region"])[signal].transform("mean")
            ) / data.groupby(["date", "region"])[signal].transform("std")
            data[signal] = data[signal].fillna(0)
        elif normalize == "Standardize_winsor":
            Q_cut = 0.025
            Q = data.groupby(["date", "region"])[signal].transform("quantile", 1 - Q_cut)
            data.loc[data[signal] > Q, signal] = Q
            Q = data.groupby(["date", "region"])[signal].transform("quantile", Q_cut)
            data.loc[data[signal] < Q, signal] = Q
            data[signal] = (
                data[signal] - data.groupby(["date", "region"])[signal].transform("mean")
            ) / data.groupby(["date", "region"])[signal].transform("std")
            data[signal] = data[signal].fillna(0)
    return data


def MktMeanRet(sPath, regions=["USA", "Europe", "Japan", "Asia Pacific"]):
    result = []
    for region in regions:
        if region == "USA":
            source = "WRDS"
            start_date = "1969-12-01"
        else:
            source = "DST"
            start_date = "1989-12-01"

        DM = DataManager(
            sPath,
            Date="2050-01-01",
            source=source,
            Inputs={"dt": {"start": -1800, "end": 0, "items": ["r", "RI"]}},
            region=region,
        )

        FetchedData = DM.fetch({"dt": {"start": -1600, "end": 0, "items": ["r", "RI"]}})
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
        res = res.loc[res["date"] >= start_date]
        res["region"] = region
        result += [res]
        del dt, DM, FetchedData

    result = pd.concat(result)

    return result


def GetDailyRetMC(sPath, start_date, end_date, regions=["USA", "Europe", "Japan", "Asia Pacific"]):
    ret = []
    for region in regions:
        if region == "USA":
            source = "WRDS"
        else:
            source = "DST"

        DM = DataManager(
            sPath,
            Date="2050-01-01",
            source=source,
            Inputs={"dt": {"start": -1600, "end": 0, "items": ["r", "MC"]}},
            region=region,
        )

        FetchedData = DM.fetch({"dt": {"start": -1600, "end": 0, "items": ["r", "MC"]}})
        dt = FetchedData["dt"].copy()
        dt["MC"] = dt.groupby("DTID")["MC"].shift(1)  # use MC from the previous day
        dt.reset_index(inplace=True)
        dt.dropna(subset="r", inplace=True)
        dt = dt.loc[(dt["date"] >= start_date) & (dt["date"] <= end_date)].copy()
        dt["region"] = region
        ret += [dt]
        del dt, DM, FetchedData
    ret = pd.concat(ret)
    return ret


def GetFutureVola1M(sPath, start_date, end_date):
    result = []
    for DataSource in ["WRDS", "DST"]:
        if DataSource == "WRDS":
            DM = DataManager(
                sPath,
                Date="2050-01-01",
                source="WRDS",
                Inputs={"dt": {"start": -1600, "end": 0, "items": ["r"]}},
            )
        elif DataSource == "DST":
            DM = DataManager(
                sPath,
                Date="2050-01-01",
                source="DST",
                Inputs={"dt": {"start": -1600, "end": 0, "items": ["r"]}},
            )
        FetchedData = DM.fetch({"dt": {"start": -1600, "end": 0, "items": ["r"]}})
        dt = FetchedData["dt"].copy()
        dt = dt.dropna(subset="r")
        dt["r2"] = dt["r"] ** 2
        dt.reset_index(inplace=True)
        for date in tqdm(TimeSequence(start=start_date, end=end_date)):
            dt_ = dt.loc[
                (dt["date"] >= date) & (dt["date"] < pd.to_datetime(date) + relativedelta(months=1))
            ].copy()
            dt_ = dt_.loc[dt_.groupby("DTID")["r2"].transform("count") >= 15].copy()
            res = dt_.groupby("DTID")["r2"].sum().reset_index()
            res["date"] = date
            res.rename({"r2": "var"}, axis=1, inplace=True)
            res["vol"] = np.sqrt(res["var"])
            result += [res]
    result = pd.concat(result)
    return result
