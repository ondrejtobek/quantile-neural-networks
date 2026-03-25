import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from Signals.Fundamental import Accruals
from Signals.SignalClass import SignalClass


class IBESForecastChange(SignalClass):
    """
    Barber et al. (2001) JF
    Down Forecast: DF
    Up Forecast: UF
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -2, "end": 0, "items": ["MEANEST", "FPI", "TICKER"]}}
        self.Output = ["DF", "UF"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes["MEANEST.ch"] = ibes["MEANEST"] - ibes["MEANEST"].groupby("DTID").shift(1)
        # Down Forecast
        down = (ibes["MEANEST.ch"] < 0) * 1
        # Up Forecast
        up = (ibes["MEANEST.ch"] > 0) * 1
        result = pd.DataFrame({"DF": down, "UF": up})
        result = result.groupby("DTID").last()
        return result[Out]


class IBESForecastDispersion(SignalClass):
    """
    dispersion in analyst forecasts: Diether, Malloy, and Scherbina (2002)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -1, "end": 0, "items": ["STDEV", "MEANEST", "NUMEST", "FPI"]}}
        self.Output = ["FD"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        fd = ibes["STDEV"] / ibes["MEANEST"]
        fd[ibes["MEANEST"] == 0] = np.inf
        fd[ibes["NUMEST"] < 3] = np.nan
        result = pd.DataFrame({"FD": fd})
        result = result.groupby("DTID").last()
        return result[Out]


class IBESAnalystsCoverage(SignalClass):
    """
    Analysts Coverage: Elgers, Lo, and Pfeiffer (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -1, "end": 0, "items": ["NUMEST", "FPI"]}}
        self.Output = ["AC"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ac = ibes["NUMEST"]
        result = pd.DataFrame({"AC": ac})
        result = result.groupby("DTID").last()
        return result[Out]


class IBESDispLT(SignalClass):
    """
    Dispersion in Analyst Long-term Growth Forecasts: Anderson, Ghysels, and Juergens (2005)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -1, "end": 0, "items": ["STDEV", "NUMEST", "FPI"]}}
        self.Output = ["DispLT"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 0].copy()
        ibes.loc[ibes["NUMEST"] < 3] = np.nan
        disp = ibes["STDEV"]
        result = pd.DataFrame({"DispLT": disp})
        result = result.groupby("DTID").last()
        return result[Out]


class IBESDispLTST(SignalClass):
    """
    Disparity between Long- and Short-term Earnings Growth Forecasts: Da and Warachka (2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {"start": -18, "end": 0, "items": ["epspi", "DTID"]},
            "IBESsum": {"start": -1, "end": 0, "items": ["MEDEST", "FPI"]},
        }
        self.Output = ["DispLTST"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        FT = FetchedData["FT"]
        ibes_red = ibes.loc[ibes["FPI"] == 0].copy()
        ibes_red.rename(columns={"MEDEST": "MEDEST_LT"}, inplace=True)
        ibes_red = ibes_red.groupby("DTID").last()
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes = ibes.groupby("DTID").last()
        ibes = pd.merge(ibes[["MEDEST"]], ibes_red[["MEDEST_LT"]], on="DTID")
        FT = FT.groupby("DTID").last()
        ibes = pd.merge(ibes, FT, on="DTID")
        disp = ibes["MEDEST_LT"] - 100 * (ibes["MEDEST"] - ibes["epspi"]) / abs(ibes["epspi"])
        result = pd.DataFrame({"DispLTST": disp})
        return result[Out]


class ChangesInAnalystFC(SignalClass):
    """
    hanges in Analyst Earnings Forecasts: Hawkins, Chamberlin, and Daniel (1984)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -2, "end": 0, "items": ["MEANEST", "MEDEST", "FPI", "TICKER"]}}
        self.Output = ["dAEF"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes["MEANEST.ch"] = ibes["MEANEST"] - ibes["MEANEST"].groupby("DTID").shift(1)
        chaf = ibes["MEANEST.ch"] / (abs(ibes["MEDEST"]) / 2 + abs(ibes.groupby("TICKER")["MEDEST"].shift(1)) / 2)
        result = pd.DataFrame({"dAEF": chaf})
        result = result.groupby("DTID").last()
        return result[Out]


class LongtermGrowthForecasts(SignalClass):
    """
    Long-term Growth Forecasts: La Porta (1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESsum": {"start": -1, "end": 0, "items": ["MEDEST", "FPI"]}}
        self.Output = ["LTGrF"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        ibes = ibes.loc[ibes["FPI"] == 0].copy()
        ltgf = ibes["MEDEST"]
        result = pd.DataFrame({"LTGrF": ltgf})
        result = result.groupby("DTID").last()
        return result[Out]


class IBESChForecastAccrual(SignalClass):
    """
    Change in Forecast + Accrual
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {"start": -18, "end": 0, "items": ["at", "sale", "DTID"]},
            "IBESsum": {"start": -2, "end": 0, "items": ["MEANEST", "FPI", "TICKER"]},
        }
        self.Output = ["ChiFA"]
        self.DependsOn = ["Accruals"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        FT = FetchedData["FT"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes["MEANEST.ch"] = ibes["MEANEST"] - ibes["MEANEST"].groupby("DTID").shift(1)
        ibes = ibes.groupby("DTID").last()

        # accruals measure
        accr = Accruals().CreateSignal(DM)["Accr"]
        FT = FT.groupby("FTID").last()
        FT = FT.merge(accr, on="FTID")
        FT["Accr"] = FT["Accr"] - FT["Accr"].median()  # subtract median
        FT.loc[(FT["at"] < 50) | (FT["sale"] < 25), "Accr"] = np.nan  # at_50M, sales_25M filters
        FT = FT[["Accr", "DTID"]].groupby("DTID").last()

        ibes = ibes.merge(FT, on="DTID")
        chfa = ((ibes["Accr"] < 0) & (ibes["MEANEST.ch"] > 0)) * 1 - (
            (ibes["Accr"] > 0) & (ibes["MEANEST.ch"] < 0)
        ) * 1
        result = pd.DataFrame({"ChiFA": chfa})
        return result[Out]


class IBESChAnalystValue(SignalClass):
    """
    Analyst Value: Frankel and Lee (1998)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {"start": -30, "end": 0, "items": ["bkvlps", "DTID"]},
            "dt": {"start": -1, "end": 0, "items": ["PRC"]},
            "IBESsum": {"start": -1, "end": 0, "items": ["MEDEST", "FPI"]},
        }
        self.Output = ["AV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        FT = FetchedData["FT"]
        dt = FetchedData["dt"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes = ibes.groupby("DTID").last()

        # average of current book value per share a and its value one year ago
        FT["be"] = (FT["bkvlps"] + FT["bkvlps"].groupby("FTID").shift(1)) / 2
        FT.loc[FT["be"].isnull(), "be"] = FT["bkvlps"]
        FT = FT[["be", "DTID"]].groupby("DTID").last()
        ibes = ibes.merge(FT, on="DTID")

        dt = dt.groupby("DTID").last()
        ibes = ibes.merge(dt, on="DTID")

        ibes["froe"] = ibes["MEDEST"] / ibes["be"]
        # remove negative book value stocks, stocks with ROE over 100%
        ibes.loc[ibes["froe"] > 1, "froe"] = np.nan
        ibes.loc[ibes["be"] < 0, "froe"] = np.nan
        av = (1 + (ibes["froe"] - 0.1) / 1.1 + (ibes["froe"] - 0.1) / (1.1 * 0.1)) * (ibes["be"] / ibes["PRC"])
        result = pd.DataFrame({"AV": av})
        return result[Out]


class ChangeInRecommendation(SignalClass):
    """
    Change in Recommendation: Jegadeesh et al. (2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"IBESdet": {"start": -60, "end": 0, "items": ["TICKER", "ESTIMID", "ITEXT"]}}
        self.Output = ["ChR"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESdet"]
        ibes.reset_index(inplace=True)
        ibes.drop_duplicates(subset=["date", "TICKER", "ESTIMID"], keep="last", inplace=True)
        ibes.set_index(["TICKER", "ESTIMID", "date"], inplace=True)
        ibes.sort_index(inplace=True)
        ibes["l.ITEXT"] = ibes["ITEXT"].groupby(["TICKER", "ESTIMID"]).shift(1)
        ibes.reset_index(inplace=True)
        ibes["S"] = ibes["ITEXT"] == "STRONG BUY"
        ibes["l.S"] = ibes["l.ITEXT"] == "STRONG BUY"
        # fill all the changed recommendations as -1 and 0 otherwise
        ibes["dREC"] = -1 * (ibes["ITEXT"] != ibes["l.ITEXT"])
        # fill missing values as 0
        ibes.loc[ibes["l.ITEXT"].isnull(), "dREC"] = 0
        # fill all new strong buys as 1
        ibes.loc[(ibes["S"] == True) & (ibes["l.S"] == False), "dREC"] = 1
        # fill all recommendations older than 1 month as 0
        ibes.loc[ibes.date < DM.Date - relativedelta(months=1), "dREC"] = 0
        # take the last observation per analyst and stock and compute average change
        ibes = ibes.groupby(["TICKER", "ESTIMID"]).last()
        chrec = ibes.groupby("DTID")["dREC"].mean()
        result = pd.DataFrame({"ChR": chrec})
        return result[Out]


class EarningsForecastToPrice(SignalClass):
    """
    Earnings Forecast-to-price: Elgers, Lo, and Pfeiffer (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "dt": {"start": -1, "end": 0, "items": ["PRC"]},
            "IBESsum": {"start": -1, "end": 0, "items": ["MEDEST", "FPI"]},
        }
        self.Output = ["EFoP"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        ibes = FetchedData["IBESsum"]
        dt = FetchedData["dt"]
        ibes = ibes.loc[ibes["FPI"] == 1].copy()
        ibes = ibes.groupby("DTID").last()

        dt.dropna(inplace=True)
        dt = dt.groupby("DTID").last()
        ibes = ibes.merge(dt, on="DTID")

        fep = ibes["MEDEST"] / ibes["PRC"]
        result = pd.DataFrame({"EFoP": fep})
        return result[Out]
