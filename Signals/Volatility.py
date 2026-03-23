import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from Signals.SignalClass import SignalClass


class EWMAVolatility(SignalClass):
    """
    EWMA filters for volatility
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -30, "end": 0, "items": ["r"]}}
        self.Output = [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt["r2"] = dt["r"] ** 2
        dt.loc[dt["r2"] > 0.09, "r2"] = 0.09
        for decay in [20, 10, 6, 4, 2, 1]:
            x = dt.groupby("DTID")["r2"].ewm(alpha=decay / 100, adjust=False).mean()
            x = x.reset_index(drop=True, level=1)
            dt[f"EWMAVol{decay}"] = np.sqrt(x)
        result = dt.groupby("DTID")[self.Output].last()
        return result[Out]


class EWMARange(SignalClass):
    """
    EWMA filters for volatility using high low
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -30, "end": 0, "items": ["r", "H", "L"]}}
        self.Output = [f"EWMARange{i}" for i in [20, 10, 6, 4, 2, 1]]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt["Range"] = 0.361 * ((dt["H"] - dt["L"]) / (dt["H"] + dt["L"]) * 2) ** 2
        dt.loc[dt["Range"] == 0, "Range"] = dt["r"] ** 2
        dt.loc[dt["Range"] > 0.09, "Range"] = 0.09
        for decay in [20, 10, 6, 4, 2, 1]:
            x = dt.groupby("DTID")["Range"].ewm(alpha=decay / 100, adjust=False).mean()
            x = x.reset_index(drop=True, level=1)
            dt[f"EWMARange{decay}"] = np.sqrt(x)
        result = dt.groupby("DTID")[self.Output].last()
        return result[Out]


class EWMAVolatilityD(SignalClass):
    """
    EWMA filters for volatility
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -30, "end": 0, "items": ["r"]}}
        self.Output = [f"EWMAVolD{i}" for i in [20, 10, 6]]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt.loc[dt["r"] > 0, "r"] = 0
        dt["r2"] = dt["r"] ** 2
        for decay in [20, 10, 6]:
            x = dt.groupby("DTID")["r2"].ewm(alpha=decay / 100, adjust=False).mean()
            x = x.reset_index(drop=True, level=1)
            dt[f"EWMAVolD{decay}"] = np.sqrt(x)
        result = dt.groupby("DTID")[self.Output].last()
        return result[Out]


class Volatility3M(SignalClass):
    """
    Total Volatility over the past 3 months
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -3, "end": 0, "items": ["r"]}}
        self.Output = ["TV3M"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        vol = dt["r"].groupby("DTID").std()
        result = pd.DataFrame({"TV3M": vol})
        return result[Out]


class Volatility6M(SignalClass):
    """
    Total Volatility over the past 6 months
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -6, "end": 0, "items": ["r"]}}
        self.Output = ["TV6M"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        vol = dt["r"].groupby("DTID").std()
        result = pd.DataFrame({"TV6M": vol})
        return result[Out]


class Volatility12M(SignalClass):
    """
    Total Volatility over the past 12 months
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -12, "end": 0, "items": ["r"]}}
        self.Output = ["TV12M"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        vol = dt["r"].groupby("DTID").std()
        result = pd.DataFrame({"TV12M": vol})
        return result[Out]
