import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from Signals.SignalClass import SignalClass


class ShareIssuance5Y(SignalClass):
    """
    Share Issuance (5-Year): Daniel and Titman (JF 2006)  = composite equity issuance
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -61, "end": 0, "items": ["MC", "RI"]}}
        self.Output = ["CEI5Y"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["MC"] = dt["MC"].replace({0: np.nan})
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=60)].copy()
        dt2.rename(columns={"RI": "RI_lag", "MC": "MC_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        cei = np.log(dt["MC"] / dt["MC_lag"]) - np.log(dt["RI"] / dt["RI_lag"])
        result = pd.DataFrame({"CEI5Y": cei})
        return result[Out]


class ShareIssuance1Y(SignalClass):
    """
    Share Issuance (1-Year): Pontiff and Woodgate (JF 2008) ! net stock issues
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -13, "end": 0, "items": ["SHROUT", "AF"]}}
        self.Output = ["SI1Y"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["RealShares"] = dt["SHROUT"] / dt["AF"]
        dt = dt[["RealShares"]]
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=12)].copy()
        dt2.rename(columns={"RealShares": "RealShares_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        nsi = np.log(dt["RealShares"] / dt["RealShares_lag"])
        result = pd.DataFrame({"SI1Y": nsi})
        return result[Out]


class CashFlowVariance(SignalClass):
    """
    Share Issuance (1-Year): Pontiff and Woodgate (JF 2008) ! net stock issues
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {"start": -78, "end": 0, "items": ["ib", "dp", "DTID", "FinYearEnd"]},
            "dt": {"start": -61, "end": 0, "items": ["MC"]},
        }
        self.Output = ["CFV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        FT = FetchedData["FT"]

        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(0, 60)]
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
        FT.reset_index(inplace=True)
        # use date that corresponds to FT accounting year end
        FT["date"] = FT["FinYearEnd"]
        FT.set_index("date", inplace=True)
        FT.sort_index(inplace=True)
        DT = pd.merge_asof(DT, FT[["ib", "dp", "DTID"]], on="date", by="DTID")

        def ComputeVar(g):
            x = g["cfm"]
            if len(x) < 36:
                return np.nan
            else:
                return np.std(x)

        DT["cfm"] = (DT["ib"] + DT["dp"]) / DT["MC"]
        DT.dropna(subset=["cfm"], inplace=True)
        cfm = DT[["cfm", "DTID"]].groupby("DTID").apply(ComputeVar)
        result = pd.DataFrame({"CFV": cfm})
        return result[Out]


class BidAskSpread(SignalClass):
    """
    Bid-Ask Spread: Amihud and Mendelson (1986)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["r", "VOL", "PRC"]}}
        self.Output = ["BidAsk"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["VOL"].groupby("DTID").transform("count") >= 10].copy()
        liq = 8 * (dt["r"].groupby("DTID").std() ** (2 / 3)) / (dt["VOL"].groupby("DTID").sum() ** (1 / 3))
        liq.replace(np.inf, 0.05, inplace=True)
        result = pd.DataFrame({"BidAsk": liq})
        return result[Out]


class Amihud(SignalClass):
    """
    Bid-Ask Spread: Amihud and Mendelson (1986)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -12, "end": 0, "items": ["r", "VOL", "PRC"]}}
        self.Output = ["Amihud"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt["amih"] = abs(dt["r"]) / dt["VOL"]
        dt.loc[dt["VOL"] == 0, "amih"] = np.nan
        dt.dropna(inplace=True)
        dt = dt.loc[dt["amih"].groupby("DTID").transform("count") >= 50].copy()
        liq = dt["amih"].groupby("DTID").mean()
        liq.replace(np.inf, np.nan, inplace=True)
        result = pd.DataFrame({"Amihud": liq})
        return result[Out]


class LiquidityShocks(SignalClass):
    """
    Liquidity Shocks: Bali, Peng, Shen, and Tang (2013)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -13, "end": 0, "items": ["r", "VOL", "PRC"]}}
        self.Output = ["LiqShck"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt["amih"] = abs(dt["r"]) / dt["VOL"]
        dt.loc[dt["VOL"] == 0, "amih"] = np.nan
        dt = dt[["amih"]].copy()
        dt.dropna(inplace=True)
        dt["month"] = -np.ceil((DM.Date - dt.index.get_level_values("date")).days / 30)
        dt.reset_index(inplace=True)
        EnoughObs = dt.groupby(["DTID", "month"])["amih"].count() >= 10
        EnoughObs = EnoughObs.loc[EnoughObs].copy()
        dt = pd.merge(dt, pd.DataFrame({"EnoughObs": EnoughObs}), on=["DTID", "month"], how="inner")
        liq = dt.groupby(["DTID", "month"])["amih"].mean()
        liq.replace(np.inf, 1, inplace=True)
        liqshck = (
            liq[liq.index.get_level_values("month") == -1].groupby("DTID").mean()
            - liq[liq.index.get_level_values("month") != -1].groupby("DTID").mean()
        )
        result = pd.DataFrame({"LiqShck": liqshck})
        return result[Out]


class Size(SignalClass):
    """
    Size: Banz (1981)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["MC"]}}
        self.Output = ["Size"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt = dt.groupby("DTID").last()
        size = dt["MC"]
        result = pd.DataFrame({"Size": size})
        return result[Out]


class Volatility(SignalClass):
    """
    Total Volatility: Ang, Hodrick, Xing, and Zhang (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["r"]}}
        self.Output = ["TV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["r"].groupby("DTID").transform("count") >= 15].copy()
        vol = dt["r"].groupby("DTID").std()
        result = pd.DataFrame({"TV": vol})
        return result[Out]


class ShareTurnover(SignalClass):
    """
    Share Turnover: Chordia, Subranhmanyam, and Anshuman (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -3, "end": 0, "items": ["VOL", "SHROUT"]}}
        self.Output = ["ST"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["VOL"].groupby("DTID").transform("count") >= 40].copy()
        st = dt["VOL"].groupby("DTID").sum() / dt["SHROUT"].groupby("DTID").last() / 1000000
        result = pd.DataFrame({"ST": st})
        return result[Out]


class ShareTurnoverVar(SignalClass):
    """
    Coefficient of Variation of Share Turnover: Chordia, Subranhmanyam, and Anshuman (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -6, "end": 0, "items": ["VOL", "SHROUT"]}}
        self.Output = ["CVoST"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["turnover"] = dt["VOL"] / dt["SHROUT"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["turnover"].groupby("DTID").transform("count") >= 50].copy()
        st = dt["turnover"].groupby("DTID").std() / dt["turnover"].groupby("DTID").mean()
        result = pd.DataFrame({"CVoST": st})
        return result[Out]


class VolumeToMarketCap(SignalClass):
    """
    Volume / Market Value of Equity: Haugen and Baker (1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -12, "end": 0, "items": ["VOL", "PRC", "MC"]}}
        self.Output = ["VolMV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["VOL"].groupby("DTID").transform("count") >= 100].copy()
        volmc = dt["VOL"].groupby("DTID").sum() / dt["MC"].groupby("DTID").last() / 1000000
        result = pd.DataFrame({"VolMV": volmc})
        return result[Out]


class VolumeVariance(SignalClass):
    """
    Volume Variance: Chordia, Subranhmanyam, and Anshuman (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -36, "end": 0, "items": ["VOL", "PRC"]}}
        self.Output = ["VarVol"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["VOL"].groupby("DTID").transform("count") >= 250].copy()
        vol = dt["VOL"].groupby("DTID").std()
        result = pd.DataFrame({"VarVol": vol})
        return result[Out]


class MaxReturn(SignalClass):
    """
    Max Return: Bali, Cakici, and Whitelaw (2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["r"]}}
        self.Output = ["Max"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt = dt.loc[dt["r"].groupby("DTID").transform("count") >= 15].copy()
        Max = dt["r"].groupby("DTID").max()
        result = pd.DataFrame({"Max": Max})
        return result[Out]


class Price(SignalClass):
    """
    Price: Blume and Husic (1973)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["PRC"]}}
        self.Output = ["PRC"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt = dt.groupby("DTID").last()
        result = dt[["PRC"]]
        return result[Out]


class Age(SignalClass):
    """
    Firm Age: Barry and Brown (1984)
    limit at 20 years
    Firm Age-Momentum: Zhang (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "dt": {"start": -240, "end": 0, "items": []},
            "FT": {"start": -240, "end": 0, "items": ["DTID"]},
        }
        self.Output = ["Age", "MomAge"]
        self.DependsOn = ["Momentum"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        FT = FetchedData["FT"]
        dt.reset_index(inplace=True)
        FT.reset_index(inplace=True)
        dt = dt.groupby("DTID").first()
        FT = FT.groupby("DTID").first()
        FT.rename(columns={"date": "dateFT"}, inplace=True)
        dt = dt.merge(FT, on="DTID", how="left")
        dt.loc[dt.dateFT < dt.date, "date"] = dt.dateFT
        dt["Age"] = DM.Date - dt["date"]
        dt["Age"] = dt["Age"].dt.days / 365

        # momentum-age
        mom = Momentum().CreateSignal(DM)["Mom"]
        age = dt["Age"]
        momage = pd.merge(mom, age, on="DTID")
        momage = momage.loc[momage.Age >= 12].copy()
        momage = momage.loc[momage.Age < momage.Age.quantile(0.2)].copy()
        result = pd.merge(pd.DataFrame({"MomAge": momage["Mom"]}), dt["Age"], how="outer", on="DTID")
        return result[Out]


class YearHigh(SignalClass):
    """
    52-Week High: George and Hwang (2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -12, "end": 0, "items": ["RI"]}}
        self.Output = ["52WH"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        yh = dt["RI"].groupby("DTID").last() / dt["RI"].groupby("DTID").max()
        result = pd.DataFrame({"52WH": yh})
        return result[Out]


class ShortTermReversal(SignalClass):
    """
    Short-Term Reversal: Jegadeesh (1990)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -2, "end": 0, "items": ["RI"]}}
        self.Output = ["STR"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=1)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        dt["STR"] = dt["RI"] / dt["RI_lag"] - 1
        result = dt[["STR"]]
        return result[Out]


class Momentum(SignalClass):
    """
    Momentum: Jegadeesh and Titman (1993)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -8, "end": -1, "items": ["RI"]}}
        self.Output = ["Mom"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=1)].copy()
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=7)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        dt["Mom"] = dt["RI"] / dt["RI_lag"] - 1
        result = dt[["Mom"]]
        return result[Out]


class LaggedMomentum(SignalClass):
    """
    Lagged Momentum: Novy-Marx (2012)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -13, "end": -6, "items": ["RI"]}}
        self.Output = ["MomLag"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=6)].copy()
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=12)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        dt["MomLag"] = dt["RI"] / dt["RI_lag"] - 1
        result = dt[["MomLag"]]
        return result[Out]


class LongTermReversal(SignalClass):
    """
    Long-Term Reversal: Debondt and Thaler (1985)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -61, "end": -12, "items": ["RI"]}}
        self.Output = ["LTR"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=12)].copy()
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=60)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        dt["LTR"] = dt["RI"] / dt["RI_lag"] - 1
        result = dt[["LTR"]]
        return result[Out]


class MomentumReversal(SignalClass):
    """
    Momentum-Reversal: Jegadeesh and Titman (1993)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -19, "end": -12, "items": ["RI"]}}
        self.Output = ["MomRev"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=12)].copy()
        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=18)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2.groupby("DTID").last()
        dt = dt.join(dt2)
        dt["MomRev"] = dt["RI"] / dt["RI_lag"] - 1
        result = dt[["MomRev"]]
        return result[Out]


class MomentumLTReversal(SignalClass):
    """
    Momentum and LT Reversal: Chan and Kot (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Output = ["MomLTRev"]
        self.DependsOn = ["Momentum", "LongTermReversal"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        mom = Momentum().CreateSignal(DM)["Mom"]
        ltrev = LongTermReversal().CreateSignal(DM)["LTR"]
        top = ((ltrev < ltrev.quantile(0.2)) & (mom > mom.quantile(0.8))) * 1
        bottom = ((ltrev > ltrev.quantile(0.8)) & (mom < mom.quantile(0.2))) * 1
        mom_ltrev = top - bottom
        result = pd.DataFrame({"MomLTRev": mom_ltrev})
        return result[Out]


class MomentumVolume(SignalClass):
    """
    Momentum-Volume: Lee and Swaminathan (2000)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -6, "end": 0, "items": ["VOL", "PRC"]}}
        self.Output = ["MomVol"]
        self.DependsOn = ["Momentum"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        vol = dt.groupby("DTID")["VOL"].sum()
        mom = Momentum().CreateSignal(DM)["Mom"]
        momvol = pd.merge(mom, vol, on="DTID")
        momvol.loc[momvol.VOL < momvol.VOL.quantile(0.8), "Mom"] = np.nan
        result = pd.DataFrame({"MomVol": momvol["Mom"]})
        return result[Out]


class IndustryMomentum(SignalClass):
    """
    Industry Momentum: Grinblatt and Moskwotiz (1999)
    note that the industry information is fetched from fundamental data and this results in
    many missing observations in the earlier time period where fundamental data was not fully
    covered in DST, should change it to static data for DST later
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "dt": {"start": -7, "end": 0, "items": ["RI", "MC"]},
            "FT": {"start": -18, "end": 0, "items": ["DTID", "INDM3"]},
        }
        self.Output = ["IndMom"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dt = FetchedData["dt"]

        dt2 = dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=6)].copy()
        dt2.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt = dt.groupby("DTID").last()
        dt2 = dt2[["RI_lag"]].groupby("DTID").last()
        dt = dt.join(dt2)
        dt["r"] = dt["RI"] / dt["RI_lag"] - 1
        FT = FT.groupby("DTID").last()
        dt = dt.merge(FT, on="DTID")
        dt.dropna(inplace=True)
        indmom = dt.groupby("INDM3").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())

        dt.reset_index(inplace=True)
        dt = dt.merge(pd.DataFrame({"IndMom": indmom}), on="INDM3")
        dt.set_index("DTID", inplace=True)
        result = dt[["IndMom"]]
        return result[Out]


class Seasonality(SignalClass):
    """
    Seasonality: Heston and Sadka (2008)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -241, "end": 0, "items": ["RI"]}}
        self.Output = [
            "Seas",
            "Seas1A",
            "Seas1N",
            "Seas2t5A",
            "Seas2t5N",
            "Seas6t10A",
            "Seas6t10N",
            "Seas11t15A",
            "Seas11t15N",
            "Seas16t20A",
            "Seas16t20N",
        ]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]

        # subset to firms that are available in the last month
        RecentObs = dt.loc[
            dt.index.get_level_values("date") > DM.Date - relativedelta(months=1)
        ].index.get_level_values("DTID")
        dt = dt.loc[dt.index.get_level_values("DTID").isin(RecentObs)].copy()

        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(241)]
        index = pd.MultiIndex.from_product(
            [dt.index.get_level_values("DTID").unique(), new_index], names=["DTID", "date"]
        )
        DT = pd.DataFrame(index=index).reset_index()
        DT = pd.merge(DT, pd.DataFrame({"date": new_index, "period": range(1, 242)}), on="date")
        DT.set_index("date", inplace=True)
        DT.sort_index(inplace=True)
        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        DT = pd.merge_asof(DT, dt, on="date", by="DTID")
        DT.set_index(["DTID", "date"], inplace=True)
        DT.sort_index(inplace=True)
        DT["ret"] = DT["RI"] / DT["RI"].groupby("DTID").shift(1) - 1

        Seas = DT.loc[DT["period"].isin(range(12, 241, 12))].groupby("DTID")["ret"].mean()
        Seas1A = DT.loc[DT["period"].isin([12])].groupby("DTID")["ret"].mean()
        Seas1N = DT.loc[DT["period"].isin(range(1, 11))].groupby("DTID")["ret"].mean()
        Seas2t5A = DT.loc[DT["period"].isin([24, 36, 48, 60])].groupby("DTID")["ret"].mean()
        Seas2t5N = (
            DT.loc[DT["period"].isin(list(set(range(13, 61)) - set([24, 36, 48, 60])))]
            .groupby("DTID")["ret"]
            .mean()
        )
        Seas6t10A = DT.loc[DT["period"].isin([72, 84, 96, 108, 120])].groupby("DTID")["ret"].mean()
        Seas6t10N = (
            DT.loc[DT["period"].isin(list(set(range(61, 121)) - set([72, 84, 96, 108, 120])))]
            .groupby("DTID")["ret"]
            .mean()
        )
        Seas11t15A = DT.loc[DT["period"].isin([132, 144, 156, 168, 180])].groupby("DTID")["ret"].mean()
        Seas11t15N = (
            DT.loc[DT["period"].isin(list(set(range(121, 181)) - set([132, 144, 156, 168, 180])))]
            .groupby("DTID")["ret"]
            .mean()
        )
        Seas16t20A = DT.loc[DT["period"].isin([192, 204, 216, 228, 240])].groupby("DTID")["ret"].mean()
        Seas16t20N = (
            DT.loc[DT["period"].isin(list(set(range(181, 241)) - set([192, 204, 216, 228, 240])))]
            .groupby("DTID")["ret"]
            .mean()
        )

        result = pd.DataFrame(
            {
                "Seas": Seas,
                "Seas1A": Seas1A,
                "Seas1N": Seas1N,
                "Seas2t5A": Seas2t5A,
                "Seas2t5N": Seas2t5N,
                "Seas6t10A": Seas6t10A,
                "Seas6t10N": Seas6t10N,
                "Seas11t15A": Seas11t15A,
                "Seas11t15N": Seas11t15N,
                "Seas16t20A": Seas16t20A,
                "Seas16t20N": Seas16t20N,
            }
        )
        return result[Out]


class ResidualMomentum(SignalClass):
    """
    11-Month Residual Momentum: Blitz, Huij, and Martens (2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -37, "end": 0, "items": ["RI", "MC"]}}
        self.Output = ["ResidMom"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]

        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(37)]
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

        def MomRes(g):
            """
            function that computes residual 2-12 momentum
            """
            if g.shape[0] < 36:
                return np.nan
            else:
                y = g["r"].values
                X = g[["ones", "rm"]].values
                resid = (y - X.dot(np.linalg.pinv(X).dot(y)))[24:35]
                std = resid.std()
                if std == 0:
                    return np.nan
                else:
                    return (1 + resid).prod() / std

        momres = DT.groupby("DTID").apply(MomRes)
        result = pd.DataFrame({"ResidMom": momres})
        return result[Out]


class IdiosyncraticRisk(SignalClass):
    """
    Idiosyncratic Risk: Ang et al. (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -1, "end": 0, "items": ["r", "MC"]}}
        self.Output = ["IdioRisk"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt["ones"] = 1
        mkt = dt.groupby("date").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt})
        dt.reset_index(inplace=True)
        dt = dt.merge(mkt, on="date")

        def GetResiduals(g):
            """
            function that computes residuals after adjusting for market returns (originally FF3)
            """
            if g.shape[0] < 15:
                return np.nan
            else:
                y = g["r"].values
                X = g[["ones", "rm"]].values
                resid = y - X.dot(np.linalg.pinv(X).dot(y))
                return resid.std()

        ideorisk = dt.groupby("DTID").apply(GetResiduals)
        result = pd.DataFrame({"IdioRisk": ideorisk})
        return result[Out]


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

        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(61)]
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


class DownsideBeta(SignalClass):
    """
    Downside Beta: Ang, Chen, and Xing (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -12, "end": 0, "items": ["r", "MC"]}}
        self.Output = ["DownBeta"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)
        dt["ones"] = 1
        mkt = dt.groupby("date").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt})
        dt.reset_index(inplace=True)
        dt = dt.merge(mkt, on="date")
        dt = dt.loc[dt["rm"] < mkt["rm"].mean()].copy()

        def GetBeta(g):
            """
            function that computes loading on market returns
            """
            if g.shape[0] < 50:
                return np.nan
            else:
                y = g["r"].values
                X = g[["ones", "rm"]].values
                beta = np.linalg.pinv(X).dot(y)
                return beta[1]

        downbeta = dt.groupby("DTID").apply(GetBeta)
        result = pd.DataFrame({"DownBeta": downbeta})
        return result[Out]


class TailRisk(SignalClass):
    """
    Tail Risk: Kelly and Jiang (2014)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -121, "end": 0, "items": ["r", "RI"]}}
        self.Output = ["TailRisk"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt.dropna(inplace=True)

        # get tail risk index for each month
        def GetTailRiskIndex(x):
            quant = x.quantile(0.05)
            return (np.log(x[x < quant] / quant)).mean()

        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        for month in range(130):
            dt.loc[dt.index.get_level_values("date") < DM.Date - relativedelta(months=month), "month"] = month + 1
        TailRiskIndex = pd.DataFrame({"TailRisk": dt.groupby("month")["r"].apply(GetTailRiskIndex)})

        dt = dt.loc[dt.index.get_level_values("date") > DM.Date - relativedelta(months=120)].copy()
        # subset to firms that are available in the last month
        RecentObs = dt.loc[dt.index.get_level_values("date") > DM.Date - relativedelta(months=1), "DTID"]
        dt = dt.loc[dt["DTID"].isin(RecentObs)].copy()

        # get monthly returns for the individual stocks
        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(121)]
        index = pd.MultiIndex.from_product([dt["DTID"].unique(), new_index], names=["DTID", "date"])
        DT = pd.DataFrame(index=index).reset_index()
        DT = pd.merge(DT, pd.DataFrame({"date": new_index, "month": range(1, 122)}), on="date")
        DT.set_index("date", inplace=True)
        DT.sort_index(inplace=True)
        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        DT = pd.merge_asof(DT, dt[["RI", "DTID"]], on="date", by="DTID")
        DT.set_index(["DTID", "date"], inplace=True)
        DT.sort_index(inplace=True)
        DT["r"] = DT["RI"] / DT["RI"].groupby("DTID").shift(1) - 1
        del DT["RI"]
        DT.dropna(inplace=True)
        DT.reset_index(inplace=True)
        DT = DT.merge(TailRiskIndex, on="month")
        DT["ones"] = 1

        def GetBeta(g):
            """
            function that computes loading on market returns
            """
            if g.shape[0] < 36:
                return np.nan
            else:
                y = g["r"].values
                X = g[["ones", "TailRisk"]].values
                beta = np.linalg.pinv(X).dot(y)
                return beta[1]

        tailrisk = DT.groupby("DTID").apply(GetBeta)
        result = pd.DataFrame({"TailRisk": tailrisk})
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

        # subset to firms that are available in the last month
        RecentObs = dt.loc[
            dt.index.get_level_values("date") > DM.Date - relativedelta(months=1)
        ].index.get_level_values("DTID")
        dt = dt.loc[dt.index.get_level_values("DTID").isin(RecentObs)].copy()

        dt["FirstYear"] = 0
        dt.loc[dt.index.get_level_values("date") > DM.Date - relativedelta(months=12), "FirstYear"] = 1
        dt.dropna(inplace=True)
        dt["ones"] = 1
        mkt = dt.groupby("date").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt})
        dt.reset_index(inplace=True)
        dt = dt.merge(mkt, on="date")
        dt.set_index(["DTID", "date"], inplace=True)
        dt["r3"] = dt["r"] + dt["r"].groupby("DTID").shift(1) + dt["r"].groupby("DTID").shift(2)
        dt["rm3"] = dt["rm"] + dt["rm"].groupby("DTID").shift(1) + dt["rm"].groupby("DTID").shift(2)

        # subset to firms with enough observations in the last year
        last_year = dt.loc[dt["FirstYear"] == 1, "r"].groupby("DTID").count()
        dt = dt.loc[dt.index.get_level_values("DTID").isin(last_year[last_year >= 100].index)].copy()

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


class VolumeTrend(SignalClass):
    """
    Volume Trend: Haugen and Baker (1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -60, "end": 0, "items": ["VOL", "PRC"]}}
        self.Output = ["VolTrend"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        dt.dropna(inplace=True)
        dt["month"] = -np.ceil((DM.Date - dt.index.get_level_values("date")).days / 30)
        vol = dt.groupby(["DTID", "month"])["VOL"].sum()
        vol = pd.DataFrame({"VOL": vol})
        vol["ones"] = 1
        vol.reset_index(inplace=True)

        def VolTrend(g):
            """
            function that computes trend of monthly volume
            """
            if g.shape[0] < 36:
                return np.nan
            else:
                y = g["VOL"].values
                X = g[["ones", "month"]].values
                beta = np.linalg.pinv(X).dot(y)
                return beta[1] / y.mean()

        voltrend = vol.groupby("DTID").apply(VolTrend)
        result = pd.DataFrame({"VolTrend": voltrend})
        return result[Out]


class LiquidityRisk(SignalClass):
    """
    Liquidity Risk: Arachya and Pedersen (2005)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"dt": {"start": -121, "end": 0, "items": ["r", "VOL", "PRC", "MC", "RI"]}}
        self.Output = ["LB1", "LB2", "LB3", "LB4", "LB5"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        dt = FetchedData["dt"]
        dt["VOL"] = dt["VOL"] * dt["PRC"]
        del dt["PRC"]
        dt.dropna(inplace=True)

        # get monthly returns for the individual stocks
        new_index = [DM.Date + pd.offsets.DateOffset(months=-mnth) for mnth in range(121)]
        index = pd.MultiIndex.from_product(
            [dt.index.get_level_values("DTID").unique(), new_index], names=["DTID", "date"]
        )
        DT = pd.DataFrame(index=index).reset_index()
        DT = pd.merge(DT, pd.DataFrame({"date": new_index, "month": range(-1, -122, -1)}), on="date")
        DT.set_index("date", inplace=True)
        DT.sort_index(inplace=True)
        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        DT = pd.merge_asof(DT, dt[["RI", "MC", "DTID"]], on="date", by="DTID")
        DT.set_index(["DTID", "date"], inplace=True)
        DT["r"] = DT["RI"] / DT["RI"].groupby("DTID").shift(1) - 1
        del DT["RI"]
        DT.dropna(inplace=True)
        DT.reset_index(inplace=True)
        DT["ones"] = 1

        # get monthly liquidity measure
        for m in range(130):
            dt.loc[dt.index.get_level_values("date") <= DM.Date - relativedelta(months=m), "month"] = -(m + 1)
        dt.reset_index(inplace=True)
        dt.set_index(["DTID", "month"], inplace=True)
        liq = (
            8
            * (dt["r"].groupby(["DTID", "month"]).std() ** (2 / 3))
            / (dt["VOL"].groupby(["DTID", "month"]).sum() ** (1 / 3))
        )
        liq[dt["r"].groupby(["DTID", "month"]).count() < 10] = np.nan
        liq.replace(np.inf, 0.05, inplace=True)
        liq = pd.DataFrame({"liq": liq})
        DT = DT.merge(liq, on=["DTID", "month"])

        # get market-wide measures of liquidity and returns
        mkt = DT.groupby("month").apply(lambda x: (x["r"] * x["MC"]).sum() / x["MC"].sum())
        mktliq = DT.groupby("month").apply(lambda x: (x["liq"] * x["MC"]).sum() / x["MC"].sum())
        mkt = pd.DataFrame({"rm": mkt, "liq": mktliq})

        def LiqInnovation(g):
            """
            function that computes unexplected innovations in liquidity
            """
            y = g["liq"].values
            X = g[["ones", "l.liq", "l2.liq"]].values
            resid = y - X.dot(np.linalg.pinv(X).dot(y))
            return pd.Series(resid, index=g.index.get_level_values("month"))

        mkt["l.liq"] = mkt["liq"].shift(1)
        mkt["l2.liq"] = mkt["liq"].shift(2)
        mkt["ones"] = 1
        mkt.dropna(inplace=True)
        mkt["mktliq"] = LiqInnovation(mkt)

        # join back to stock-month data
        DT = DT.merge(mkt[["rm", "mktliq"]], on="month", how="left")
        DT.set_index(["DTID", "month"], inplace=True)

        # subset to firms that are available in the last month
        RecentObs = DT.loc[DT.index.get_level_values("month") == -1].index.get_level_values("DTID")
        DT = DT.loc[DT.index.get_level_values("DTID").isin(RecentObs)].copy()

        # compute liquidity innovations for the individual
        DT["l.liq"] = DT["liq"].groupby("DTID").shift(1)
        DT["l2.liq"] = DT["liq"].groupby("DTID").shift(2)
        DT.dropna(inplace=True)

        # discard firms with less than 24 observations
        DT = DT.loc[DT["liq"].groupby(by="DTID").transform("count") >= 24].copy()

        DT["LiqInov"] = DT.groupby("DTID").apply(LiqInnovation)

        # compute liquidity betas
        DT = DT.loc[DT["date"] > DM.Date - relativedelta(months=60), ["LiqInov", "rm", "mktliq", "r"]].copy()

        # discard firms with less than 24 observations
        DT = DT.loc[DT["LiqInov"].groupby(by="DTID").transform("count") >= 24].copy()

        def LiqBetas(g):
            """
            function that computes liquidity betas
            """
            denom = (g["rm"] - g["mktliq"]).var()
            lb1 = g["r"].cov(g["rm"]) / denom
            lb2 = g["LiqInov"].cov(g["mktliq"]) / denom
            lb3 = g["r"].cov(g["mktliq"]) / denom
            lb4 = g["LiqInov"].cov(g["rm"]) / denom
            lb5 = lb1 + lb2 - lb3 - lb4
            return pd.Series(
                {"LB1": lb1, "LB2": lb2, "LB3": lb3, "LB4": lb4, "LB5": lb5},
                index=["LB1", "LB2", "LB3", "LB4", "LB5"],
            )

        result = DT.groupby("DTID").apply(LiqBetas)
        return result[Out]
