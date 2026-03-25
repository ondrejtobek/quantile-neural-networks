from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from Signals.SignalClass import SignalClass


class BookToMarket(SignalClass):
    """
    Book Equity / Market Equity: FF (JF 1992)
    Because of changes in the treatment of deferred taxes described in FASB 109, no longer
    add Deferred Taxes and Investment Tax Credit to BE.
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": ["pstkrv", "pstkl", "pstk", "seq", "ceq", "lt", "at", "MC"],
            }
        }
        self.Output = ["BM"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        bm = be / FT["MC"]
        result = pd.DataFrame({"BM": np.log(bm)})
        result = result.groupby("FTID").last()
        return result[Out]


class Accruals(SignalClass):
    """
    The orginal Accruals from Sloan 1996
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["act", "che", "lct", "dlc", "txp", "dp", "at"]}}
        self.Output = ["Accr"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dca = FT["act"] - FT["act"].groupby(by="FTID").shift(1)
        dcash = FT["che"] - FT["che"].groupby(by="FTID").shift(1)
        dcl = FT["lct"] - FT["lct"].groupby(by="FTID").shift(1)
        dstd = FT["dlc"] - FT["dlc"].groupby(by="FTID").shift(1)
        dtp = FT["txp"] - FT["txp"].groupby(by="FTID").shift(1)
        aa = (FT["at"] + FT["at"].groupby(by="FTID").shift(1)) / 2
        accr = (
            (dca.fillna(0) - dcash.fillna(0)) - (dcl.fillna(0) - dstd.fillna(0) - dtp.fillna(0)) - FT["dp"]
        ) / aa
        result = pd.DataFrame({"Accr": accr})
        result = result.groupby("FTID").last()
        return result[Out]


class AccrualsAndBM(SignalClass):
    """
    M/B and Accruals, Barton and Kim (QFA 2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Output = ["MBaAC"]
        self.DependsOn = ["BookToMarket", "Accruals"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        bm = BookToMarket().CreateSignal(DM)["BM"]
        accr = Accruals().CreateSignal(DM)["Accr"]
        top_mb_accr = ((accr < accr.quantile(0.2)) & (bm > bm.quantile(0.8))) * 1
        bot_mb_accr = ((accr > accr.quantile(0.8)) & (bm < bm.quantile(0.2))) * 1
        mb_accr = top_mb_accr - bot_mb_accr
        result = pd.DataFrame({"MBaAC": mb_accr})
        return result[Out]


class AssetGrowth(SignalClass):
    """
    Asset Growth
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["at"]}}
        self.Output = ["AGr"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ag = ((FT["at"] / FT["at"].groupby(by="FTID").shift(1))) - 1
        result = pd.DataFrame({"AGr": ag})
        result = result.groupby("FTID").last()
        return result


class SolimanFundamental(SignalClass):
    """
    set of anomalies from Soliman (AR 2008)
    Asset Turnover: ATurn | prc > 5
        changed definition to anomalies paper
    Change in Asset Turnover: dATurn
    Change in Profit Margin: dPM
    Return on Net Operating Assets: RNOA
    Profit Margin: PM
    Noncurrent Operating Assets Changes
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -42,
                "end": 0,
                "items": ["at", "che", "lt", "dltt", "dlc", "lct", "mib", "sale", "oiadp", "act", "ivaeq"],
            }
        }
        self.Output = ["ATurn", "dATurn", "dPM", "RNOA", "PM", "dNOA", "dNWC"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        oa = FT["at"] - FT["che"]
        ol = FT["lt"] - FT["dltt"].fillna(0) - FT["dlc"].fillna(0) - FT["mib"].fillna(0)
        noa = oa - ol
        noa_lag = noa.groupby(by="FTID").shift(1)
        aturn = FT["sale"] / ((noa + noa_lag) / 2)
        neg_noa = np.nan ** (noa < 0)  # firms with negative NOA
        neg_oi = np.nan ** (FT["oiadp"] < 0)  # firms with negative operating income
        aturn = aturn * neg_oi * neg_noa
        daturn = (aturn - aturn.groupby(by="FTID").shift(1)) * neg_oi * neg_noa
        dpm = (FT["oiadp"] / FT["sale"]) - (FT["oiadp"] / FT["sale"]).groupby(by="FTID").shift(1)
        dpm = dpm * neg_oi * neg_noa
        rnoa = FT["oiadp"] / noa_lag * neg_oi * neg_noa
        pm = FT["oiadp"] / FT["sale"]
        nca = FT["at"] - FT["act"] - FT["ivaeq"].fillna(0)
        ncl = FT["lt"] - FT["lct"] - FT["dltt"]
        ncoa = (nca - ncl) / FT["at"].groupby(by="FTID").shift(1)
        ncoa = ncoa * neg_oi * neg_noa
        ca = FT["act"] - FT["che"]
        cl = FT["lct"] - FT["dlc"]
        nwcch = ((ca - cl) - (ca.groupby(by="FTID").shift(1) - cl.groupby(by="FTID").shift(1))) / FT["at"].groupby(
            by="FTID"
        ).shift(2)
        nwcch = nwcch * neg_oi * neg_noa
        result = pd.DataFrame(
            {
                "ATurn": aturn,
                "dATurn": daturn,
                "dPM": dpm,
                "RNOA": rnoa,
                "PM": pm,
                "dNOA": ncoa,
                "dNWC": nwcch,
            }
        )
        result = result.groupby("FTID").last()
        return result[Out]


class CashFlowMV(SignalClass):
    """
    Cash Flow / Market Value of Equity: LSV (JF 1994) | NYSE/AMEX only
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["ib", "dp", "MC"]}}
        self.Output = ["CFoMV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        cfmv = (FT["ib"] + FT["dp"]) / FT["MC"]
        result = pd.DataFrame({"CFoMV": cfmv})
        result = result.groupby("FTID").last()
        return result[Out]


class DebtIssuance(SignalClass):
    """
    Debt Issuance: Spiess and Affleck-Graves (JFE 1999)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["dltis"]}}
        self.Output = ["DI"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        di = (FT["dltis"] > 0) * 1
        di.fillna(0, inplace=True)
        result = pd.DataFrame({"DI": di})
        result = result.groupby("FTID").last()
        return result[Out]


class EarningsConsistency(SignalClass):
    """
    Earnings Consistency: Alwathainani (BAR 2009) | constraints.
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -102, "end": 0, "items": ["epspx"]}}
        self.Output = ["EConsit"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        eg1 = (FT["epspx"] - FT["epspx"].groupby(by="FTID").shift(1)) / (
            (abs(FT["epspx"].groupby(by="FTID").shift(1)) + abs(FT["epspx"].groupby(by="FTID").shift(2))) / 2
        )
        # "the annual growth in this measure is deleted if its absolute value is greater than 6 and EPS for the
        # current and prior one period (i.e., EPSt and EPSt − 1) have opposite signs"
        # delete if eps positive & negative
        cond1 = np.nan ** ((FT["epspx"] * (FT["epspx"].groupby(by="FTID").shift(1))) < 0)
        # delete if absolute growth larger than 6
        cond2 = np.nan ** (abs(eg1) > 6)
        eg1 = eg1 * cond1 * cond2
        eg2 = eg1.groupby(by="FTID").shift(1)
        eg3 = eg1.groupby(by="FTID").shift(2)
        eg4 = eg1.groupby(by="FTID").shift(3)
        eg5 = eg1.groupby(by="FTID").shift(4)
        eg = ((1 + eg1) * (1 + eg2) * (1 + eg3) * (1 + eg4) * (1 + eg5)) ** (1 / 5) - 1
        # geometric average makes no sense because of the negative values...
        result = pd.DataFrame({"EConsit": eg})
        result = result.groupby("FTID").last()
        return result[Out]


class ComponentsOfBM(SignalClass):
    """
    Enterprise Component of Book/Price: Penman, Richardson, and Tuna (JAR 2007)
    Leverage Component of Book/Price: Penman, Richardson, and Tuna (JAR 2007)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": [
                    "pstkrv",
                    "pstkl",
                    "pstk",
                    "seq",
                    "ceq",
                    "lt",
                    "at",
                    "MC",
                    "che",
                    "tstkp",
                    "dvpa",
                    "dlc",
                    "dltt",
                ],
            }
        }
        self.Output = ["ECoBP", "LCoBP"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        nd = (
            FT["dltt"]
            + FT["dlc"]
            + FT["pstk"].fillna(0)
            + FT["dvpa"].fillna(0)
            - FT["tstkp"].fillna(0)
            - FT["che"]
        )
        ebp = (be + nd) / (nd + FT["MC"])
        bpebp = (
            be / FT["MC"] - ebp + (FT["tstkp"].fillna(0) - FT["dvpa"].fillna(0)).divide(np.log(FT["MC"])).fillna(0)
        )
        result = pd.DataFrame({"ECoBP": ebp, "LCoBP": bpebp})
        result = result.groupby("FTID").last()
        return result[Out]


class EnterpriseMultiple(SignalClass):
    """
    Enterprise Multiple: Loughran and Wellman (JFQA 2011)
    """

    def __init__(self):
        SignalClass.__init__(self)

        self.Input = {"FT": {"start": -42, "end": 0, "items": ["pstk", "oibdp", "MC", "che", "dlc", "dltt"]}}
        self.Output = ["EM"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ev = FT["MC"] + FT["dltt"] + FT["dlc"] + FT["pstk"].fillna(0) - FT["che"]
        age = FT["oibdp"].groupby(by="FTID").count()
        em = ev / FT["oibdp"] * np.nan ** (age < 2)
        result = pd.DataFrame({"EM": em})
        result = result.groupby("FTID").last()
        return result[Out]


class GrossProfitability(SignalClass):
    """
    Gross Profitability: Novy-Marx (JFE 2013)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["revt", "cogs", "at"]}}
        self.Output = ["GP"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        gp = (FT["revt"] - FT["cogs"]) / FT["at"].groupby(by="FTID").shift(1)
        result = pd.DataFrame({"GP": gp})
        result = result.groupby("FTID").last()
        return result[Out]


class GrowthInInventory(SignalClass):
    """
    Growth in Inventory (dINVoAvgA): Thomas and Zhang (RAS 2002)
    Inventory change (dINVoLagA): Thomas and Zhang (RAS 2002)
    Inventory growth (InventGr): Belo and Lin (2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["invt", "at"]}}
        self.Output = ["dINVoAvgA", "dINVoLagA", "InventGr"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dinv = (
            (FT["invt"] - FT["invt"].groupby(by="FTID").shift(1))
            / (FT["at"] + FT["at"].groupby(by="FTID").shift(1))
            / 2
        )
        dinv2 = (
            (FT["invt"] - FT["invt"].groupby("FTID").shift(1))
            / FT["at"].groupby(by="FTID").shift(1)
            * (np.nan ** (FT["invt"] + FT["invt"].groupby("FTID").shift(1) == 0))
        )
        invgr = (FT["invt"] - FT["invt"].groupby("FTID").shift(1)) / FT["invt"].groupby("FTID").shift(1)
        result = pd.DataFrame({"dINVoAvgA": dinv, "dINVoLagA": dinv2, "InventGr": invgr})
        result = result.groupby("FTID").last()
        return result[Out]


class GrowthInLTNOA(SignalClass):
    """
    Growth in LTNOA: Fairfield, Whisenant, and Yohn (AR 2003) (Growth in Net Operating Assets minus Accruals)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["at", "che", "lt", "dltt", "dlc", "mib"]}}
        self.Output = ["dLTNOA"]
        self.DependsOn = ["Accruals"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        oa = FT["at"] - FT["che"]
        ol = FT["lt"] - FT["dltt"].fillna(0) - FT["dlc"].fillna(0) - FT["mib"].fillna(0)
        noa = oa - ol
        noa_lag = noa.groupby(by="FTID").shift(1)
        dnoa = noa - noa_lag
        accr = Accruals().CreateSignal(DM)["Accr"]
        dltnoa = dnoa / (FT["at"] + FT["at"].groupby(by="FTID").shift(1)) / 2 - accr
        result = pd.DataFrame({"dLTNOA": dltnoa})
        result = result.groupby("FTID").last()
        return result[Out]


class Investment(SignalClass):
    """
    Investment: Titman, Wei, and Xie (JFQA 2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -54, "end": 0, "items": ["capx", "sale"]}}
        self.Output = ["Invest"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        capex_sales = FT["capx"] / FT["sale"]
        invest = (FT["capx"] / FT["sale"]) / (
            (
                capex_sales.groupby("FTID").shift(1)
                + capex_sales.groupby("FTID").shift(2)
                + capex_sales.groupby("FTID").shift(3)
            )
            / 3
        )
        revt10 = np.nan ** (FT["sale"] < 10)  # revenues at least 10M
        invest = invest * revt10
        result = pd.DataFrame({"Invest": invest})
        result = result.groupby("FTID").last()
        return result[Out]


class FScore(SignalClass):
    """
    F-Score: Piotroski (AR 2000) A16
    different definitions of quintile in international sample
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": [
                    "ni",
                    "oancf",
                    "at",
                    "dltt",
                    "act",
                    "lct",
                    "sstk",
                    "pstk",
                    "oiadp",
                    "sale",
                ],
            }
        }
        self.Output = ["FSc"]
        self.DependsOn = ["BookToMarket", "SolimanFundamental"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        bm = BookToMarket().CreateSignal(DM)["BM"]
        bm_top = np.nan ** ((bm < bm.quantile(0.8)) * 1)
        daturn = SolimanFundamental().CreateSignal(DM)["dATurn"]
        f1 = (FT["ni"] > 0) * 1
        f2 = (FT["oancf"] > 0) * 1
        f3 = ((FT["ni"] / FT["at"]) - (FT["ni"] / FT["at"]).groupby(by="FTID").shift(1) > 0) * 1
        f4 = ((FT["ni"] - FT["oancf"]) < 0) * 1
        f5 = ((FT["dltt"] / FT["at"]) - (FT["dltt"] / FT["at"]).groupby(by="FTID").shift(1) < 0) * 1
        f6 = ((FT["act"] / FT["lct"]) - (FT["act"] / FT["lct"]).groupby(by="FTID").shift(1) > 0) * 1
        f7 = ((FT["sstk"] - (FT["pstk"] - FT["pstk"].groupby(by="FTID").shift(1)).fillna(0)).fillna(0) <= 0) * 1
        f8 = ((FT["oiadp"] / FT["sale"]) - (FT["oiadp"] / FT["sale"]).groupby(by="FTID").shift(1) > 0) * 1
        f9 = (daturn > 0) * 1
        f_score = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        f_score = f_score * bm_top
        result = pd.DataFrame({"FSc": f_score})
        result = result.groupby("FTID").last()
        return result[Out]


class HerfindahlIndex(SignalClass):
    """
    Herfindahl Index: Hou and Robinson (JF 2006) | can be calculated using sales, ta or be
        exclude regulated industries - " Removing these industries has no material effect on our findings"
        -> better ignore it
    Industry Concentration Sales: Hou and Robinson (2006)
    Industry Concentration Assets: Hou and Robinson (2006)
    Industry Concentration Book Equity: Hou and Robinson (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -54,
                "end": 0,
                "items": ["at", "sale", "pstkrv", "pstkl", "pstk", "seq", "ceq", "lt", "INDM3"],
            }
        }
        self.Output = ["HI", "HIAT", "HIBE"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]

        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        FT["be"] = be

        FT1 = FT[["be", "at", "sale", "INDM3"]].copy()
        FT1.reset_index(inplace=True)
        FT1["year"] = np.nan
        FT1.loc[FT1.date > DM.Date - relativedelta(months=36), "year"] = 3
        FT1.loc[FT1.date > DM.Date - relativedelta(months=24), "year"] = 2
        FT1.loc[FT1.date > DM.Date - relativedelta(months=12), "year"] = 1
        FT1.dropna(subset=["INDM3", "year"], inplace=True)

        h_sales = (
            FT1[["sale", "INDM3", "year"]]
            .groupby(["INDM3", "year"])
            .apply(lambda x: ((x["sale"] / x["sale"].sum()) ** 2).sum())
            .groupby("INDM3")
            .mean()
        )
        h_be = (
            FT1[["be", "INDM3", "year"]]
            .groupby(["INDM3", "year"])
            .apply(lambda x: ((x["be"] / x["be"].sum()) ** 2).sum())
            .groupby("INDM3")
            .mean()
        )
        h_at = (
            FT1[["at", "INDM3", "year"]]
            .groupby(["INDM3", "year"])
            .apply(lambda x: ((x["at"] / x["at"].sum()) ** 2).sum())
            .groupby("INDM3")
            .mean()
        )

        FT1 = FT1.merge(pd.DataFrame({"HI": h_sales, "HIAT": h_at, "HIBE": h_be}), on=["INDM3"], how="left")
        result = FT1[["date", "FTID", "HI", "HIAT", "HIBE"]]
        result.set_index(["FTID", "date"], inplace=True)
        result = result.groupby("FTID").last()
        return result[Out]


class NetOperatingAssets(SignalClass):
    """
    Hirshleifer et al. (JAE 2004)
    Net Operating Assets: NOA
    Changes in Net Operating Assets: ChNOA
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["at", "che", "lt", "mib", "dlc", "dltt"]}}
        self.Output = ["NOA", "ChNOA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        oa = FT["at"] - FT["che"]
        ol = FT["lt"] - FT["dltt"] - FT["dlc"] - FT["mib"].fillna(0)
        lta = FT["at"].groupby(by="FTID").shift(1)
        noa = oa - ol
        chnoa = (noa - noa.groupby("FTID").shift(1)) / lta
        noa = noa / lta
        result = pd.DataFrame({"NOA": noa, "ChNOA": chnoa})
        result = result.groupby("FTID").last()
        return result[Out]


class OperatingLeverage(SignalClass):
    """
    Operating Leverage: Novy-Marx (ROF 2010)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["xsga", "cogs", "at"]}}
        self.Output = ["OperLvrg"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        oper_lev = (FT["xsga"] + FT["cogs"]) / FT["at"]
        result = pd.DataFrame({"OperLvrg": oper_lev})
        result = result.groupby("FTID").last()
        return result[Out]


class OrgCapital(SignalClass):
    """
    Eisfeldt and Papanikolaou (JF 2013)
    Org. Capital: OrgCap
    Industry-adjusted Organizational Capital-to-assets: IAOrgCapital
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -258, "end": 0, "items": ["xsga", "CPI", "at", "INDM3"]}}
        self.Output = ["OrgCap", "IAOrgCap"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        # subset to firms that are available in the last one and half year
        RecentObs = FT.loc[
            FT.index.get_level_values("date") > DM.Date - relativedelta(months=18)
        ].index.get_level_values("FTID")
        FT = FT.loc[FT.index.get_level_values("FTID").isin(RecentObs)].copy()
        # compute the organization capital
        FT["xsgaDefl"] = (FT["xsga"] / FT["CPI"]).fillna(0)
        FT.reset_index(inplace=True)
        FT["Order"] = FT.groupby(by="FTID")["date"].rank()
        FT.set_index(["FTID", "date"], inplace=True)
        FT.loc[FT["Order"] == 1, "xsgaDefl"] = FT["xsgaDefl"] * 4
        FT["xsgaDefl"] = FT["xsgaDefl"] * (0.85 ** (FT.groupby(by="FTID")["Order"].max() - FT["Order"]))
        OrgCap = FT["xsgaDefl"].groupby(by="FTID").sum() / FT["at"].groupby(by="FTID").last()
        OrgCap.replace({0: np.nan, np.inf: np.nan}, inplace=True)
        # compute industry adjusted capital
        FT = pd.DataFrame({"OrgCap": OrgCap, "INDM3": FT["INDM3"].groupby("FTID").last()})
        FT.dropna(inplace=True)
        FT["IAOrgCap"] = (FT[["OrgCap"]] - FT[["OrgCap", "INDM3"]].groupby(["INDM3"]).transform("mean")) / FT[
            ["OrgCap", "INDM3"]
        ].groupby(["INDM3"]).transform("std")
        result = FT[["OrgCap", "IAOrgCap"]]
        return result[Out]


class OScore(SignalClass):
    """
    O-Score (More Financial Distress): Dichev (JFE 1998)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": [
                    "lt",
                    "ni",
                    "at",
                    "dltt",
                    "dlc",
                    "act",
                    "lct",
                    "pi",
                    "dp",
                    "INDM3",
                    "INDM",
                    "CPI",
                ],
            }
        }
        self.Output = ["OSc"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        oeneg = (FT["lt"] > FT["at"]) * 1
        intwo = ((FT["ni"] < 0) & (FT["ni"].groupby(by="FTID").shift(1) < 0)) * 1
        chin = (FT["ni"] - FT["ni"].groupby(by="FTID").shift(1)) / (
            abs(FT["ni"]) + abs(FT["ni"].groupby(by="FTID").shift(1))
        )
        o_score = (
            -1.32
            - 0.407 * np.log(FT["at"] / FT["CPI"])
            + 6.03 * ((FT["dltt"] + FT["dlc"]) / FT["at"])
            - 1.43 * ((FT["act"] - FT["lct"]) / FT["at"])
            + 0.076 * (FT["lct"] / FT["act"])
            - 1.72 * (oeneg)
            - 2.37 * (FT["ni"] / FT["at"])
            - 1.83 * ((FT["pi"] + FT["dp"]) / FT["lt"])
            + 0.285 * (intwo)
            - 0.521 * (chin)
        )
        utilities = np.nan ** ((FT["INDM3"] == "Utilities") | (FT["INDM"] == "Waste, Disposal Svs."))
        o_score = o_score * utilities
        result = pd.DataFrame({"OSc": o_score})
        result = result.groupby("FTID").last()
        return result[Out]


class PercentOperatingAccrual(SignalClass):
    """
    Percent Operating Accrual: Hafzalla, Lundholm, and Van Winkle (AR 2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["ib", "oancf"]}}
        self.Output = ["prcOA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        pca = (FT["ib"] - FT["oancf"]) / abs(FT["ib"])
        result = pd.DataFrame({"prcOA": pca})
        result = result.groupby("FTID").last()
        return result[Out]


class PercentTotalAccrual(SignalClass):
    """
    Percent Total Accrual: Hafzalla, Lundholm, and Van Winkle (AR 2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": ["ib", "oancf", "ivncf", "fincf", "sstk", "prstkc", "dv"],
            }
        }
        self.Output = ["prcTA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        pta = (
            FT["ib"]
            - FT["oancf"]
            - FT["ivncf"]
            - FT["fincf"]
            + FT["sstk"].fillna(0)
            - FT["prstkc"].fillna(0)
            - FT["dv"].fillna(0)
        ) / abs(FT["ib"])
        result = pd.DataFrame({"prcTA": pta})
        result = result.groupby("FTID").last()
        return result[Out]


class RDtoMV(SignalClass):
    """
    R&D / Market Value of Equity: Chan et al. (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["xrd", "MC"]}}
        self.Output = ["RDoMV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        rdm = FT["xrd"] / FT["MC"] * (np.nan ** (FT["xrd"] <= 0))
        result = pd.DataFrame({"RDoMV": rdm})
        result = result.groupby("FTID").last()
        return result[Out]


class ReturnOnEquity(SignalClass):
    """
    Return-on-Equity: Haugen and Baker (JFE 1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": ["ib", "pstkrv", "pstkl", "pstk", "seq", "ceq", "lt", "at"],
            }
        }
        self.Output = ["RoE"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        roe = FT["ib"] / be
        result = pd.DataFrame({"RoE": roe})
        result = result.groupby("FTID").last()
        return result[Out]


class SalesGrowth(SignalClass):
    """
    Sales Growth: LSV (JF 1994) ! 5-year sales growth rank
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -78, "end": 0, "items": ["sale"]}}
        self.Output = ["SalesGr"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]

        for i in range(7):
            FT.loc[FT.index.get_level_values("date") < DM.Date - relativedelta(years=i), "year"] = -(i + 1)

        FT["sales_growth"] = FT["sale"] / FT["sale"].groupby(by="FTID").shift(1) - 1
        sales_growth_rank = FT.groupby(by="year")["sales_growth"].transform(lambda x: x.rank())
        sales_growth_rank2 = sales_growth_rank.groupby(by="FTID").shift(1)
        sales_growth_rank3 = sales_growth_rank.groupby(by="FTID").shift(2)
        sales_growth_rank4 = sales_growth_rank.groupby(by="FTID").shift(3)
        sales_growth_rank5 = sales_growth_rank.groupby(by="FTID").shift(4)
        sg_rank_final = (
            5 * sales_growth_rank
            + 4 * sales_growth_rank2
            + 3 * sales_growth_rank3
            + 2 * sales_growth_rank4
            + 1 * sales_growth_rank5
        ) / 15.0
        result = pd.DataFrame({"SalesGr": sg_rank_final})
        result = result.groupby("FTID").last()
        return result[Out]


class SalesToPrice(SignalClass):
    """
    Sales/Price: Barbee et al (FAJ - 1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["sale", "MC"]}}
        self.Output = ["SaleToMV"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        sp = FT["sale"] / FT["MC"] * (np.nan ** (FT["sale"] <= 0))
        result = pd.DataFrame({"SaleToMV": sp})
        result = result.groupby("FTID").last()
        return result[Out]


class ShareRepurchases(SignalClass):
    """
    Share Repurchases: Ikenberry, Lakonishok, and Vermaelen (JFE 1995)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["prstkc"]}}
        self.Output = ["SR"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        sr = (FT["prstkc"] > 0) * 1
        result = pd.DataFrame({"SR": sr})
        result = result.groupby("FTID").last()
        return result[Out]


class SustainableGrowth(SignalClass):
    """
    Sustainable Growth: Lockwood and Prombutr (JFR 2010)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -54,
                "end": 0,
                "items": ["pstkrv", "pstkl", "pstk", "seq", "ceq", "lt", "at"],
            }
        }
        self.Output = ["SuGr"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        FT["age"] = FT["at"].groupby(by="FTID").count()
        sg = ((be / be.groupby(by="FTID").shift(1)) - 1) * (np.nan ** (FT["age"] < 3))
        result = pd.DataFrame({"SuGr": sg})
        result = result.groupby("FTID").last()
        return result[Out]


class TotalXFIN(SignalClass):
    """
    Total XFIN: Bradshaw, Richardson, and Sloan (JAE 2006) ! net external financing
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": ["sstk", "prstkc", "dv", "dltis", "dltr", "dlcch", "at"],
            }
        }
        self.Output = ["TXFIN"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        xfin = (
            FT["sstk"].fillna(0) - FT["prstkc"].fillna(0) - FT["dv"] + FT["dltis"] - FT["dltr"] + FT["dlcch"]
        ) / FT["at"]
        result = pd.DataFrame({"TXFIN": xfin})
        result = result.groupby("FTID").last()
        return result[Out]


class UnexpectedRDIncreases(SignalClass):
    """
    Unexpected R&D Increases: Eberhart, Maxwell, and Siddique (JF 2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["xrd", "revt", "at"]}}
        self.Output = ["URDI"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        r1 = (FT["xrd"] / FT["revt"] > 0.05) & (FT["xrd"] / FT["at"] > 0.05)
        r2 = (FT["xrd"] / FT["xrd"].groupby(by="FTID").shift(1)) > 1.05
        r3 = (FT["xrd"] / FT["at"]) / ((FT["xrd"] / FT["at"]).groupby(by="FTID").shift(1)) > 1.05
        urdi = (r1 & r2 & r3) * 1
        result = pd.DataFrame({"URDI": urdi})
        result = result.groupby("FTID").last()
        return result[Out]


class ZScore(SignalClass):
    """
    Z-Score (Less Financial Distress): Dichev (JFE 1998) | NYSE only + SIC constraint
    exclude Railroad, Other Transport, and Utilities
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": ["act", "lct", "oiadp", "re", "at", "lt", "sale", "MC", "INDM3", "INDM"],
            }
        }
        self.Output = ["ZSc"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        utilities = np.nan ** ((FT["INDM3"] == "Utilities") | (FT["INDM"] == "Waste, Disposal Svs."))
        z_score = (
            1.2 * ((FT["act"] - FT["lct"]) / FT["at"])
            + 1.4 * (FT["re"] / FT["at"])
            + 3.3 * (FT["oiadp"] / FT["at"])
            + 0.6 * ((FT["MC"] / FT["lt"]))
            + (FT["sale"] / FT["at"])
        ) * utilities
        result = pd.DataFrame({"ZSc": z_score})
        result = result.groupby("FTID").last()
        return result[Out]


class AbarbanellBushee(SignalClass):
    """
    Anomalies from Abarbanell and Bushee (AR 1998)
    ΔCAPEX-ΔIndustry CAPEX: indCAPEX
    ΔSales-ΔInventory: SmI
    ΔSales-ΔSG&A: SmSGA
    ΔSales-ΔAcounts Receivable: dSdAR
    ΔGross Marging-ΔSales: dGMdS
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -42,
                "end": 0,
                "items": ["capx", "sale", "xsga", "invt", "rect", "cogs", "INDM3"],
            }
        }
        self.Output = ["indCAPEX", "SmI", "SmSGA", "dSdAR", "dGMdS"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]

        for i in range(5):
            FT.loc[FT.index.get_level_values("date") < DM.Date - relativedelta(years=i), "year"] = -(i + 1)

        # ΔCAPEX-ΔIndustry CAPEX
        dcapx = (
            FT["capx"] / (FT["capx"].groupby(by="FTID").shift(1) + FT["capx"].groupby(by="FTID").shift(2)) / 2
        ) - 1
        FT["capx_ind"] = FT[["capx", "INDM3", "year"]].groupby(by=["INDM3", "year"]).transform("mean")
        dcapx_ind = (
            FT["capx_ind"]
            / (FT["capx_ind"].groupby(by="FTID").shift(1) + FT["capx_ind"].groupby(by="FTID").shift(2))
            / 2
        ) - 1
        dcapx_adj = dcapx - dcapx_ind
        # ΔSales-ΔInventory
        sales_2y_avg = (FT["sale"].groupby(by="FTID").shift(1) + FT["sale"].groupby(by="FTID").shift(2)) / 2
        inv_2y_avg = (FT["invt"].groupby(by="FTID").shift(1) + FT["invt"].groupby(by="FTID").shift(2)) / 2
        delta_si = ((FT["sale"] - sales_2y_avg) / sales_2y_avg) - ((FT["invt"] - inv_2y_avg) / inv_2y_avg)
        delta_si = delta_si * (np.nan ** (sales_2y_avg <= 0)) * (np.nan ** (inv_2y_avg <= 0))
        # ΔSales-ΔSG&A
        sga_2y_avg = (FT["xsga"].groupby(by="FTID").shift(1) + FT["xsga"].groupby(by="FTID").shift(2)) / 2
        delta_ss = ((FT["sale"] - sales_2y_avg) / sales_2y_avg) - ((FT["xsga"] - sga_2y_avg) / sga_2y_avg)
        delta_ss = delta_ss * (np.nan ** (sales_2y_avg <= 0)) * (np.nan ** (sga_2y_avg <= 0))
        # ΔSales-ΔAcounts Receivable
        rect_2y_avg = (FT["rect"].groupby(by="FTID").shift(1) + FT["rect"].groupby(by="FTID").shift(2)) / 2
        delta_sa = ((FT["sale"] - sales_2y_avg) / sales_2y_avg) - ((FT["rect"] - rect_2y_avg) / rect_2y_avg)
        delta_sa = delta_sa * (np.nan ** (sales_2y_avg <= 0)) * (np.nan ** (rect_2y_avg <= 0))
        # ΔGross Marging-ΔSales
        gm = FT["sale"] - FT["cogs"]
        gm_2y_avg = (gm.groupby(by="FTID").shift(1) + gm.groupby(by="FTID").shift(2)) / 2
        delta_gs = ((gm - gm_2y_avg) / gm_2y_avg) - ((FT["sale"] - sales_2y_avg) / sales_2y_avg)
        delta_gs = delta_gs * (np.nan ** (sales_2y_avg <= 0)) * (np.nan ** (gm_2y_avg <= 0))

        result = pd.DataFrame(
            {
                "indCAPEX": dcapx_adj,
                "SmI": delta_si,
                "SmSGA": delta_ss,
                "dSdAR": delta_sa,
                "dGMdS": delta_gs,
            }
        )
        result = result.groupby("FTID").last()
        return result[Out]


class AssetLiquidity(SignalClass):
    """
    Ortiz-Molina and Phillips (JFQA 2014)
    Asset Liquidity: AL - divided by Assets
    Asset Liquidity 2: AL2 - divided by MV
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": ["act", "che", "intan", "gdwl", "at", "MC", "INDM3", "INDM"],
            }
        }
        self.Output = ["AL", "AL2"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        utilities = np.nan ** ((FT["INDM3"] == "Utilities") | (FT["INDM"] == "Waste, Disposal Svs."))
        tfa = FT["at"] - FT["gdwl"].fillna(0) - FT["intan"].fillna(0)  # tangible fixed assets
        al = FT["che"] + 0.75 * (FT["act"] - FT["che"]) - tfa
        ala = al / FT["at"].groupby(by="FTID").shift(1) * utilities
        alm = al / FT["MC"].groupby(by="FTID").shift(1) * utilities
        result = pd.DataFrame({"AL": ala, "AL2": alm})
        result = result.groupby("FTID").last()
        return result[Out]


class CashToAssets(SignalClass):
    """
    Cash-to-assets: Palazzo (JFE 2012)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["che", "at"]}}
        self.Output = ["CtA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        cta = FT["che"] / FT["at"]
        result = pd.DataFrame({"CtA": cta})
        result = result.groupby("FTID").last()
        return result[Out]


class CashBasedOperatingProfitability(SignalClass):
    """
    Cash-based Operating Profitability: Ball, Gerakos, Linnainmaa, and Nikolaev (JFE 2016)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": [
                    "revt",
                    "cogs",
                    "xsga",
                    "xrd",
                    "rect",
                    "invt",
                    "xpp",
                    "drc",
                    "drlt",
                    "ap",
                    "xacc",
                    "at",
                ],
            }
        }
        self.Output = ["CBOP"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        cop = (
            FT["revt"]
            - FT["cogs"]
            - FT["xsga"]
            + FT["xrd"].fillna(0)
            - (FT["rect"] - FT["rect"].groupby(by="FTID").shift(1)).fillna(0)
            - (FT["invt"] - FT["invt"].groupby(by="FTID").shift(1)).fillna(0)
            - (FT["xpp"] - FT["xpp"].groupby(by="FTID").shift(1)).fillna(0)
            + ((FT["drc"] + FT["drlt"]) - (FT["drc"] + FT["drlt"]).groupby(by="FTID").shift(1)).fillna(0)
            + (FT["ap"] - FT["ap"].groupby(by="FTID").shift(1)).fillna(0)
            + (FT["xacc"] - FT["xacc"].groupby(by="FTID").shift(1)).fillna(0)
        ) / FT["at"]
        result = pd.DataFrame({"CBOP": cop})
        result = result.groupby("FTID").last()
        return result[Out]


class EarningsSmoothness(SignalClass):
    """
    Earnings Smoothness: Francis, LaFond, Olsson, and Schipper (AR 2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -138, "end": 0, "items": ["act", "che", "lct", "dlc", "ib", "dp", "at"]}}
        self.Output = ["ES"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dca = FT["act"] - FT["act"].groupby(by="FTID").shift(1)
        dcash = FT["che"] - FT["che"].groupby(by="FTID").shift(1)
        dcl = FT["lct"] - FT["lct"].groupby(by="FTID").shift(1)
        dstd = FT["dlc"] - FT["dlc"].groupby(by="FTID").shift(1)
        FT["ela"] = FT["ib"] / FT["at"].groupby(by="FTID").shift(1)
        FT["cfoa"] = FT["ib"] - (dca - dcl - dcash + dstd - FT["dp"])
        FT = FT[["ela", "cfoa"]].dropna()
        FT = FT.loc[FT["ela"].groupby(by="FTID").transform("count") >= 10].copy()
        FT = FT.groupby("FTID").tail(10)
        esm = FT["ela"].groupby("FTID").std() / FT["cfoa"].groupby("FTID").std()
        result = pd.DataFrame({"ES": esm})
        result = result.groupby("FTID").last()
        return result[Out]


class HiringRate(SignalClass):
    """
    Hiring rate: Belo, Lin, and Bazdresch (JPE 2014)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["emp", "INDM3", "INDM"]}}
        self.Output = ["HR"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        utilities = np.nan ** ((FT["INDM3"] == "Utilities") | (FT["INDM"] == "Waste, Disposal Svs."))
        hr = (
            (FT["emp"] - FT["emp"].groupby(by="FTID").shift(1))
            / (FT["emp"] + FT["emp"].groupby(by="FTID").shift(1))
            * 2
        )
        hr = hr * utilities
        hr = hr[hr != 0]  # 0 often due to stale information
        result = pd.DataFrame({"HR": hr})
        result = result.groupby("FTID").last()
        return result[Out]


class LaborForceEfficiency(SignalClass):
    """
    Labor Force Efficiency: Abarbanell and Bushee (AR 1998)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["sale", "emp"]}}
        self.Output = ["LFE"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        lfe = ((FT["sale"] / FT["emp"]) / ((FT["sale"] / FT["emp"]).groupby(by="FTID").shift(1))) - 1
        result = pd.DataFrame({"LFE": lfe})
        result = result.groupby("FTID").last()
        return result[Out]


class OperatingProfitsToEquity(SignalClass):
    """
    Operating Profits to Equity: Fama and French (JFE 2015)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -18,
                "end": 0,
                "items": [
                    "xint",
                    "cogs",
                    "xsga",
                    "revt",
                    "pstkrv",
                    "pstkl",
                    "pstk",
                    "seq",
                    "ceq",
                    "lt",
                    "at",
                ],
            }
        }
        self.Output = ["OPoE"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        FT["be"] = be
        FT = FT.dropna(
            how="all", subset=["xint", "cogs", "xsga"]
        )  # at least one from xint, cogs, xsga should not be missing
        ope = (FT["revt"] - FT["cogs"].fillna(0) - FT["xsga"].fillna(0) - FT["xint"].fillna(0)) / FT["be"]
        result = pd.DataFrame({"OPoE": ope})
        result = result.groupby("FTID").last()
        return result[Out]


class OperatingProfitsToAssets(SignalClass):
    """
    Operating Profits to Assets: Ball, Gerakos, Linnainmaa, and Nikolaev (JFE 2016)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["cogs", "xsga", "revt", "xrd", "at"]}}
        self.Output = ["OPoA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ola = (FT["revt"] - FT["cogs"] - FT["xsga"] + FT["xrd"].fillna(0)) / FT["at"]
        result = pd.DataFrame({"OPoA": ola})
        result = result.groupby("FTID").last()
        return result[Out]


class IndustryAdjRealEstateRatio(SignalClass):
    """
    Industry-adjusted Real Estate Ratio: Tuzel (RFS 2010)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["fatb", "fatl", "ppent", "INDM3"]}}
        self.Output = ["IARER"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        FT["rer"] = (FT["fatb"] + FT["fatl"].fillna(0)) / FT["ppent"]
        FT = FT.groupby("FTID").last()
        ind_rer_mean = FT[["rer", "INDM3"]].groupby(by=["INDM3"])["rer"].transform("mean")
        irer = FT["rer"] - ind_rer_mean
        enough_firms = FT[["rer", "INDM3"]].groupby(by=["INDM3"])["INDM3"].transform("count") <= 2
        irer = irer * (np.nan ** (FT["INDM3"] == "Real Estate")) * (np.nan**enough_firms)
        result = pd.DataFrame({"IARER": irer})
        result = result.groupby("FTID").last()
        return result[Out]


class Tangibility(SignalClass):
    """
    Tangibility: Hahn and Lee (JF 2009)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["che", "rect", "invt", "ppegt", "at"]}}
        self.Output = ["TAN"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        tang = (FT["che"] + 0.715 * FT["rect"] + 0.547 * FT["invt"] + 0.535 * FT["ppegt"]) / FT["at"]
        result = pd.DataFrame({"TAN": tang})
        result = result.groupby("FTID").last()
        return result[Out]


class WhitedWuIndex(SignalClass):
    """
    Whited-Wu Index: Whited and Wu (RFS 2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": ["ib", "dp", "at", "dv", "dltt", "INDM3", "INDM", "sale"],
            }
        }
        self.Output = ["WWI"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        cf = (FT["ib"] + FT["dp"]) / FT["at"]
        cf = (1 + cf / 4) ** (1.0 / 4) - 1  # quarterly compounding
        divpos = FT["dv"] > 0
        tltd = FT["dltt"] / FT["at"]
        lnta = np.log(FT["at"])

        FT1 = FT[["sale", "INDM3"]].copy()
        FT1.reset_index(inplace=True)
        FT1["year"] = np.nan
        FT1.loc[FT1.date > DM.Date - relativedelta(months=24), "year"] = 1
        FT1.loc[FT1.date > DM.Date - relativedelta(months=12), "year"] = 2
        FT1.dropna(subset=["INDM3", "year"], inplace=True)
        isg = FT1.groupby(["INDM3", "year"])["sale"].sum()
        isg = isg[FT1.groupby(["INDM3", "year"])["sale"].count() > 2]
        isg = (isg / isg.groupby(by="INDM3").shift(1)) - 1
        isg = isg[isg.index.get_level_values("year") == 2]
        FT = FT.reset_index().merge(pd.DataFrame({"isg": isg}), how="left", on="INDM3").set_index(["FTID", "date"])
        isg = FT["isg"]

        isg = (1 + isg / 4) ** (1.0 / 4) - 1  # quarterly compounding
        sg = (FT["sale"] / FT["sale"].groupby(by="FTID").shift(1)) - 1
        sg = (1 + sg / 4) ** (1.0 / 4) - 1  # quarterly compounding
        wwi = -0.091 * cf - 0.062 * divpos + 0.021 * tltd - 0.044 * lnta + 0.102 * isg - 0.035 * sg
        utilities = np.nan ** ((FT["INDM3"] == "Utilities") | (FT["INDM"] == "Waste, Disposal Svs."))
        wwi = wwi * utilities
        result = pd.DataFrame({"WWI": wwi})
        result = result.groupby("FTID").last()
        return result[Out]


class EarningsToPrice(SignalClass):
    """
    Earnings / Price:  Basu (JF 1977)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["ib", "MC"]}}
        self.Output = ["EoP"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ep = FT["ib"] / FT["MC"] * (np.nan ** (FT["ib"] <= 0))
        result = pd.DataFrame({"EoP": ep})
        result = result.groupby("FTID").last()
        return result[Out]


class Leverage(SignalClass):
    """
    Leverage:  Bhandari (JFE 1988)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["dltt", "dlc", "MC"]}}
        self.Output = ["Lvrg"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        lvrg = (FT["dltt"] + FT["dlc"]) / FT["MC"]
        result = pd.DataFrame({"Lvrg": lvrg})
        result = result.groupby("FTID").last()
        return result[Out]


class PayoutYield(SignalClass):
    """
    Boudoukh, Michaely, Richardson, and Roberts (2007)
    Payout Yield: PY
    Net Payout Yield: NPY
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["dvc", "prstkc", "pstkrv", "sstk", "MC"]}}
        self.Output = ["PY", "NPY"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        py = (FT["dvc"] + FT["prstkc"] - FT["pstkrv"] + FT["pstkrv"].groupby("FTID").shift(1)) / FT["MC"]
        npy = (FT["dvc"] + FT["prstkc"] - FT["sstk"]) / FT["MC"]
        result = pd.DataFrame({"PY": py, "NPY": npy})
        result = result.groupby("FTID").last()
        return result[Out]


class ChPPEIA(SignalClass):
    """
    Changes in PPE and Inventory-to-assets: Lyandres, Sun, and Zhang 2007 RFS
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["ppegt", "invt", "at"]}}
        self.Output = ["ChPPEIA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        chppeia = (
            FT["ppegt"] - FT["ppegt"].groupby("FTID").shift(1) + FT["invt"] - FT["invt"].groupby("FTID").shift(1)
        ) / FT["at"].groupby(by="FTID").shift(1)
        result = pd.DataFrame({"ChPPEIA": chppeia})
        result = result.groupby("FTID").last()
        return result[Out]


class NetFinance(SignalClass):
    """
    Bradshaw, Richardson, and Sloan (2006)
    Net debt finance: NDF
    Net equity finance: NEF
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": ["dltis", "dltr", "dlcch", "at", "sstk", "prstkc", "dv"],
            }
        }
        self.Output = ["NDF", "NEF"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ndf = (FT["dltis"] - FT["dltr"] + FT["dlcch"]) / ((FT["at"] + FT["at"].groupby(by="FTID").shift(1)) / 2)
        nef = (FT["sstk"].fillna(0) - FT["prstkc"].fillna(0) - FT["dv"]) / (
            (FT["at"] + FT["at"].groupby(by="FTID").shift(1)) / 2
        )
        result = pd.DataFrame({"NDF": ndf, "NEF": nef})
        result = result.groupby("FTID").last()
        return result[Out]


class CapitalTurnover(SignalClass):
    """
    Capital Turnover: Haugen and Baker (1996)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["sale", "at"]}}
        self.Output = ["CT"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        ct = FT["sale"] / FT["at"].groupby(by="FTID").shift(1)
        result = pd.DataFrame({"CT": ct})
        result = result.groupby("FTID").last()
        return result[Out]


class RichardsonVars(SignalClass):
    """
    variables from Richardson, Sloan, Soliman, and Tuna (2005)
    Change in net non-cash working capital: dNNCWC
    hange in current operating assets: dCOA
    Change in current operating liabilities: dCOL
    Change in net non-current operating assets: dNNCOA
    Change in non-current operating assets: dNCOA
    Change in non-current operating liabilities: dNCOL
    Change in net financial assets: dNFA
    Change in short-term investments: dSTI
    Change in long-term investments: dLTI
    Change in common equity: dCE
    Change in financial liabilities: dFL
    Total accruals: TA
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": [
                    "act",
                    "at",
                    "che",
                    "lct",
                    "dlc",
                    "ivao",
                    "dltt",
                    "lt",
                    "ivst",
                    "pstk",
                    "ceq",
                ],
            }
        }
        self.Output = [
            "dNNCWC",
            "dCOA",
            "dCOL",
            "dNNCOA",
            "dNCOA",
            "dNCOL",
            "dNFA",
            "dSTI",
            "dLTI",
            "dCE",
            "dFL",
            "TA",
        ]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        lta = FT["at"].groupby(by="FTID").shift(1)
        # Change in net non-cash working capital
        coa = FT["act"] - FT["che"]
        col = FT["lct"] - FT["dlc"].fillna(0)
        wc = coa - col
        FT["dNNCWC"] = (wc - wc.groupby("FTID").shift(1)) / lta
        # Change in current operating assets
        FT["dCOA"] = (coa - coa.groupby("FTID").shift(1)) / lta
        # Change in current operating liabilities
        FT["dCOL"] = (col - col.groupby("FTID").shift(1)) / lta
        # Change in net non-current operating assets
        nca = FT["at"] - FT["act"] - FT["ivao"].fillna(0)
        ncl = FT["lt"] - FT["lct"] - FT["dltt"].fillna(0)
        nco = nca - ncl
        FT["dNNCOA"] = (nco - nco.groupby("FTID").shift(1)) / lta
        # Change in non-current operating assets
        FT["dNCOA"] = (nca - nca.groupby("FTID").shift(1)) / lta
        # Change in non-current operating liabilities
        FT["dNCOL"] = (ncl - ncl.groupby("FTID").shift(1)) / lta
        # Change in net financial assets
        fna = (FT["ivst"].fillna(0) + FT["ivao"].fillna(0)) * (
            np.nan ** (FT["ivst"].isnull() & FT["ivao"].isnull())
        )
        fnl = (FT["dltt"].fillna(0) + FT["dlc"].fillna(0) + FT["pstk"].fillna(0)) * (
            np.nan ** (FT["dltt"].isnull() & FT["dlc"].isnull() & FT["pstk"].isnull())
        )
        nfna = fna - fnl
        FT["dNFA"] = (nfna - nfna.groupby("FTID").shift(1)) / lta
        # Change in short-term investments
        FT["dSTI"] = (FT["ivst"] - FT["ivst"].groupby("FTID").shift(1)) / lta
        # Change in long-term investments
        FT["dLTI"] = (FT["ivao"] - FT["ivao"].groupby("FTID").shift(1)) / lta
        # Change in common equity
        FT["dCE"] = (FT["ceq"] - FT["ceq"].groupby("FTID").shift(1)) / lta
        # Change in financial liabilities
        FT["dFL"] = (fnl - fnl.groupby("FTID").shift(1)) / lta
        # Total accruals
        ta = nco + wc + nfna
        FT["TA"] = (ta - ta.groupby("FTID").shift(1)) / lta
        result = FT[
            [
                "dNNCWC",
                "dCOA",
                "dCOL",
                "dNNCOA",
                "dNCOA",
                "dNCOL",
                "dNFA",
                "dSTI",
                "dLTI",
                "dCE",
                "dFL",
                "TA",
            ]
        ]
        result = result.groupby("FTID").last()
        return result[Out]


class AssetsTomarket(SignalClass):
    """
    Assets-to-market: Fama and French (1992)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["MC", "at"]}}
        self.Output = ["AtM"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        atm = FT["at"] / FT["MC"]
        result = pd.DataFrame({"AtM": atm})
        result = result.groupby("FTID").last()
        return result[Out]


class CompositeDebtIssuance(SignalClass):
    """
    Composite Debt Issuance: Lyandres, Sun, and Zhang (2008)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -78, "end": 0, "items": ["dltt", "dlc"]}}
        self.Output = ["CDI"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        debt = FT["dltt"] + FT["dlc"]
        cdi = np.log(debt / debt.groupby("FTID").shift(5) - 1)
        result = pd.DataFrame({"CDI": cdi})
        result = result.groupby("FTID").last()
        return result[Out]


class RDtoSales(SignalClass):
    """
    R&D Expenses-to-sales: Chan, Lakonishok, and Sougiannis (2001)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -18, "end": 0, "items": ["xrd", "sale"]}}
        self.Output = ["RDtS"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        rds = FT["xrd"] / FT["sale"] * (np.nan ** (FT["xrd"] <= 0))
        result = pd.DataFrame({"RDtS": rds})
        result = result.groupby("FTID").last()
        return result[Out]


class RDtoAssets(SignalClass):
    """
    R&D Capital-to-assets: Li (2011)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -66, "end": 0, "items": ["xrd", "at"]}}
        self.Output = ["RDtA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        xrd = FT["xrd"].fillna(0)
        rc = (
            xrd
            + 0.8 * xrd.groupby("FTID").shift(1)
            + 0.6 * xrd.groupby("FTID").shift(2)
            + 0.4 * xrd.groupby("FTID").shift(3)
            + 0.2 * xrd.groupby("FTID").shift(4)
        )
        rda = rc / FT["at"] * (np.nan ** (rc <= 0)) * (np.nan ** (FT["xrd"].isnull()))
        result = pd.DataFrame({"RDtA": rda})
        result = result.groupby("FTID").last()
        return result[Out]


class DurationOfEquity(SignalClass):
    """
    Duration of Equity: Dechow, Sloan, and Soliman (2004)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {"FT": {"start": -30, "end": 0, "items": ["ceq", "ib", "sale", "MC"]}}
        self.Output = ["DurE"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        be0 = FT["ceq"]
        roe0 = FT["ib"] / FT["ceq"].groupby("FTID").shift(1)
        g0 = FT["sale"] / FT["sale"].groupby("FTID").shift(1) - 1
        g1 = 0.06 + 0.24 * g0
        g2 = 0.06 + 0.24 * g1
        g3 = 0.06 + 0.24 * g2
        g4 = 0.06 + 0.24 * g3
        g5 = 0.06 + 0.24 * g4
        g6 = 0.06 + 0.24 * g5
        g7 = 0.06 + 0.24 * g6
        g8 = 0.06 + 0.24 * g7
        g9 = 0.06 + 0.24 * g8
        be1 = be0 * (1 + g1)
        be2 = be0 * (1 + g2)
        be3 = be0 * (1 + g3)
        be4 = be0 * (1 + g4)
        be5 = be0 * (1 + g5)
        be6 = be0 * (1 + g6)
        be7 = be0 * (1 + g7)
        be8 = be0 * (1 + g8)
        be9 = be0 * (1 + g9)
        roe1 = 0.12 + 0.57 * roe0
        roe2 = 0.12 + 0.57 * roe1
        roe3 = 0.12 + 0.57 * roe2
        roe4 = 0.12 + 0.57 * roe3
        roe5 = 0.12 + 0.57 * roe4
        roe6 = 0.12 + 0.57 * roe5
        roe7 = 0.12 + 0.57 * roe6
        roe8 = 0.12 + 0.57 * roe7
        roe9 = 0.12 + 0.57 * roe8
        roe10 = 0.12 + 0.57 * roe9
        cd1 = roe1 * be0
        cd2 = roe2 * be1
        cd3 = roe3 * be2
        cd4 = roe4 * be3
        cd5 = roe5 * be4
        cd6 = roe6 * be5
        cd7 = roe7 * be6
        cd8 = roe8 * be7
        cd9 = roe9 * be8
        cd10 = roe10 * be9
        dur = 58 / 3 + 1 / FT["MC"] * (
            cd1 / 1.12**1 * (1 - 58 / 3)
            + cd2 / 1.12**2 * (2 - 58 / 3)
            + cd3 / 1.12**3 * (3 - 58 / 3)
            + cd4 / 1.12**4 * (4 - 58 / 3)
            + cd5 / 1.12**5 * (5 - 58 / 3)
            + cd6 / 1.12**6 * (6 - 58 / 3)
            + cd7 / 1.12**7 * (7 - 58 / 3)
            + cd8 / 1.12**8 * (8 - 58 / 3)
            + cd9 / 1.12**9 * (9 - 58 / 3)
            + cd10 / 1.12**10 * (10 - 58 / 3)
        )
        dur = dur * (
            np.nan
            ** (
                (be0 <= 0)
                | (be1 <= 0)
                | (be2 <= 0)
                | (be3 <= 0)
                | (be4 <= 0)
                | (be5 <= 0)
                | (be6 <= 0)
                | (be7 <= 0)
                | (be8 <= 0)
                | (be9 <= 0)
            )
        )
        result = pd.DataFrame({"DurE": dur})
        result = result.groupby("FTID").last()
        return result[Out]


class DiscretionaryAccrual(SignalClass):
    """
    Discretionary Accruals: Dechow, Sloan, and Sweeney (1995)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -30,
                "end": 0,
                "items": [
                    "act",
                    "che",
                    "lct",
                    "dlc",
                    "txp",
                    "dp",
                    "sale",
                    "at",
                    "rect",
                    "ppegt",
                    "INDM3",
                ],
            }
        }
        self.Output = ["DA"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        lta = FT["at"].groupby(by="FTID").shift(1)
        FT["ex1"] = 1 / lta
        FT["ex2"] = (
            FT["sale"] - FT["sale"].groupby("FTID").shift(1) - FT["rect"] + FT["rect"].groupby("FTID").shift(1)
        ) / lta
        FT["ex3"] = FT["ppegt"] / lta
        #  accruals with lagged (not avg. assets)
        dca = FT["act"] - FT["act"].groupby(by="FTID").shift(1)
        dcash = FT["che"] - FT["che"].groupby(by="FTID").shift(1)
        dcl = FT["lct"] - FT["lct"].groupby(by="FTID").shift(1)
        dstd = FT["dlc"] - FT["dlc"].groupby(by="FTID").shift(1)
        dtp = FT["txp"] - FT["txp"].groupby(by="FTID").shift(1)
        FT["eg"] = (
            (dca.fillna(0) - dcash.fillna(0)) - (dcl.fillna(0) - dstd.fillna(0) - dtp.fillna(0)) - FT["dp"]
        ) / lta
        dac = FT[["eg", "ex1", "ex2", "ex3", "INDM3"]].dropna()
        dac = dac.groupby("FTID").last()
        dac.replace([np.inf, -np.inf], np.nan, inplace=True)
        dac.dropna(inplace=True)
        dac["ones"] = 1

        def GroupReg(g):
            """
            takes residuals from the regression
            """
            g["DA"] = sm.OLS(g["eg"], g[["ones", "ex1", "ex2", "ex3"]].values).fit().resid
            g["DA"] = g["DA"].astype(float)
            return g[["DA"]]

        result = (
            dac[["eg", "ex1", "ex2", "ex3", "ones", "INDM3"]].groupby(["INDM3"], group_keys=False).apply(GroupReg)
        )
        return result[Out]


class EarningsFeatures(SignalClass):
    """
    Francis, Lafond, Olsson, and Schipper (2004)
    Earnings Persistence: EPer
    Earnings Predictability: EPred
    Earnings Timeliness: ETime
    Earnings Conservatism: ECon
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -138,
                "end": 0,
                "items": ["epspx", "ib", "MC", "DTID", "FinYearEnd", "ajex"],
            },
            "dt": {"start": -161, "end": 0, "items": ["RI"]},
        }
        self.Output = ["EPer", "EPred", "ETime", "ECon"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dt = FetchedData["dt"]

        # subset to firms that are available in the last one and half year
        RecentObs = FT.loc[FT.index.get_level_values("date") > DM.Date - relativedelta(months=18), "DTID"]
        FT = FT.loc[FT["DTID"].isin(RecentObs)].copy()
        dt = dt.loc[dt.index.get_level_values("DTID").isin(RecentObs)].copy()

        # create returns over 15 months
        dt.reset_index(inplace=True)
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        FT.reset_index(inplace=True)
        # use date that corresponds to FT accounting year end plus 3 months
        FT["date"] = FT["FinYearEnd"] + pd.offsets.DateOffset(months=3)
        FT.set_index("date", inplace=True)
        FT.sort_index(inplace=True)
        FT = pd.merge_asof(FT, dt, on="date", by="DTID")
        dt.rename(columns={"RI": "RI_lag"}, inplace=True)
        dt.index = dt.index + pd.offsets.DateOffset(months=15)
        FT = pd.merge_asof(FT, dt[["DTID", "RI_lag"]], on="date", by="DTID")
        FT["RET15"] = FT["RI"] / FT["RI_lag"] - 1

        # create earnings frame for the regressions
        FT.set_index(["FTID", "date"], inplace=True)
        FT["earn"] = FT["ib"] / FT["MC"]
        FT["RET15negind"] = 1 * (FT["RET15"] < 0)
        FT["RET15neg"] = FT["RET15negind"] * FT["RET15"]
        earn_frame = FT[["earn", "RET15negind", "RET15neg", "RET15"]].copy()
        earn_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        earn_frame.dropna(inplace=True)
        earn_frame["ones"] = 1

        # crate eps frame for the regressions
        FT["eps"] = FT["epspx"] / FT["ajex"]
        FT["l_eps"] = FT["eps"].groupby("FTID").shift(1)
        eps_frame = FT[["eps", "l_eps"]].copy()
        eps_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        eps_frame.dropna(inplace=True)
        eps_frame["ones"] = 1

        eps_frame = eps_frame.loc[eps_frame.groupby("FTID")["ones"].transform("count") >= 10].copy()
        earn_frame = earn_frame.loc[earn_frame.groupby("FTID")["ones"].transform("count") >= 10].copy()

        def GroupReg1(g):
            y = g["eps"].values
            X = g[["ones", "l_eps"]].values
            beta = np.linalg.pinv(X).dot(y)
            resid = y - X.dot(beta)
            mse = (resid**2).sum()
            return pd.Series({"EPer": beta[1], "EPred": mse}, index=["EPer", "EPred"])

        def GroupReg2(g):
            y = g["earn"].values
            X = g[["ones", "RET15negind", "RET15", "RET15neg"]].values
            beta = np.linalg.pinv(X).dot(y)
            if beta[1] != 0:
                econ = (beta[1] + beta[2]) / beta[1]
            else:
                econ = np.nan
            resid = y - X.dot(beta)
            r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
            return pd.Series({"ETime": r2, "ECon": econ}, index=["ETime", "ECon"])

        dt1 = eps_frame.groupby("FTID", group_keys=True).apply(GroupReg1)
        dt2 = earn_frame.groupby("FTID", group_keys=True).apply(GroupReg2)
        result = dt1.join(dt2, how="outer")
        return result[Out]


class IntangibleReturn(SignalClass):
    """
    Intangible Return: Daniel and Titman (2006)
    """

    def __init__(self):
        SignalClass.__init__(self)
        self.Input = {
            "FT": {
                "start": -78,
                "end": 0,
                "items": [
                    "pstkrv",
                    "pstkl",
                    "pstk",
                    "seq",
                    "ceq",
                    "lt",
                    "at",
                    "MC",
                    "DTID",
                    "FinYearEnd",
                ],
            },
            "dt": {"start": -101, "end": 0, "items": ["RI", "PRC"]},
        }
        self.Output = ["IntanRet"]

    def CreateSignal(self, DM, Out=["default"]):
        Out = self.CheckOutput(Out)
        FetchedData = DM.fetch(self.Input)
        FT = FetchedData["FT"]
        dt = FetchedData["dt"]

        ps = FT["pstkrv"].copy()
        ps.loc[pd.isnull(ps)] = FT["pstkl"]
        ps.loc[pd.isnull(ps)] = FT["pstk"]
        be = (FT["seq"] - ps.fillna(0)).copy()
        be.loc[pd.isnull(be)] = FT["ceq"] + FT["pstk"] - ps
        be.loc[pd.isnull(be)] = FT["at"] - FT["lt"] - ps.fillna(0)
        FT["be"] = be
        FT["bm"] = FT["be"] / FT["MC"]
        FT["bm"] = FT["bm"].replace([np.inf, -np.inf], np.nan)
        FT = FT[["bm", "be", "FinYearEnd", "DTID"]].dropna()
        FT.reset_index(inplace=True)
        FT = FT.loc[FT.groupby("FTID")["be"].transform("count") >= 6].copy()
        dt.reset_index(inplace=True)
        dt = dt.loc[dt["DTID"].isin(FT["DTID"])].copy()

        # create returns over 12 months
        dt.set_index("date", inplace=True)
        dt.sort_index(inplace=True)
        # use date that corresponds to FT accounting year end
        FT["date"] = FT["FinYearEnd"]
        FT.set_index("date", inplace=True)
        FT.sort_index(inplace=True)
        FT = pd.merge_asof(FT, dt, on="date", by="DTID")
        dt.rename(columns={"RI": "RI_lag", "PRC": "PRC_lag"}, inplace=True)
        dt.index = dt.index + pd.offsets.DateOffset(months=12)
        FT = pd.merge_asof(FT, dt, on="date", by="DTID")
        FT.set_index(["FTID", "date"], inplace=True)

        FT["RET12"] = np.log(FT["RI"] / FT["RI_lag"])
        FT["r5y"] = (
            FT["RET12"]
            + FT["RET12"].groupby("FTID").shift(1)
            + FT["RET12"].groupby("FTID").shift(2)
            + FT["RET12"].groupby("FTID").shift(3)
            + FT["RET12"].groupby("FTID").shift(4)
        )
        FT["bm5y"] = np.log(FT["bm"].groupby("FTID").shift(5))
        bmret = FT["RET12"] - np.log(FT["PRC"] / FT["PRC_lag"])
        FT["rb5y"] = (
            FT["be"] / FT["be"].groupby("FTID").shift(5)
            + bmret.groupby("FTID").shift(1)
            + bmret.groupby("FTID").shift(2)
            + bmret.groupby("FTID").shift(3)
            + bmret.groupby("FTID").shift(4)
            + bmret.groupby("FTID").shift(5)
        )
        ir = FT[["r5y", "bm5y", "rb5y"]].copy()
        ir.replace([np.inf, -np.inf], np.nan, inplace=True)
        ir.dropna(inplace=True)
        ir["ones"] = 1
        ir = ir.groupby("FTID").last()
        # get residual from the regression
        ir["IntanRet"] = sm.OLS(ir["r5y"], ir[["ones", "bm5y", "rb5y"]]).fit().resid
        result = ir[["IntanRet"]]
        return result[Out]
