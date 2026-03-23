import os
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class DataManager:
    """
    Main workhorse to fetch data to be used in the analysis
    """

    def ProcessInputs(self, Inputs):
        """
        function that checks inputs and respahes them for internal purpouses
        """

        # if MC in FT then add it to dt
        if (Inputs != None) and ("FT" in Inputs.keys()) and ("MC" in Inputs["FT"]["items"]):
            if "dt" in Inputs.keys():
                if not "MC" in Inputs["dt"]["items"]:
                    Inputs["dt"]["items"] = Inputs["dt"]["items"] + ["MC"]
            else:
                Inputs["dt"] = {"start": -1, "end": 0, "items": ["MC"]}
        return Inputs

    def InitDBs(self):
        """
        initialize individual datasets
        """

        # for market data
        if (self.Inputs == None) or ("dt" in self.Inputs.keys()):
            if self.Inputs == None:
                items = None
                # load 20 years of data as default
                start = (self.Date + relativedelta(months=-241)).strftime("%Y-%m-%d")
                end = self.Date.strftime("%Y-%m-%d")
            else:
                items = self.Inputs["dt"]["items"] + ["DTID", "date", "region"]
                start = (self.Date + relativedelta(months=self.Inputs["dt"]["start"])).strftime("%Y-%m-%d")
                end = (self.Date + relativedelta(months=self.Inputs["dt"]["end"])).strftime("%Y-%m-%d")

            dt = pd.read_parquet(os.path.join(self.sPath, "ProcessedData", self.dtName + ".gzip"), columns=items)
            if self.region != None:
                dt = dt.loc[(dt.region == self.region) & (dt.date < end) & (dt.date >= start)].copy()
            else:
                dt = dt.loc[(dt.date < end) & (dt.date >= start)].copy()
            dt.set_index(["DTID", "date"], inplace=True)
            self.dt = dt
        else:
            self.dt = pd.DataFrame()

        # for fundamental data
        if (self.Inputs == None) or ("FT" in self.Inputs.keys()):
            if self.Inputs == None:
                items = None
                # load 21.5 years of data as default
                start = (self.Date + relativedelta(months=-258)).strftime("%Y-%m-%d")
                end = self.Date.strftime("%Y-%m-%d")
            else:
                items = [item for item in self.Inputs["FT"]["items"] if item not in ["DTID", "FinYearEnd"]]
                items += ["DTID", "FinYearEnd", "FTID", "date", "region"]
                start = (self.Date + relativedelta(months=self.Inputs["FT"]["start"])).strftime("%Y-%m-%d")
                end = (self.Date + relativedelta(months=self.Inputs["FT"]["end"])).strftime("%Y-%m-%d")

            FT = pd.read_parquet(os.path.join(self.sPath, "ProcessedData", self.FTName + ".gzip"), columns=items)
            if self.region != None:
                FT = FT.loc[(FT.region == self.region) & (FT.date < end) & (FT.date >= start)].copy()
            else:
                FT = FT.loc[(FT.date < end) & (FT.date >= start)].copy()
            FT.set_index(["FTID", "date"], inplace=True)
            self.FT = FT
        else:
            self.FT = pd.DataFrame()

        # for IBES summary
        if (self.Inputs == None) or ("IBESsum" in self.Inputs.keys()):
            # do not load anything as a default
            if self.Inputs == None:
                self.IBESsum = pd.DataFrame()
            else:
                items = self.Inputs["IBESsum"]["items"] + ["DTID", "PERMNO", "date", "region"]
                start = (self.Date + relativedelta(months=self.Inputs["IBESsum"]["start"])).strftime("%Y-%m-%d")
                end = (self.Date + relativedelta(months=self.Inputs["IBESsum"]["end"])).strftime("%Y-%m-%d")
                IBES = pd.read_parquet(
                    os.path.join(self.sPath, "ProcessedData", self.IBESsumName + ".gzip"),
                    columns=items,
                )
                if self.region != None:
                    IBES = IBES.loc[(IBES.region == self.region) & (IBES.date < end) & (IBES.date >= start)].copy()
                else:
                    IBES = IBES.loc[(IBES.date < end) & (IBES.date >= start)].copy()
                self.IBESsum = self.ProcessIBES(IBES)
        else:
            self.IBESsum = pd.DataFrame()

        # for IBES detailed
        if (self.Inputs == None) or ("IBESdet" in self.Inputs.keys()):
            # do not load anything as a default
            if self.Inputs == None:
                self.IBESdet = pd.DataFrame()
            else:
                items = self.Inputs["IBESdet"]["items"] + ["DTID", "PERMNO", "date", "region"]
                start = (self.Date + relativedelta(months=self.Inputs["IBESdet"]["start"])).strftime("%Y-%m-%d")
                end = (self.Date + relativedelta(months=self.Inputs["IBESdet"]["end"])).strftime("%Y-%m-%d")
                IBES = pd.read_parquet(
                    os.path.join(self.sPath, "ProcessedData", self.IBESdetName + ".gzip"),
                    columns=items,
                )
                if self.region != None:
                    IBES = IBES.loc[(IBES.region == self.region) & (IBES.date < end) & (IBES.date >= start)].copy()
                else:
                    IBES = IBES.loc[(IBES.date < end) & (IBES.date >= start)].copy()
                self.IBESdet = self.ProcessIBES(IBES)
        else:
            self.IBESdet = pd.DataFrame()

    def FetchDBs(self, DBs):
        """
        method to fetch DBs from in memory instance of DataManager so that SQL is not run each time
        """

        DBs.Date = self.Date
        FetchedData = DBs.fetch(self.Inputs)
        if "FT" in FetchedData.keys():
            self.FT = FetchedData["FT"]
        if "dt" in FetchedData.keys():
            self.dt = FetchedData["dt"]
        if "IBESsum" in FetchedData.keys():
            self.IBESsum = FetchedData["IBESsum"]
        if "IBESdet" in FetchedData.keys():
            self.IBESdet = FetchedData["IBESdet"]

    def ProcessFT(self):
        """
        method that adds the latest capitalization to the last observation of FT and drops duplicated financial
        year-firm observations
        """

        FT = self.FT
        FT.reset_index(inplace=True)
        # add market cap for the last observation if requested
        if (self.Inputs == None) or ("MC" in self.Inputs["FT"]["items"]):
            MC = self.dt[["MC"]].groupby("DTID").last()
            MC.rename(columns={"MC": "MClast"}, inplace=True)
            FT = FT.merge(MC, how="left", on="DTID")
            FT.loc[
                (FT["date"] > self.Date + relativedelta(months=-12)) & (FT["MClast"].isnull() == False), "MC"
            ] = FT["MClast"]
            del FT["MClast"], MC

        # process so that there is only one observation per financial year
        FT["year"] = -np.ceil((self.Date - FT["FinYearEnd"]) / np.timedelta64(365, "D"))
        FT = FT.drop_duplicates(subset=["FTID", "year"], keep="last")
        FT.set_index(["FTID", "date"], inplace=True)
        del FT["year"]
        self.FT = FT

    def ProcessIBES(self, IBES):
        """
        method that selects appropriate DTID for IBES depending on data source
        """

        # use PERMNO for CRSP
        if self.source == "WRDS":
            IBES["DTID"] = IBES["PERMNO"]
        del IBES["PERMNO"]
        IBES.dropna(subset=["DTID", "date"], inplace=True)
        IBES.set_index(["DTID", "date"], inplace=True)
        return IBES

    def fetch(self, Input):
        """
        callable function to provide required inputs
        """

        Output = {}
        if "dt" in Input.keys():
            start = self.Date + relativedelta(months=Input["dt"]["start"])
            end = self.Date + relativedelta(months=Input["dt"]["end"])
            dt = (
                self.dt.loc[
                    (self.dt.index.get_level_values("date") >= start)
                    & (self.dt.index.get_level_values("date") < end),
                    Input["dt"]["items"],
                ]
            ).copy()
            Output["dt"] = dt

        if "FT" in Input.keys():
            start = self.Date + relativedelta(months=Input["FT"]["start"])
            end = self.Date + relativedelta(months=Input["FT"]["end"])
            FT = (
                self.FT.loc[
                    (self.FT.index.get_level_values("date") >= start)
                    & (self.FT.index.get_level_values("date") < end),
                    Input["FT"]["items"],
                ]
            ).copy()
            Output["FT"] = FT

        if "IBESsum" in Input.keys():
            start = self.Date + relativedelta(months=Input["IBESsum"]["start"])
            end = self.Date + relativedelta(months=Input["IBESsum"]["end"])
            IBESsum = (
                self.IBESsum.loc[
                    (self.IBESsum.index.get_level_values("date") >= start)
                    & (self.IBESsum.index.get_level_values("date") < end),
                    Input["IBESsum"]["items"],
                ]
            ).copy()
            Output["IBESsum"] = IBESsum

        if "IBESdet" in Input.keys():
            start = self.Date + relativedelta(months=Input["IBESdet"]["start"])
            end = self.Date + relativedelta(months=Input["IBESdet"]["end"])
            IBESdet = (
                self.IBESdet.loc[
                    (self.IBESdet.index.get_level_values("date") >= start)
                    & (self.IBESdet.index.get_level_values("date") < end),
                    Input["IBESdet"]["items"],
                ]
            ).copy()
            Output["IBESdet"] = IBESdet

        return Output

    def __init__(self, sPath, Date, source="DST", region=None, Inputs=None, DBs=None):
        self.sPath = sPath
        if isinstance(Date, datetime):
            self.Date = Date
        elif type(Date) == str:
            self.Date = datetime.strptime(Date, "%Y-%m-%d")
        else:
            raise Exception("Please provide date in either string '%Y-%m-%d' or datetime.datetime object.")
        self.source = source
        if source == "DST":
            self.dtName = "DST"
            self.FTName = "WorldScope"
        elif source == "WRDS":
            self.dtName = "CRSP"
            self.FTName = "Compustat"
        else:
            raise Exception("Currently only WRDS and DST is supported as a source of data.")
        self.IBESsumName = "IBESsum"
        self.IBESdetName = "IBESdet"
        self.region = region
        self.Inputs = self.ProcessInputs(Inputs)
        if DBs == None:
            self.InitDBs()
        elif isinstance(DBs, DataManager):
            self.FetchDBs(DBs)
        else:
            raise Exception("DBs must be either None or an instance of DataManager.")
