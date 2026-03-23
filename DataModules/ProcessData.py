import os
import re
from datetime import datetime

import fastparquet
import sqlite3
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def ProcessData(sPath, RunDB=["CRSP", "DST", "WorldScope", "Compustat", "IBESsum", "IBESdet"]):
    """
    defines function to process raw data and save it in a format retrievable by python
    arguments:
        sPath(str): points to the directory
        RunDB(list): list of string describing which databases should be run
    """

    # create folder for the results
    os.makedirs(os.path.join(sPath, "ProcessedData"), exist_ok=True)

    if "CRSP" in RunDB:
        print("Processing daily CRSP")
        CRSP = ProcessCRSPData(sPath)
        CRSP.to_parquet(os.path.join(sPath, "ProcessedData", "CRSP.gzip"), compression="gzip")
        del CRSP
        print("CRSP done")

    if "DST" in RunDB:
        print("Processing daily Datastream")
        DST = ProcessDSTData(sPath)
        DST.to_parquet(os.path.join(sPath, "ProcessedData", "DST.gzip"), compression="gzip")
        del DST
        print("DST done")

    if "WorldScope" in RunDB:
        print("Processing WorldScope")
        WorldScope = ProcessWorldScopeData(sPath)
        WorldScope = FTAddMC(WorldScope, sPath, DTtype="DST")
        WorldScope.to_parquet(os.path.join(sPath, "ProcessedData", "WorldScope.gzip"), compression="gzip")
        del WorldScope
        print("WorldScope done")

    if "Compustat" in RunDB:
        print("Processing Compustat")
        Compustat = ProcessCompustatData(sPath)
        Compustat = FTAddMC(Compustat, sPath, DTtype="CRSP")
        Compustat.to_parquet(os.path.join(sPath, "ProcessedData", "Compustat.gzip"), compression="gzip")
        del Compustat
        print("Compustat done")

    if "CRSPm" in RunDB:
        print("Processing monthly CRSP")
        CRSPm = ProcessCRSPMonthlyData(sPath)
        CRSPm.to_parquet(os.path.join(sPath, "ProcessedData", "CRSPm.gzip"), compression="gzip")
        del CRSPm
        print("CRSP monthly done")

    if "IBESsum" in RunDB:
        print("Processing IBES summary")
        ibes = ProcessIBESsum(sPath)
        ibes.to_parquet(os.path.join(sPath, "ProcessedData", "IBESsum.gzip"), compression="gzip")
        del ibes
        print("IBES summary done")

    if "IBESdet" in RunDB:
        print("Processing IBES detailed")
        ibes = ProcessIBESdet(sPath)
        ibes.to_parquet(os.path.join(sPath, "ProcessedData", "IBESdet.gzip"), compression="gzip")
        del ibes
        print("IBES detailed done")

    print("All databases processed")


def ProcessInflation(sPath):
    """loads US inflation data"""
    infl = pd.read_csv(os.path.join(sPath, "Inputs", "CPIAUCNS.csv"), header=0, low_memory=False)
    infl.rename(columns={"CPIAUCNS": "CPI", "DATE": "date"}, inplace=True)
    infl["date"] = pd.to_datetime(infl["date"].astype(str), format="%d/%m/%Y")
    infl["CPI"] = infl["CPI"] / 77.8  # convert to 1980 level
    infl.set_index("date", inplace=True)
    infl.sort_index(inplace=True)
    return infl


def ProcessCompustatData(sPath):
    """loads annual Compustat data
    note that the fundamental database has been merged in a way so that there is just one PERMNO per company (GVKEY)
    the merge back to trade data is then done through PERMCO which is the same for all shares for a given company
    difference between using GVKEY and PERMCO is 150 observations so it should not matter too much"""

    ## get raw Compustat data
    FT = pd.read_csv(
        os.path.join(sPath, "Inputs", "comp.funda.csv"),
        header=0,
        low_memory=False,
    )

    ## add CRSP ID link
    link = pd.read_csv(os.path.join(sPath, "Inputs", "crsp.ccmxpf_linktable.csv"), header=0, low_memory=False)
    del link["usedflag"]
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"])
    link["datadate"] = link["linkdt"]
    link = link.loc[link["lpermno"].notnull()].copy()
    FT["datadate"] = pd.to_datetime(FT["datadate"])
    FT.sort_values(["datadate", "gvkey"], inplace=True)
    link.sort_values(["datadate", "gvkey"], inplace=True)
    link_cols = [col for col in link.columns if col not in ["gvkey", "datadate"]]
    # link dates can be overlapping so need to do it in sequence one mapping at a time
    link.rename({col: col + "_" for col in link_cols}, axis=1, inplace=True)
    link["order"] = link.groupby("gvkey").cumcount()
    FT["lpermno"] = np.nan
    for order in link["order"].unique():
        FT = pd.merge_asof(FT, link.loc[link["order"] == order], on="datadate", by="gvkey")
        FT["add"] = False
        FT.loc[
            (
                ((FT["datadate"] <= FT["linkenddt_"]) | FT["linkenddt_"].isnull())
                & FT["lpermno_"].notnull()
                & FT["lpermno"].isnull()
            ),
            "add",
        ] = True
        for col in link_cols:
            FT.loc[FT["add"], col] = FT[col + "_"]
        FT.drop([col + "_" for col in link_cols] + ["order", "add"], axis=1, inplace=True)

    ## filter entries
    FT = FT.loc[FT["datafmt"] == "STD"].copy()  # SUMM_STD has restated data e.g. post merger
    FT = FT.loc[FT["indfmt"] == "INDL"].copy()  # there is a special format - expclude FS for financial servises
    FT = FT.loc[FT["consol"] == "C"].copy()  # level of consolidation - C for consolidated

    ## add SIC
    comp = pd.read_csv(os.path.join(sPath, "Inputs", "crsp.comphist.csv"), header=0, low_memory=False)
    comp["hchgdt"] = pd.to_datetime(comp["hchgdt"])
    comp["hchgenddt"] = pd.to_datetime(comp["hchgenddt"])
    comp["datadate"] = comp["hchgdt"]
    FT.sort_values(["datadate", "gvkey"], inplace=True)
    comp.sort_values(["datadate", "gvkey"], inplace=True)
    FT = pd.merge_asof(FT, comp, on="datadate", by="gvkey")
    FT.drop(["hchgdt", "hchgenddt"], axis=1, inplace=True)
    FT.rename({"hsic": "sic"}, axis=1, inplace=True)

    ## fill missing SIC with the latest data
    comp = pd.read_csv(os.path.join(sPath, "Inputs", "crsp.comphead.csv"), header=0, low_memory=False)
    comp = comp[["gvkey", "sic"]].rename({"sic": "sic_add"}, axis=1)
    FT = FT.merge(comp, on="gvkey", how="left")
    FT.loc[FT["sic"].isnull(), "sic"] = FT["sic_add"]
    del FT["sic_add"]

    # rename identifiers to make them in line with other databases
    FT.rename(columns={"lpermno": "DTID", "lpermco": "PERMCO", "datadate": "date", "gvkey": "FTID"}, inplace=True)
    FT.sort_values(["FTID", "date"], inplace=True)

    # eliminate all companies without PERMCO, i.e. without possibility to make a link from FT to CRSP
    # this is a bit shady as some anomalies can be created even without CRSP history but makes life easier
    FT = FT[FT["PERMCO"].notnull()].copy()
    # eliminate observation if there is a missing financial year
    FT.dropna(subset=["date"], inplace=True)
    # financial year end date
    FT["FinYearEnd"] = pd.to_datetime(FT["date"].astype(str))
    FT["date"] = FT["FinYearEnd"].transform(lambda x: x + relativedelta(months=6))
    # throw out duplicities in indexes
    FT = FT.drop_duplicates(subset=["FTID", "date"], keep="last")

    # add the same industry classification as in DST
    xl = pd.ExcelFile(os.path.join(sPath, "Inputs", "AnomaliesMeta.xlsx"))
    xl = xl.parse("sic")
    xl = xl[["sic", "INDM", "INDM3"]]
    FT = FT.merge(xl, on=["sic"], how="left")

    # add region
    FT["region"] = "USA"

    # add date for the first observation
    FT["FirstObs"] = FT.groupby("FTID")["date"].transform(lambda x: min(x))

    # Add inflation
    infl = ProcessInflation(sPath)
    FT.set_index("date", inplace=True)
    FT.sort_index(inplace=True)
    FT = pd.merge_asof(FT, infl, on=["date"])

    # final changes
    FT["DTID"] = FT["DTID"].astype(int).astype(str)
    return FT


def ProcessWorldScopeData(sPath):
    """loads WorldScope data"""

    # load fundamentals
    FT = pd.read_csv(os.path.join(sPath, "Inputs", "DSTfundamental.csv"), header=0, low_memory=False, sep="\t")
    FT.rename(
        columns={
            name: re.search("WC[0-9]*", name).group(0) for name in FT.columns if bool(re.search("WC[0-9]*", name))
        },
        inplace=True,
    )
    FT.rename(columns={"dscd": "DTID", "WC05350": "date"}, inplace=True)
    FT["DTID"] = FT["DTID"].astype(str)

    # add information stored in a static file
    static = pd.read_csv(
        os.path.join(sPath, "Inputs", "DSTstatic.csv"),
        header=0,
        usecols=["DSCD", "INDM3", "INDM", "WC06105", "region"],
    )
    static.rename(columns={"DSCD": "DTID", "WC06105": "FTID"}, inplace=True)
    FT = FT.merge(static, on="DTID")
    FT = FT[FT.region.isin(["Asia Pacific", "Europe", "North America", "Japan"])]

    # throw out observations with missing date or FTID
    FT = FT[FT["FTID"].notnull() & FT["date"].notnull()]

    # rename variable to be in line with Compustat
    FT["ivao"] = FT["WC02258"] + FT["WC02250"]
    FT["lco"] = FT["WC03066"] + FT["WC03054"] + FT["WC03063"] + FT["WC03061"]
    FT["seq"] = FT["WC03501"] + FT["WC03451"]
    FT["aco"] = FT["WC02149"] + FT["WC02140"]
    FT["lo"] = FT["WC03273"] + FT["WC03262"]
    FT["oibdp"] = FT["WC01151"] + FT["WC01250"]
    FT["pstkrv"] = FT["WC03451"]
    FT["pstkl"] = FT["WC03451"]
    FT["revt"] = FT["WC01001"]
    FT["ni"] = FT["WC01551"]
    FT.rename(
        columns={
            "WC02001": "che",
            "WC02008": "ivst",
            "WC02051": "rect",
            "WC02101": "invt",
            "WC02140": "xpp",
            "WC02201": "act",
            "WC02501": "ppent",
            "WC02301": "ppegt",
            "WC18376": "fatb",
            "WC18381": "fatl",
            "WC02256": "ivaeq",
            "WC02649": "intan",
            "WC02999": "at",
            "WC03051": "dlc",
            "WC03040": "ap",
            "WC03063": "txp",
            "WC03101": "lct",
            "WC03251": "dltt",
            "WC03262": "drlt",
            "WC03351": "lt",
            "WC03426": "mib",
            "WC03451": "pstk",
            "WC03495": "re",
            "WC03501": "ceq",
            "WC01001": "sale",
            "WC01051": "cogs",
            "WC01101": "xsga",
            "WC01201": "xrd",
            "WC01151": "dp",
            "WC01250": "oiadp",
            "WC01401": "pi",
            "WC01451": "txt",
            "WC01551": "ib",
            "WC01251": "xint",
            "WC04860": "oancf",
            "WC04601": "capx",
            "WC04751": "prstkc",
            "WC04251": "sstk",
            "WC04551": "dv",
            "WC05376": "dvc",
            "WC04401": "dltis",
            "WC04701": "dltr",
            "WC04821": "dlcch",
            "WC04890": "fincf",
            "WC05476": "bkvlps",
            "WC05210": "epspx",
            "WC05230": "epspi",
            "WC07011": "emp",
        },
        inplace=True,
    )
    FT["ivncf"] = -FT["WC04870"]
    FT["dvpa"] = 0  # set these to 0 as they are missing in DST
    FT["tstkp"] = 0
    FT["drc"] = 0
    FT["xacc"] = 0
    FT["gdwl"] = 0  # is already in INTAN in DST

    # divide all the variables by 1000 to transform them into millions of US dollars
    RescaleField = [
        "che",
        "ivst",
        "ivao",
        "rect",
        "invt",
        "xpp",
        "act",
        "ppent",
        "ppegt",
        "ivaeq",
        "intan",
        "at",
        "dlc",
        "ap",
        "txp",
        "lct",
        "dltt",
        "drlt",
        "lt",
        "mib",
        "pstk",
        "re",
        "ceq",
        "sale",
        "cogs",
        "xsga",
        "xrd",
        "dp",
        "oiadp",
        "pi",
        "txt",
        "ib",
        "xint",
        "oancf",
        "capx",
        "ivncf",
        "prstkc",
        "sstk",
        "dv",
        "dltis",
        "dlcch",
        "dvc",
        "dltr",
        "fincf",
        "lo",
        "lco",
        "seq",
        "aco",
        "fatb",
        "fatl",
        "oibdp",
        "pstkrv",
        "pstkl",
        "ni",
        "revt",
        "emp",
    ]
    FT[RescaleField] = FT[RescaleField].astype(float) / 1000

    # change date format
    FT["FinYearEnd"] = pd.to_datetime(FT["date"].astype(str), format="%Y-%m-%d")
    FT["date"] = FT["FinYearEnd"].transform(lambda x: x + relativedelta(months=6))

    # throw out duplicities when firms change accounting year and when there are multiple DTID per one FTID
    FT = FT.drop_duplicates(subset=["FTID", "date"], keep="last")

    # add date for the first observation
    FT["FirstObs"] = FT.groupby("FTID")["date"].transform(lambda x: min(x))

    # Add inflation
    infl = ProcessInflation(sPath)
    FT.set_index("date", inplace=True)
    FT.sort_index(inplace=True)
    FT = pd.merge_asof(FT, infl, on=["date"])

    # have to add ajex in later step - corresponds to WC05575 which is useless due to low coverage
    return FT


def ProcessDSTData(sPath):
    """loads Datastream data"""

    NameDict = {
        "dscd": "DTID",
        "date_d": "date",
        "DPL#((X(RI)~U$),4)": "RI",
        "DPL#((X(UP)~U$),4)": "PRC",
        "DPL#((X(UPA)~U$),4)": "ASK",
        "DPL#((X(UPB)~U$),4)": "BID",
        "DPL#((X(UPH)~U$),4)": "H",
        "DPL#((X(UPL)~U$),4)": "L",
        "DPL#(X(UVO),4)": "VOL",
        "DPL#(X(AF),4)": "AF",
        "DPL#(X(CAI),4)": "CAI",
        "DPL#(X(NOSH),4)": "SHROUT",
        "DPL#(X(RI),4)": "RI_OC",
        "DPL#((X(UP#S)~U$),4)": "PRCunpadded",
    }

    static = pd.read_csv(
        os.path.join(sPath, "Inputs", "DSTstatic.csv"),
        header=0,
        usecols=["DSCD", "region", "GEOGN", "WC06105"],
        dtype={"DSCD": str, "region": str, "GEOGN": str, "WC06105": str},
    )
    static.rename({"DSCD": "DTID", "WC06105": "FTID"}, inplace=True, axis=1)

    dtfull = pd.DataFrame()
    regions = ["North America", "Europe", "Japan", "Asia Pacific"]
    for reg in regions:
        # load raw data
        DTIDs = static.loc[static.region == reg, "DTID"]
        iter_csv = pd.read_csv(
            os.path.join(sPath, "Inputs", "DSTd.csv"),
            iterator=True,
            chunksize=1000000,
            low_memory=False,
            sep="\t",
            dtype={"dscd": str},
        )
        dt = pd.concat([chunk[chunk["dscd"].isin(DTIDs)] for chunk in iter_csv])
        # load second batch of raw data
        iter_csv = pd.read_csv(
            os.path.join(sPath, "Inputs", "DSTd2.csv"),
            iterator=True,
            chunksize=1000000,
            low_memory=False,
            sep="\t",
            dtype={"dscd": str},
        )
        dt2 = pd.concat([chunk[chunk["dscd"].isin(DTIDs)] for chunk in iter_csv])
        # merge together
        dt = dt.merge(dt2, on=["dscd", "date_d"], how="outer")

        # change names
        dt.rename(NameDict, inplace=True, axis=1)

        # change types
        dt["date"] = pd.to_datetime(dt["date"].astype(str), format="%Y-%m-%d")
        dt["DTID"] = dt["DTID"].astype(str)
        floatcols = ["RI", "PRC", "ASK", "BID", "H", "L", "VOL", "AF", "CAI", "SHROUT", "RI_OC", "PRCunpadded"]
        for colname in floatcols:
            dt[colname] = pd.to_numeric(dt[colname], errors="coerce")

        # add information stored in a static file
        # note that securities that are not in static are dropped
        dt = dt.merge(static, on="DTID")

        ### clean up the data
        # drop observations with missing return index
        dt = dt.loc[dt.RI.notnull()].copy()
        # 0 in volume is due to rounding down of traded shares
        dt.VOL.replace(0, 0.025, inplace=True)
        # if volume is missing and it was available in the past then fill it with 0
        dt["VOLmissing"] = dt.VOL.notnull() * 1
        dt["VOLprev"] = dt.groupby("DTID").VOLmissing.cummax()
        dt.loc[dt.VOL.isnull() & (dt.VOLprev == 1), "VOL"] = 0
        del dt["VOLprev"], dt["VOLmissing"]
        # replace 0 in PRC to slighly positive number
        dt.PRC.replace(0, 0.0001, inplace=True)
        # replace 0 in RI with 0.0001 - this can ocur due to rounding and 4 decimal places precision
        dt.loc[dt.RI < 0.0001, "RI"] = 0.0001

        ## discard holidays
        # load time series file with holidays
        holidays = pd.read_csv(os.path.join(sPath, "Inputs", "DST_holidays_ts.csv"), sep="\t")
        # load mapping of holiday lists
        holidaysMap = pd.read_csv(os.path.join(sPath, "Inputs", "DST_holidays_map.csv"), sep="\t")
        holidaysMap.dropna(inplace=True)
        holidays.rename(columns={row[1]["MNEM"]: row[1]["GEOGN"] for row in holidaysMap.iterrows()}, inplace=True)
        holidays.rename(columns={"dscd": "date"}, inplace=True)
        holidays["date"] = pd.to_datetime(holidays["date"].astype(str), format="%Y-%m-%d")
        holidays = pd.melt(holidays, id_vars=["date"], var_name="GEOGN", value_name="hol")
        # keep all the observations at the start where the holiday distinction is missing
        holidays.fillna(1, inplace=True)
        holidays.loc[holidays.groupby("GEOGN").hol.cummin() == 1, "hol"] = 0
        dt = dt.merge(holidays, on=["date", "GEOGN"], how="left")
        dt = dt.loc[dt.hol != 1].copy()
        del dt["GEOGN"], dt["hol"]

        ## throw out the latest observations with no trading
        dt.set_index(["DTID", "date"], inplace=True)
        dt.sort_index(inplace=True)
        # have to use unpadded PRC to discard since exchange rate fluctuations change PRC in USD outside US
        # dt['PaddedPRC'] = (dt.PRC == dt.groupby('DTID').PRC.transform('last')) * 1
        dt["PaddedPRC"] = dt.PRCunpadded.isnull() * 1
        dt["PaddedPRC"] = dt.PaddedPRC.iloc[::-1].groupby("DTID").cummin().iloc[::-1]
        # the last value is wrongly labled as padded if discarded based on PRC
        # dt.loc[dt['PaddedPRC'] != dt.groupby('DTID').PaddedPRC.shift(1), 'PaddedPRC'] = 0
        # discard values only if missing for more than one month
        dt["PaddedPRC"] = dt["PaddedPRC"] * (dt["PaddedPRC"].groupby("DTID").transform("sum") > 20)
        dt = dt.loc[dt["PaddedPRC"] == 0].copy()
        del dt["PaddedPRC"], dt["PRCunpadded"]

        ## discard probable errors in the data
        # discard prices that are larger than 100k dollars - checked it and it is all bugs ~ 10 tickers
        # except for '982325' Berkshire Hathaway A class
        dt.loc[(dt.PRC > 100000) & (dt.index.get_level_values("DTID") != "982325"), ["PRC", "H", "L"]] = np.nan

        # fix cases when AF happens but there are no trades so the price stays padded
        dt["l.RI"] = dt.groupby("DTID").RI.shift(1)
        dt["f.RI"] = dt.groupby("DTID").RI.shift(-1)
        dt["l.AF"] = dt.groupby("DTID").AF.shift(1)
        dt["f.AF"] = dt.groupby("DTID").AF.shift(-1)
        dt["PaddedRI"] = False
        dt.loc[
            (
                ((dt["RI"] / dt["l.RI"] > 1.5) & (dt["f.AF"] / dt["l.AF"] > 1.5))
                | ((dt["RI"] / dt["l.RI"] < 0.5) & (dt["f.AF"] / dt["l.AF"] < 0.5))
            )
            & (dt["VOL"] == 0)  # the volume has to be zero on that day
            & (  # and it has to be non-negative in the next 5 days
                (dt.groupby("DTID").VOL.shift(-1) != 0)
                | (dt.groupby("DTID").VOL.shift(-2) != 0)
                | (dt.groupby("DTID").VOL.shift(-3) != 0)
                | (dt.groupby("DTID").VOL.shift(-4) != 0)
                | (dt.groupby("DTID").VOL.shift(-5) != 0)
            ),
            "PaddedRI",
        ] = True
        # volume is zero and the previous was padded then the next observation is also padded
        if dt.PaddedRI.sum() > 0:
            for i in range(5):
                dt.loc[(dt["VOL"] == 0) & dt["PaddedRI"].groupby("DTID").shift(1), "PaddedRI"] = True
                dt.loc[dt["PaddedRI"], "RI"] = dt["RI"].groupby("DTID").shift(1)
                dt.loc[dt["PaddedRI"], "RI_OC"] = dt["RI_OC"].groupby("DTID").shift(1)
                dt.loc[dt["PaddedRI"], "PRC"] = np.nan
        del dt["PaddedRI"]

        # fix cases when there is a reversal the next day - usually just some error or strange quotes
        dt["l.RI_OC"] = dt.groupby("DTID").RI_OC.shift(1)
        dt["f.RI_OC"] = dt.groupby("DTID").RI_OC.shift(-1)
        if dt.loc[(dt["RI"] / dt["l.RI"] > 2) & (dt["f.RI"] / dt["l.RI"] < 1.1)].shape[0] > 0:
            dt.loc[(dt["RI"] / dt["l.RI"] > 2) & (dt["f.RI"] / dt["l.RI"] < 1.1), "RI_OC"] = (
                dt["f.RI_OC"] + dt["l.RI_OC"]
            ) / 2
            dt.loc[(dt["RI"] / dt["l.RI"] > 2) & (dt["f.RI"] / dt["l.RI"] < 1.1), "RI"] = (
                dt["f.RI"] + dt["l.RI"]
            ) / 2
        if dt.loc[(dt["RI"] / dt["l.RI"] < 0.5) & (dt["f.RI"] / dt["l.RI"] > 0.9)].shape[0] > 0:
            dt.loc[(dt["RI"] / dt["l.RI"] < 0.5) & (dt["f.RI"] / dt["l.RI"] > 0.9), "RI_OC"] = (
                dt["f.RI_OC"] + dt["l.RI_OC"]
            ) / 2
            dt.loc[(dt["RI"] / dt["l.RI"] < 0.5) & (dt["f.RI"] / dt["l.RI"] > 0.9), "RI"] = (
                dt["f.RI"] + dt["l.RI"]
            ) / 2

        # set RI to missing if daily return is larger than 500%
        #   most of these are just errors that screw up analysis when daily returns are used
        dt["l.RI"] = dt.groupby("DTID").RI.shift(1)
        dt.loc[dt.RI / dt["l.RI"] > 6, ["RI", "RI_OC"]] = np.nan

        # create daily returns
        dt["r"] = dt["RI"] / dt["l.RI"] - 1

        # discard temporary variables
        del dt["l.RI"], dt["f.RI"], dt["l.RI_OC"], dt["f.RI_OC"], dt["l.AF"], dt["f.AF"]
        dt.reset_index(inplace=True)

        # throw out observations where BID and ask are equal
        dt.loc[dt.BID == dt.ASK, ["BID", "ASK"]] = np.nan
        # throw out wrond values for high and low
        dt.loc[(dt.H < dt.L) | (dt.H / dt.L > 8), ["H", "L"]] = np.nan

        # rescale to be in line with CRSP
        dt["VOL"] = dt["VOL"] * 1000
        dt["SHROUT"] = dt["SHROUT"] / 1000  # rescale to million

        # compute capitalization of the firm as sum of capitalization of all its traded shares
        dt["MCshare"] = abs(dt["PRC"]) * dt["SHROUT"]
        dt["MC"] = dt.groupby(["FTID", "date"])["MCshare"].transform("sum")
        # fill with capitalization based on a single class if MC missing
        dt.loc[dt.MC.isnull(), "MC"] = dt["MCshare"]
        del dt["MCshare"]
        # replace zero and negative market cap with 0.0001
        dt.loc[dt["MC"] <= 0, "MC"] = 0.0001

        # cap VOL if it is larger than 25% of market cap
        # VOL * PRC is in $ while MC is in million$
        dt.loc[(dt.PRC * dt.VOL / 1000000) > (dt.MC * 0.25), "VOL"] = dt.MC * 0.25 * 1000000 / dt.PRC

        # replace infinities with NA
        dt.replace([-np.inf, np.inf], np.nan, inplace=True)

        # multiple PERMNOs for the same date - have to delete duplicates
        dt = dt.drop_duplicates(subset=["DTID", "date"], keep="last")
        dtfull = pd.concat([dtfull, dt])
        del dt
    return dtfull


def ProcessCRSPMonthlyData(sPath):
    """loads CRSP data"""

    read_cols = [
        "permco",
        "permno",
        "mthcaldt",
        "mthprc",
        "mthret",
        "issuernm",
        "shrout",
        "primaryexch",
        "conditionaltype",
        "tradingstatusflg",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
    ]
    DT = pd.read_csv(
        os.path.join(sPath, "Inputs", "crsp.msf_v2.csv"),
        header=0,
        usecols=read_cols,
        low_memory=False,
    )
    DT.columns = [i.upper() for i in DT.columns]
    col_map = {
        "MTHCALDT": "date",
        "MTHRET": "RET",
        "MTHPRC": "PRC",
        "ISSUERNM": "COMNAM",
    }
    DT.rename(col_map, axis=1, inplace=True)

    ### monthly CRSP preprocessing date
    DT["date"] = pd.to_datetime(DT["date"])
    DT["PRC"] = abs(DT["PRC"])

    # compute capitalization of the firm as sum of capitalization of all its traded shares
    DT["MC"] = abs(DT["PRC"]) * DT["SHROUT"] / 1000
    DT["MC"] = DT.groupby(["PERMCO", "date"])["MC"].transform("sum")
    # replace zero an negative market cap with nan
    DT.loc[DT["MC"] <= 0, "MC"] = np.nan

    # add region
    DT["region"] = "USA"

    # subset to the 3 primary exchanges in the US and common equity
    # the same as DT.EXCHCD.isin([1, 2, 3]) -> subset to NYSE, AMEX, NASDAQ
    DT = DT.loc[
        (
            DT["PRIMARYEXCH"].isin(["Q", "N", "A"])
            & (DT["CONDITIONALTYPE"] == "RW")
            & (DT["TRADINGSTATUSFLG"] == "A")
        )
    ].copy()
    # the same as DT.SHRCD.isin([10, 11])
    DT = DT.loc[
        (
            (DT["SHARETYPE"] == "NS")
            & (DT["SECURITYTYPE"] == "EQTY")
            & (DT["SECURITYSUBTYPE"] == "COM")
            & (DT["USINCFLG"] == "Y")
            & (DT["ISSUERTYPE"].isin(["ACOR", "CORP"]))
        )
    ].copy()
    DT.drop(
        [
            "USINCFLG",
            "ISSUERTYPE",
            "SECURITYTYPE",
            "SECURITYSUBTYPE",
            "SHARETYPE",
            "PRIMARYEXCH",
            "TRADINGSTATUSFLG",
            "CONDITIONALTYPE",
        ],
        axis=1,
        inplace=True,
    )

    # transform returns to floats - looks like C, S, T flags are gone now
    DT["RET"] = DT["RET"].astype(float)

    # multiple PERMNOs for the same date - have to delete duplicates
    DT = DT.drop_duplicates(subset=["PERMNO", "date"], keep="last")
    DT.rename({"PERMNO": "DTID"}, inplace=True, axis=1)
    DT["DTID"] = DT["DTID"].astype(int).astype(str)
    return DT


def ProcessCRSPData(sPath):
    """loads CRSP data"""

    dbfile = os.path.join(sPath, "Inputs", "crsp.dsf_v2.sqlite")
    conn = sqlite3.connect(dbfile)
    dt = pd.read_sql_query(
        """
        SELECT
            permco, permno, yyyymmdd, dlyprc, shrout, dlyret, dlyvol, disfacshr,
            dlybid, dlyask, dlylow, dlyhigh
        FROM crsp_daily
        """,
        conn,
    )
    conn.close()
    dt.columns = [i.upper() for i in dt.columns]
    col_map = {
        "YYYYMMDD": "date",
        "DLYRET": "r",
        "DLYPRC": "PRC",
        "DLYLOW": "L",
        "DLYHIGH": "H",
        "PERMNO": "DTID",
        "DLYVOL": "VOL",
        "DLYBID": "BID",
        "DLYASK": "ASK",
    }
    dt.rename(col_map, axis=1, inplace=True)

    # covert variables
    dt["date"] = pd.to_datetime(dt["date"].astype(str))
    dt["PRC"] = abs(dt["PRC"])

    # compute capitalization of the firm as sum of capitalization of all its traded shares
    dt["MC"] = abs(dt["PRC"]) * dt["SHROUT"] / 1000
    dt["MC"] = dt.groupby(["PERMCO", "date"])["MC"].transform("sum")
    # replace zero an negative market cap with nan
    dt.loc[dt["MC"] <= 0, "MC"] = np.nan

    # transform returns to floats - looks like C, S, T flags are gone now
    dt["r"] = dt["r"].astype(float)

    # create cumulative adjusment factor for share number
    dt["DISFACSHR"] = dt.groupby("DTID")["DISFACSHR"].shift(1)
    dt["DISFACSHR"] = 1 / (1 + dt["DISFACSHR"].fillna(0))
    dt.loc[dt["DISFACSHR"] == np.inf, "DISFACSHR"] = np.nan
    dt["AF"] = dt.groupby("DTID")["DISFACSHR"].cumprod()
    del dt["DISFACSHR"]
    last_AF = (
        dt.loc[dt["AF"].notnull()].groupby("DTID")["AF"].last().reset_index().rename({"AF": "AF_last"}, axis=1)
    )
    dt = dt.merge(last_AF, on="DTID", how="left")
    dt["AF_last"] = dt["AF_last"].fillna(1)
    dt["AF"] = dt["AF"] / dt["AF_last"]
    del dt["AF_last"]

    ## subset to the 3 primary exchanges in the US and common equity
    # need to use special tables with the information
    issuer = pd.read_csv(os.path.join(sPath, "Inputs", "crsp.stkissuerinfohist.csv"), header=0)
    issuer.columns = [i.upper() for i in issuer.columns]
    issue = pd.read_csv(os.path.join(sPath, "Inputs", "crsp.stksecurityinfohist.csv"), header=0)
    issue.columns = [i.upper() for i in issue.columns]
    # the same as DT.EXCHCD.isin([1, 2, 3]) -> subset to NYSE, AMEX, NASDAQ
    issue = issue.loc[
        (
            issue["PRIMARYEXCH"].isin(["Q", "N", "A"])
            & (issue["CONDITIONALTYPE"] == "RW")
            & (issue["TRADINGSTATUSFLG"] == "A")
        )
    ].copy()
    # the same as DT.SHRCD.isin([10, 11])
    issue = issue.loc[
        ((issue["SHARETYPE"] == "NS") & (issue["SECURITYTYPE"] == "EQTY") & (issue["SECURITYSUBTYPE"] == "COM"))
    ].copy()
    issuer = issuer.loc[((issuer["USINCFLG"] == "Y") & issuer["ISSUERTYPE"].isin(["ACOR", "CORP"]))].copy()
    issue.rename({"SECINFOSTARTDT": "date", "PERMNO": "DTID"}, axis=1, inplace=True)
    issuer.rename({"ISSINFOSTARTDT": "date"}, axis=1, inplace=True)
    issue["date"] = pd.to_datetime(issue["date"])
    issuer["date"] = pd.to_datetime(issuer["date"])
    # merge
    issue.sort_values("date", inplace=True)
    issuer.sort_values("date", inplace=True)
    dt.sort_values("date", inplace=True)
    dt = pd.merge_asof(dt, issue[["DTID", "date", "SECINFOENDDT"]], on="date", by="DTID")
    dt = pd.merge_asof(dt, issuer[["PERMCO", "date", "ISSINFOENDDT"]], on="date", by="PERMCO")
    dt = dt.loc[(dt["date"] <= dt["ISSINFOENDDT"]) & (dt["date"] <= dt["SECINFOENDDT"])].copy()
    del dt["ISSINFOENDDT"], dt["SECINFOENDDT"]

    # add region
    dt["region"] = "USA"

    # multiple PERMNOs for the same date - have to delete duplicates
    dt = dt.drop_duplicates(subset=["DTID", "date"], keep="last")
    dt["DTID"] = dt["DTID"].astype(int).astype(str)
    dt.set_index(["DTID", "date"], inplace=True)
    dt.sort_index(inplace=True)

    # compute cummulative return
    dt["RI"] = 1 + dt["r"].fillna(0)
    dt["RI"] = dt["RI"].groupby("DTID").cumprod()
    dt.reset_index(inplace=True)

    return dt


def corr_ticker(y):
    """need to get rid of @ at the start of the tickers with is missing in data from WRDS"""
    if pd.isnull(y):
        return y
    if y[0:2] == "@:":
        return y[2 : len(y)]
    if y[0] == "@":
        return y[1 : len(y)]
    else:
        return y


def ProcessIBESsum(sPath):
    """loads IBES summary data"""

    ## load IBES summary
    ibes_cols = [
        "ticker",
        "statpers",
        "numest",
        "fpi",
        "fiscalp",
        "medest",
        "curcode",
        "numup",
        "numdown",
        "stdev",
        "highest",
        "lowest",
        "usfirm",
        "meanest",
    ]
    ibes = pd.read_csv(
        os.path.join(sPath, "Inputs", "ibes.statsumu_epsus.csv"), usecols=ibes_cols, low_memory=False
    )
    ibes_int = pd.read_csv(
        os.path.join(sPath, "Inputs", "ibes.statsumu_epsint.csv"), usecols=ibes_cols, low_memory=False
    )
    ibes = pd.concat([ibes, ibes_int])
    ibes.columns = [i.upper() for i in ibes.columns]
    ibes = ibes.reset_index(drop=True)
    ibes = ibes.loc[ibes["FPI"].isin(["1", "0"])].copy()
    ibes["FPI"] = ibes["FPI"].astype(int)
    ibes = ibes.loc[ibes["FISCALP"].isin(["ANN", "LTG"])].copy()  # filter out quarterly
    ibes.rename(columns={"STATPERS": "date"}, inplace=True)
    ibes["date"] = pd.to_datetime(ibes["date"].astype(str), format="%Y-%m-%d")
    ibes.set_index("date", inplace=True)
    ibes.sort_index(inplace=True)

    # load file with currencies - have to convert all estimates to USD
    curr = pd.read_csv(os.path.join(sPath, "Inputs", "ibes.hdxrati.csv"), usecols=["exrat", "curr", "anndats"])
    curr.rename(columns={"anndats": "date", "curr": "CURCODE"}, inplace=True)
    curr["date"] = pd.to_datetime(curr["date"].astype(str))
    curr = curr.drop_duplicates(subset=["CURCODE", "date"], keep="last")
    curr.set_index("date", inplace=True)
    curr.sort_index(inplace=True)
    ibes = pd.merge_asof(ibes, curr, on="date", by="CURCODE")
    # convert all estimates to USD
    ibes["MEDEST"] = ibes["MEDEST"] / ibes["exrat"]
    ibes["MEANEST"] = ibes["MEANEST"] / ibes["exrat"]
    ibes["STDEV"] = ibes["STDEV"] / ibes["exrat"]
    ibes["HIGHEST"] = ibes["HIGHEST"] / ibes["exrat"]
    ibes["LOWEST"] = ibes["LOWEST"] / ibes["exrat"]
    del ibes["CURCODE"], ibes["exrat"]

    ## add CRSP tickers
    link = pd.read_csv(
        os.path.join(sPath, "Inputs", "wrdsapps.ibcrsphist.csv"),
        usecols=["permno", "sdate", "edate", "ticker"],
        low_memory=False,
    )
    link.columns = [i.upper() for i in link.columns]
    link = link.loc[link["PERMNO"].notnull()].copy()
    link["PERMNO"] = link["PERMNO"].astype(int).astype(str)
    link.rename({"SDATE": "date"}, axis=1, inplace=True)
    link["date"] = pd.to_datetime(link["date"])
    link["EDATE"] = pd.to_datetime(link["EDATE"])
    link.sort_values(["date", "TICKER"], inplace=True)
    ibes.sort_values(["date", "TICKER"], inplace=True)
    ibes = pd.merge_asof(ibes, link, on="date", by="TICKER")
    ibes.loc[ibes["date"] > ibes["EDATE"], "PERMNO"] = np.nan
    del ibes["EDATE"]

    ## add DST tickers
    static = pd.read_csv(
        os.path.join(sPath, "Inputs", "DSTstatic.csv"),
        header=0,
        usecols=["DSCD", "IBTKR", "region"],
        low_memory=False,
    )
    static.rename({"DSCD": "DTID", "IBTKR": "TICKER"}, inplace=True, axis=1)
    static["TICKER"] = static["TICKER"].transform(corr_ticker)
    static.dropna(inplace=True)
    ibes = ibes.merge(static, on="TICKER", how="left")
    return ibes


def ProcessIBESdet(sPath):
    """loads IBES detailed data"""

    ## load IBES detailed
    ibes = pd.read_csv(
        os.path.join(sPath, "Inputs", "ibes.recddet.csv"),
        usecols=["ticker", "actdats", "estimid", "itext"],
        encoding="latin-1",
        low_memory=False,
    )
    ibes.columns = [i.upper() for i in ibes.columns]
    ibes.rename(columns={"ACTDATS": "date"}, inplace=True)
    ibes["date"] = pd.to_datetime(ibes["date"])
    ibes = ibes.dropna()

    ## add CRSP tickers
    link = pd.read_csv(
        os.path.join(sPath, "Inputs", "wrdsapps.ibcrsphist.csv"),
        usecols=["permno", "sdate", "edate", "ticker"],
    )
    link.columns = [i.upper() for i in link.columns]
    link = link.loc[link["PERMNO"].notnull()].copy()
    link["PERMNO"] = link["PERMNO"].astype(int).astype(str)
    link.rename({"SDATE": "date"}, axis=1, inplace=True)
    link["date"] = pd.to_datetime(link["date"])
    link["EDATE"] = pd.to_datetime(link["EDATE"])
    link.sort_values(["date", "TICKER"], inplace=True)
    ibes.sort_values(["date", "TICKER"], inplace=True)
    ibes = pd.merge_asof(ibes, link, on="date", by="TICKER")
    ibes.loc[ibes["date"] > ibes["EDATE"], "PERMNO"] = np.nan
    del ibes["EDATE"]

    ## add DST tickers
    static = pd.read_csv(
        os.path.join(sPath, "Inputs", "DSTstatic.csv"), header=0, usecols=["DSCD", "IBTKR", "region"]
    )
    static.rename({"DSCD": "DTID", "IBTKR": "TICKER"}, inplace=True, axis=1)
    static["TICKER"] = static["TICKER"].transform(corr_ticker)
    static.dropna(inplace=True)
    ibes = ibes.merge(static, on="TICKER", how="left")
    return ibes


def FTAddMC(FT, sPath, DTtype="DST"):
    """
    function to retrieve market cap from daily data and merge it on fundamental data
    """
    if DTtype == "DST":
        # for DST have to also ad ajex which is not readily available for WorldScope
        MC = pd.read_parquet(
            os.path.join(sPath, "ProcessedData", "DST.gzip"), columns=["date", "DTID", "MC", "AF"]
        )
    elif DTtype == "CRSP":
        MC = pd.read_parquet(os.path.join(sPath, "ProcessedData", "CRSP.gzip"), columns=["date", "DTID", "MC"])
    else:
        raise Exception("Only DST and CRSP supported.")

    MC.set_index("date", inplace=True)
    MC.dropna(inplace=True)
    MC.index.rename("FinYearEnd", inplace=True)
    FT.set_index("FinYearEnd", inplace=True)
    FT.sort_index(inplace=True)

    if DTtype == "DST":
        MC.rename({"AF": "ajex"}, inplace=True, axis=1)

        # sort out cases when there are several DTIDs per FTID
        static = pd.read_csv(
            os.path.join(sPath, "Inputs", "DSTstatic.csv"),
            usecols=["DSCD", "WC06105", "MAJOR"],
            dtype={"DSCD": str, "WC06105": str},
        )
        static.rename({"DSCD": "DTID", "WC06105": "FTID"}, inplace=True, axis=1)
        MC.reset_index(inplace=True)
        MC = MC.merge(static, on="DTID", how="left")
        MC.dropna(inplace=True)

        # sort on MAJOR and keep only one observation per date-FTID giving priority to MAJOR == 'Y'
        MC["MAJOR"] = (MC["MAJOR"] != "Y") * 1
        MC.sort_values("MAJOR", inplace=True)
        MC.drop_duplicates(subset=["FinYearEnd", "FTID"], inplace=True, keep="first")
        MC.set_index("FinYearEnd", inplace=True)
        MC.sort_index(inplace=True)
        del MC["MAJOR"]
        MC.rename({"DTID": "DTID_new"}, inplace=True, axis=1)
        FT = pd.merge_asof(FT, MC, on="FinYearEnd", by="FTID")

        # overwrite DTID where the merge was successful
        FT.loc[FT.DTID_new.notnull(), "DTID"] = FT["DTID_new"]
        del FT["DTID_new"]

    elif DTtype == "CRSP":
        MC.sort_index(inplace=True)
        FT = pd.merge_asof(FT, MC, on="FinYearEnd", by="DTID")

    return FT
