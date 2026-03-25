"""Rule-based screening script for constructing the Datastream equity universe."""

import os
import pandas as pd

### filters on universe of stocks in DST, some automated some not

sPath = "/home/ondrej/Projects/Quantiles2025/Data"

## load static files
data = pd.read_csv(os.path.join(sPath, "Inputs", "DSTstatic_raw.csv"))

## the first rough filter
data = data.loc[data["TYPE"] == "EQ"].copy()  # keep only equities
data = data.loc[data["INDM4"] != "REITs"].copy()  # throw out REITs
data = data.loc[data["ISINID"] == "P"].copy()  # keep primary listings
# note that we lose some historical observations here as the primary listing can change over time and younger shares
# can then survive. Obout 600 shares are impacted

data["BDATE"] = data["BDATE"].apply(lambda x: pd.to_datetime(x))
data["TIME"] = data["TIME"].apply(lambda x: pd.to_datetime(x))

## throw out stocks from exchanges other than what we want and add region + GEOGN columns
exchanges = pd.read_excel(os.path.join(sPath, "Inputs", "ExchangeMapping.xlsx"))
# major plus add smaller exchange where delisted stocks can go - Canadian (VA), second large Singapour exchange (SS), US OTC exchange (OB),
# and 'SG', 'MU', 'HB', 'DD' in Germany
data = pd.merge(data, exchanges[["EXDSCD", "region", "GEOGN"]], on="EXDSCD")

## keep only orginary shares
data["keep"] = False

data.loc[data["TRAC"].isin(["ORD", "FULLPAID"]), "keep"] = True
# BRAZIL - keep PN for the most traded shares and discard otherwise
data.loc[data["EXDSCD"] == "SP", "keep"] = False
data.loc[
    (data["EXDSCD"] == "SP") & (data["TRAC"].isin(["ORD", "FULLPAID", "PRF"])), "keep"
] = True
data.loc[
    (data["EXDSCD"] == "SP")
    & (data["NAME"].transform(lambda x: "PN" in x))
    & (data["MAJOR"] == "N"),
    "keep",
] = False
data.loc[(data["TRAC"].isnull() == True), "keep"] = True

# ~130K securities
data = data.loc[data["keep"] == True].copy()

### US - filter using CUSIP
# throw out stocks with missing CUSIP (3.5k securities)
data.loc[
    ((data["EXDSCD"].isin(["AX", "NY", "NL", "NS", "OB"])) & (data["LOC"].isnull())),
    "keep",
] = False
data = data.loc[data["keep"] == True].copy()
del data["keep"]

### Substring Classifications

data["drop"] = False

simple_eliminate = [
    "DUPL.SEE",
    "DUPLICATE",
    "NON VOTING",
    "MUTUAL FUND",
    "WTS",
    "%",
    "SICAV",
    "REDEEMABLE",
    "REDEMPTION",
    "DEFERRED",
    "NON VTG.",
    ".TST.",
    "VVPR",
    "CERTS",
    "RESTRIC.O",
    "OPTS",
    "REDEEMED",
    "DEAD - DUP",
    "NEW SHARES",
    "PREF. DEAD",
    "SEE DUPL",
    "RECEIPT",
    "TST.UNT.",
    "RECPT.",
    "PFD",
    "DULP",
    "DEPY",
    " BONDS",
    "RCPTS",
    "PTSHP",
    "PRTF",
    "NOTES",
    "NIKKEI",
    "EXPY",
    "EXPD",
    "EXPIRED",
    " PF.",
    ".PF.",
    " PREF.",
    " PREF ",
    ".PREF",
    "CONVERTIBLE",
    " ADR ",
    " RIGHTS",
    "(500)",
    "(1000)",
    " REIT ",
    " IDR ",
    ".UTS.",
    " UTS.",
    ".FD.",
    " FD.",
    "CONVERSION INTO",
    " DUPL.",
    " UNIT ",
    "UNT.",
    ".UNIT ",
    " RTS.",
    ".RIGHT",
    "144A",
]


def delete_if_contains(df, column_name, list_of_problems):
    """Mark rows for exclusion when a string appears in a text column.

    Args:
        df (pd.DataFrame): Security-universe table containing `NAME` and `drop`.
        column_name (str): Column name to inspect for substring matches.
        list_of_problems (list[str]): List of substrings triggering row exclusion.

    """
    for problem in list_of_problems:
        dropping_condition = df[column_name].apply(lambda x: problem in x)
        df.loc[dropping_condition, "drop"] = True


delete_if_contains(data, "NAME", simple_eliminate)

### Special eliminations (more than simple NAME or NAME + EXDSCD condition)
#
data.loc[
    (data["NAME"].apply(lambda x: "PB" in x))
    & (data["EXDSCD"] == "SE")
    & (data["MAJOR"] != "Y"),
    "drop",
] = True
# deferred equities in Australia
data.loc[
    (data["NAME"].apply(lambda x: "DEF" in x))
    & (data["EXDSCD"] == "SY")
    & (data["MAJOR"] != "Y"),
    "drop",
] = True
#
data.loc[
    (data["NAME"].apply(lambda x: "UTS" in x))
    & (data["EXDSCD"] == "TR")
    & (data["TRAC"] != "ORD"),
    "drop",
] = True
# options
data.loc[
    (data["NAME"].apply(lambda x: "OPTIONS" in x)) & (data["MAJOR"] != "Y"), "drop"
] = True
# unit trusts
data.loc[
    (data["NAME"].apply(lambda x: "UNITS" in x)) & (data["ESTAT"] != "ACT."), "drop"
] = True
# rights
data.loc[
    (data["NAME"].apply(lambda x: "RIGHTS" in x))
    & (data["ESTAT"] != "ACT.")
    & (data["EXDSCD"].isin(["SY", "WL"])),
    "drop",
] = True
# $ where it means preferrential dividend and not currency
data.loc[
    (data["NAME"].apply(lambda x: "$" in x))
    & (~data["NAME"].apply(lambda x: "U$" in x)),
    "drop",
] = True

# dictionary with key being NAME and value being list of EXDSCD values in which obs. cannot be
conditions_name_exchg = {
    "GSH": ["FF", "WN", "SG", "MU", "HB", "DD"],
    "ACQ": ["SL"],  # companies issue new class of shares for acquisitions
    "SHIP INV": ["SE"],
    "WARRANT": ["LN", "BK", "WL", "NY", "TR"],
    "DFD": ["LN", "SY"],
    "FUND": [
        "TA",
        "BK",
        "SE",
        "SL",
        "JN",
        "MI",
        "LB",
        "KA",
        "LX",
        "HK",
        "BR",
        "SY",
        "WA",
    ],
    "RTS.": ["WL", "TR"],
    "UTS": ["SY"],  # unit trusts
    "CERT.": ["VN", "BR", "CP", "HL", "PR", "FF", "BD", "AM", "ZU", "OS", "TA", "IS"],
    "RESTRICTED": ["SY"],
    "AFV": ["BR"],  # VVPR strips
    "STRIP": ["BR"],
    "FB": ["JK"],  # foreign registered stocks in Jakarta
    "RNC": ["ML"],  # redeemable non-convertible
    "EDR": ["LX"],  # european deposit receipts
    "CDR": ["LX", "AM"],  # european deposit receipts
    "PDA": ["WA"],  # Allotment certificates
    " PC.": ["VN"],
    "PAID": ["LN", "SY", "DB"],
    "CONV": ["ZU", "ST"],
    "RTS": ["WL"],
    " PV": ["ML"],
    " RP": ["ML"],
    "USE ": ["HL", "ST"],
    "SPLIT": ["TR"],
    "EXH": ["TR"],
    "SUBD": ["TR"],
    "SBVTG": ["TR"],
    "INVERSION": ["LI"],
    "INVN": ["LI"],
    "'A'": ["KL"],
    " C ": ["KL"],
    "'C'": ["KL"],
    " L ": ["KL"],
    "'L'": ["KL"],
    "'O'": ["KL"],
    "1P": ["SL", "SE"],
    " 1 ": ["TV"],
    " 5 ": ["TV"],
}


# delete stocks
def delete_if_contains_and_exchg(df, conditions_dict):
    """Mark rows for exclusion based on text and exchange filters.

    Args:
        df (pd.DataFrame): Security-universe table containing `NAME`, `EXDSCD`, and `drop`.
        conditions_dict (dict[str, list[str]]): Mapping from substrings to disallowed exchange lists.

    """
    for _ in conditions_dict.items():
        name = _[0]
        exchg_list = _[1]
        exchg_cond = df["EXDSCD"].isin(exchg_list)
        name_cond = df["NAME"].apply(lambda x: name in x)
        df.loc[name_cond & exchg_cond, "drop"] = True


delete_if_contains_and_exchg(data, conditions_name_exchg)

# add back incorrectly filtred tickers
data.loc[
    data["DSCD"].isin(
        [
            "776765",
            "324719",
            "87374P",
            "8933M7",
            "2623M5",
            "871401",
            "50690C",
            "888654",
            "502595",
            "15136C",
            "51667H",
            "543593",
            "883396",
            "900670",
            "981793",
            "755415",
            "67830E",
            "326926",
        ]
    ),
    "drop",
] = False

# drop the marked stocks
data = data.loc[data["drop"] == False].copy()
del data["drop"]

# discard preferential stocks, ADR etc. - has to be done manually
DSCD = pd.read_csv(os.path.join(sPath, "Inputs", "Exclude_DSCD.csv"))
data = data.loc[~data["DSCD"].isin(DSCD["DSCD"])].copy()
data.to_csv(os.path.join(sPath, "Inputs", "DSTstatic.csv"))
