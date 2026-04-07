# Quantile neural networks

[![Python 3.10](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2408.07497)

This repo describes python code and input data files for "Forecasting stock return distributions around the globe with quantile neural networks".

Minimum working example is provided in ChampionModel.ipynb which produces forecasts for distributions of stock returns using WRDS data in the US.

---

## Reproducibility Package Information

- **Date assembled:** April 2026
- **Authors:** Jozef Barunik (`barunik@fsv.cuni.cz`), Martin Hronec, Ondrej Tobek
- **Affiliation:** Charles University, Institute of Economic Studies
- **License:** MIT

---

## Repository Structure

**TBA** create the file map incorporating the table below.

```
.
├── Data/                       # Data
...
├── requirements.txt            # All Python dependencies with versions
└── README.md                   # This file
```

| File | Type | Description |
|------|------|-------------|
| DataModules/DataManager.py | Helper functions | Helper class to handle data for backtesting. Loads appropriate data and subsets to the relevant time snapshots. |
| DataModules/ProcessData.py | Helper functions | Helper functions to process raw input data. |
| DataModules/StaticScreeningDST.py | Helper functions | Script used to filter universe of Datastream stocks. Need not be run. |
| EstimationFunctions/GARCH_Functions.py | Helper functions | Functions to estimate and simulate GARCH as benchmark for distribution forecasts. |
| EstimationFunctions/NN_Functions.py | Helper functions | Functions to estimate neural networks, derive distribution moments, and to do analytics. |
| EstimationFunctions/SimulationFunctions.py | Helper functions | Functions used for simulations in robustness tests. |
| Signals/CreateSignals.py | Helper functions | Functions to create features and returns for estimation, forecasting, and analytics. |
| Signals/SignalClass.py | Helper functions | Parent class to create individual signals. |
| Signals/Fundamental.py | Helper functions | Classes to create fundamental data signals. |
| Signals/IBES.py | Helper functions | Classes to create analyst forecast signals. |
| Signals/Market.py | Helper functions | Classes to create market data signals. |
| Signals/Volatility.py | Helper functions | Classes to create volatility signals. |
| 01_CreateData.py | Run script | Script to process all the data for estimation, forecasting, and analytics. |
| 02_QuantileNN_Estimation.py | Run script | Script to estimate models with neural network and do forecasting. |
| 03_GARCH_Estimation.py | Run script | Script to estimate and simulate GARCH as benchmark for distribution forecasts. |
| 04_DistributionMoments.py | Run script | Script to derive distribution moments from quantile forecasts. |
| 05_Analytics.py | Run script | Main script to produce the tables. |
| 06_RobustnessGBRT_RF.py | Run script | Robustness tests with tree-based methods. |
| 06_RobustnessSimulation.py | Run script | Robustness tests with simulations. |
| ChampionModel.ipynb | Notebook | Minimum working example creating data and running the champion model using WRDS data in the US. |


---

## Computing Environment

| Component | Machine |
|---|---|
| **OS** | Linux (Ubuntu 22.04.1 LTS) |
| **CPU** | 16-core |
| **GPU** | RTX 4090 with 24GB VRAM|
| **RAM** | 128 GB |
| **Python** | 3.12 |

**Language:** Python 3.12
**Package manager:** conda / pip
**Key dependencies** (see `requirements.txt` for full list with pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.3 | Numerical computation |
| `pandas` | 3.0.1 | Data manipulation |
| `scikit-learn` | 1.8.0 | Machine learning metrics and classifiers |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `scipy` | 1.17.1 | Statistical tests (Wilcoxon, Friedman) |
| `arch` | 6.3.0 | GARCH model estimation |
| `fastparquet` | 6.3.0 | Data I/O |
| `pyarrow` | 23.0.1 | Data I/O |
| `openpyxl` | 3.1.2 | Read excel files |
| `tqdm` | 4.66.5 | Progress bar |
| `numba` | 0.64.0 | Speed up for numerical computations |
| `lightgbm` | 4.6.0 | For robustness test with tree-based methods |
| `torch` | 2.2.2 | Pytorch framework for neural networks |
| `scores` | 1.3.0 | Scoring rules to compare distributional forecasts |
| `statsmodels` | 0.14.4 | Statistical models |
| `xarray` | 2024.11.0 | Dependency for other packages |
| `python_dateutil` | 2.9.0.post0 | Dependency for other packages |

---

## Installation

### 1. Create conda environment

```bash
conda create -n quantile_nn python=3.12 -y
conda activate quantile_nn
conda install numpy pandas ipykernel matplotlib tqdm seaborn scipy numba scikit-learn fastparquet statsmodels lightgbm pip pyarrow openpyxl xarray -y
pip install arch scores
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 2. Clone this repository and install dependencies

```bash
git clone https://github.com/ondrejtobek/quantile-neural-networks.git
cd quantile-neural-networks
pip install -r requirements.txt
```

### 3. Get input data from WRDS / Datastream

See next section for more details on vendor input data that is not part of the repo due to big size and licencing issues.

---

## Data Description

### Input data

Individual input files are as follows.

| Source | File | Description | Example file description |
|--------|------|-------------|--------------------------|
| FRED | CPIAUCNS.csv | Monthly US CPI time series CPIAUCNS downloaded from FRED. | Full file |
| Ken French | FF3.csv | Fama French 3 factor returns sourced from Ken French's website. Only risk free rate is used. | Full file |
| Manual | AnomaliesMeta.xlsx | Static data on anomalies. Created manually by the authors. | Full file |
| WRDS | crsp.msf_v2.csv | Sourced directly from WRDS. Monthy time series data from CRSP. | Top 10 rows |
| WRDS | crsp.dsf_v2.sqlite | Sourced directly from WRDS. Daily time series data from CRSP. | Top 10 rows, extract saved as crsp.dsf_v2.csv |
| WRDS | comp.funda.csv | Sourced directly from WRDS. Fundamental data from Compustat. | Top 10 rows |
| WRDS | crsp.ccmxpf_linktable.csv | Sourced directly from WRDS. WRDS link table between Compustatn and CRSP. | Top 10 rows |
| WRDS | crsp.comphist.csv | Sourced directly from WRDS. Compustat history table used for SIC. | Top 10 rows |
| WRDS | crsp.comphead.csv | Sourced directly from WRDS. Compustat most recent snapshot table used as a fallback for SIC. | Top 10 rows |
| WRDS | crsp.stkissuerinfohist.csv | Sourced directly from WRDS. CRSP issuer history table used for universe filter. | Top 10 rows |
| WRDS | crsp.stksecurityinfohist.csv | Sourced directly from WRDS. CRSP security history table used for universe filter. | Top 10 rows |
| WRDS | ibes.statsumu_epsus.csv | Sourced directly from WRDS. IBES summary table for the US. | Top 10 rows |
| WRDS | ibes.statsumu_epsint.csv | Sourced directly from WRDS. IBES summary table for the international markets. | Top 10 rows |
| WRDS | ibes.hdxrati.csv | Sourced directly from WRDS. IBES exchange rates to convert everything to USD. | Top 10 rows |
| WRDS | ibes.recddet.csv | Sourced directly from WRDS. IBES detailed recommendation table. | Top 10 rows |
| WRDS | wrdsapps.ibcrsphist.csv | Sourced directly from WRDS. Mapping of IBES ticker to CRSP PERMNO. | Top 10 rows |
| DataStream | DSTfundamental.csv | WorldScope fundamental data. Downloaded via excel plugin. Code for column download for WorldScope is in column names. | Top 10 rows |
| DataStream | DSTstatic_raw.csv | Raw static data from DataStream before filtering. | Top 10 rows |
| DataStream | DSTstatic.csv | Static data from DataSteam on tickers post cleaning. Downloaded via excel plugin. Code for column download for is in column names. | Top 10 rows and separately list of all DSCD in DSTstatic_all_DSCD.csv |
| DataStream | DSTd.csv and DSTd2.csv | Daily time series data for market data. Downloaded via excel plugin. Code for column download for is in column names. | Top 10 rows |
| DataStream | DST_holidays_ts.csv | Time series file with holidays | Top 10 rows |
| DataStream | DST_holidays_map.csv | Mapping of holiday lists. | Top 10 rows |
| DataStream / Manual | ExchangeMapping.xlsx | Mapping of exchanges and regions for international sample. | Full file |
| Manual | Exclude_DSCD.csv | List of DSCD that are filtered out based on manual investigation. | Full file |

Full input files that are not part of the replication package can be obtained from:

- **WRDS:** CRSP, Compustat, and IBES historical data needs to be downloaded from WRDS which is the standard source of data for equity studies in academia. The original version of the paper (and table in the main text) was written using old format of WRDS tables covering 1926-2018 which is no longer available. The current code was adjusted to consume updated version of WRDS tables covering 1926-2023.
- **Datastream:** Worldscope and Datastream market data is frequently used as a source of international equities data in academic studies. It is available from LSEG via API / excel pluggin / terminal. The data covering 1980-2018 was downloaded in 2019 with a commercial licence.

List of variables can be obtained in the individual header files in Data/Inputs/. directory.

### Intermediary data files

The following intermediary files are created during code run. The intermediary files are not shared due to their big size and licencing issues (the licence is not permitting sharing of derived data).

| File(s) | Generated by | Description |
|---|---|---|
| `Data/ProcessedData/*.gzip` | `01_CreateData.py` | Processed raw input data |
| `Data/Features/*.gzip` | `01_CreateData.py` | Transformed input data used for estimation and analysis |
| `Data/Predict/*.gzip` | `02_QuantileNN_Estimation.py`, `03_GARCH_Estimation.py`, `06_RobustnessGBRT_RF.py` | Fitted model predictions |
| `Data/Output/*.gzip` | `04_MomentAdjustment.py`, `05_Analytics.py`, `06_RobustnessSimulation.py`, `06_RobustnessGBRT_RF.py` | Tables and figures with results |
| `Data/Simulation/*.gzip` | `06_RobustnessSimulation.py` | Used for robusness test via simulations |

---

## Reproducing the Results

### Description of python run files

To replicate the main results of the paper it is needed to run 01_\*.py to 05_\*.py python run files. 06_\*.py run files provide results for robustness tests.

| File | Description |
|------|-------------|
| 01_CreateData.py | Script to process all the data for estimation, forecasting, and analytics. |
| 02_QuantileNN_Estimation.py | Script to estimate models with neural network and do forecasting. |
| 03_GARCH_Estimation.py | Script to estimate and simulate GARCH as benchmark for distribution forecasts. |
| 04_DistributionMoments.py | Script to derive distribution moments from quantile forecasts. |
| 05_Analytics.py | Main script to produce the tables. |
| 06_RobustnessGBRT_RF.py | Robustness tests with tree-based methods. |
| 06_RobustnessSimulation.py | Robustness tests with simulations. |
| ChampionModel.ipynb | Minimum working example creating data and running the champion model using WRDS data in the US. |

### Expected Runtimes

Pure run time for all the scripts is more than 2 months, most of which is taken by robustness tests and hyperparameter search.

| Description | File(s) | Runtime |
|------|------|------|
| Data preparation | 01_CreateData.py | ~ 2 weeks |
| Hyperparameter tuning | 02_QuantileNN_Estimation.py | ~ 3 weeks |
| Estimation all NN specifications | 02_QuantileNN_Estimation.py | ~ 2 weeks |
| GARCH estimation | 03_GARCH_Estimation.py | ~ 1 week |
| Derive distribution moments | 04_DistributionMoments.py | ~ few hours |
| Analysis, creation of figures / tables | 05_Analytics.py | ~ few hours |
| Tobustness - tree-based methods | 06_RobustnessGBRT_RF.py | ~ few days |
| Robustness - simulations | 06_RobustnessSimulation.py | ~ 2 weeks |
| Minimum working example | ChampionModel.ipynb | ~ 4 days |

---

## Paper Tables and Figures — Output Mapping

| Figure in pdf | File | Location in Code |
|---------------|------|------------------|
| Figure 1 | Manual in Latex | |
| Figure 2 | 05_Analytics.py | Microsoft2008Oct.png |
| Figure 3 | 05_Analytics.py | cdf_pdf_msft_corrected.pdf |
| Figure 4 | 04_MomentAdjustment.py | Moment_hist_corrected_1_4.pdf |
| Figure B.1 | Manual in Latex | |
| Figure D.1 | Manual in Latex | |
| Figure E.1 | 05_Analytics.py | DirichletDensity.png |
| Figure H.1 | 06_RobustnessSimulation.py | Simulation_GARCH_param_hist.png |
| Figure J.1 | 05_Analytics.py | r_scale_adj.png |
| Table 1 | 05_Analytics.py | Observation_counts.txt |
| Table 2 | 04_MomentAdjustment.py | Moment_summary_stat.txt |
| Table 3 | 04_MomentAdjustment.py | VariableCorrelations.txt |
| Table 4 | 05_Analytics.py | TwoStageMotivation1.txt |
| Table 5 | 05_Analytics.py | MSEvsQuantile2.txt |
| Table 6 | 05_Analytics.py | VolaPredict.txt |
| Table 7 | 05_Analytics.py | QuantilePortfoliosUSA.txt |
| Table 8 | 05_Analytics.py | SingleSorts1.txt, SingleSorts2.txt, SingleSorts3.txt, SingleSorts4.txt, SingleSorts5.txt |
| Table A.1 | Manual in Latex | |
| Table B1 | 05_Analytics.py | GARCH_estim1.txt |
| Table C1 | 04_MomentAdjustment.py | VariableCorrelationsEurope.txt, VariableCorrelationsJapan.txt, VariableCorrelationsAsiaPacific.txt |
| Table C2 | 05_Analytics.py | QuantilePortfoliosEurope.txt, QuantilePortfoliosJapan.txt, QuantilePortfoliosAsiaPacific.txt |
| Table C3 | 05_Analytics.py | SingleSorts1_VW.txt, SingleSorts2_VW.txt, SingleSorts3_VW.txt, SingleSorts4_VW.txt, SingleSorts5_VW.txt |
| Table C4 | 05_Analytics.py | QuantilePortfoliosBenchmarkUSA.txt |
| Table C5 | 05_Analytics.py | SingleSorts1_NN2.txt, SingleSorts3_NN2.txt, SingleSorts4_NN2.txt, SingleSorts5_NN2.txt |
| Table C6 | 05_Analytics.py | MSEvsQuantile2.txt |
| Table C7 | 05_Analytics.py | VolaPredict.txt |
| Table D1 | Manual in Latex | |
| Table D2 | 05_Analytics.py | HyperparameterSearch1.txt |
| Table D3 | 05_Analytics.py | HyperparameterSearch2.txt |
| Table E1 | 04_MomentAdjustment.py | MomentAdjustment1.txt |
| Table F1 | 05_Analytics.py | MSEvsQuantileUpdate2.txt |
| Table F2 | 05_Analytics.py | VolaPredictUpdate.txt |
| Table F3 | 05_Analytics.py | TwoStageMotivationUpdate1.txt |
| Table F4 | 05_Analytics.py | MSEvsQuantileUpdate1.txt |
| Table G1 | 05_Analytics.py | TwoStageMotivation1_Qs.txt |
| Table G2 | 05_Analytics.py | TwoStageScoring1.txt |
| Table H1 | 06_RobustnessSimulation.py | Simulations_RMSE.txt |
| Table I1 | 06_RobustnessGBRT_RF.py | QuantileLoss_RF.txt |
| Table I2 | 06_RobustnessGBRT_RF.py | MSEvsQuantile1_RF.txt |

Note: You can get to the relevant code section by openning the relevant python script and searching for the output file name.
