# Quantile neural networks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2408.07497)

This repo describes python code and input data files for "Forecasting stock return distributions around the globe with quantile neural networks".

Minimum working example is provided in ChampionModel.ipynb which produces forecasts for distributions of stock returns using WRDS data in the US.

---

## Reproducibility Package Information

- **Date assembled:** April 2026
- **Authors:** Jozef Barunik (`barunik@fsv.cuni.cz`), Martin Hronec, Ondrej Tobek
- **License:** MIT

---

## Repository Structure

```
.
├── Data/                       # Data folder
│   └── Inputs/                 # Input data folder
├── DataModules/                # Helper functions for data processing
│   ├── DataManager.py          # Helper class to handle data for backtesting
│   ├── ProcessData.py          # Helper functions to process raw input data
│   └── StaticScreeningDST.py   # Script used to filter universe of Datastream stocks. Need not be run
├── EstimationFunctions/        # Helper functions for estimation
│   ├── GARCH_Functions.py      # Functions to estimate and simulate GARCH as benchmark for distribution forecasts
│   ├── NN_Functions.py         # Functions to estimate neural networks, derive distribution moments, and to do analytics
│   └── SimulationFunctions.py  # Functions used for simulations in robustness tests
├── Signals/                    # Helper functions to create stock features
│   ├── CreateSignals.py        # Functions to create features and returns for estimation, forecasting, and analytics
│   ├── SignalClass.py          # Parent class to create individual signals
│   ├── Fundamental.py          # Classes to create fundamental data signals
│   ├── IBES.py                 # Classes to create analyst forecast signals
│   ├── Market.py               # Classes to create market data signals
│   └── Volatility.py           # Classes to create volatility signals
├── 01_CreateData.py            # Script to process all the data for estimation, forecasting, and analytics
├── 02_QuantileNN_Estimation.py # Script to estimate models with neural network and do forecasting
├── 03_GARCH_Estimation.py      # Script to estimate and simulate GARCH as benchmark for distribution forecasts
├── 04_DistributionMoments.py   # Script to derive distribution moments from quantile forecasts
├── 05_Analytics.py             # Main script to produce the tables
├── 06_RobustnessGBRT_RF.py     # Robustness tests with tree-based methods
├── 06_RobustnessSimulation.py  # Robustness tests with simulations
├── ChampionModel.ipynb         # Minimum working example to run the champion model using WRDS data in the US
├── requirements.txt            # All Python dependencies with versions
└── README.md                   # This file
```

---

## Computing Environment


### Hardware and expected runtime
| Component | Machine |
|---|---|
| **OS** | Linux (Ubuntu 22.04.1 LTS) |
| **CPU** | 16-core |
| **GPU** | RTX 4090 with 24GB VRAM|
| **RAM** | 128 GB |
| **Python** | 3.12 |


- **Language:** Python 3.12
- **Package manager:** `.venv` / pip
- **Key dependencies** (see `requirements.txt` for full list with pinned versions):


| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.3 | Numerical computation |
| `pandas` | 3.0.1 | Data manipulation |
| `scikit-learn` | 1.8.0 | Machine learning metrics and classifiers |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `scipy` | 1.17.1 | Statistical tests (Wilcoxon, Friedman) |
| `arch` | 6.3.0 | GARCH model estimation |
| `fastparquet` | 2024.5.0 | Data I/O |
| `ipykernel` | 7.2.0 | Jupyter kernel for the project virtual environment |
| `jupyter` | 1.1.1 | Notebook interface for `ChampionModel.ipynb` |
| `pyarrow` | 23.0.1 | Data I/O |
| `openpyxl` | 3.1.2 | Read excel files |
| `tqdm` | 4.66.5 | Progress bar |
| `numba` | 0.64.0 | Speed up for numerical computations |
| `lightgbm` | 4.6.0 | For robustness test with tree-based methods |
| `torch` | 2.2.2 | PyTorch framework for neural networks. The documented GPU setup targets the official CUDA 12.1 wheel. |
| `scores` | 1.3.0 | Scoring rules to compare distributional forecasts |
| `statsmodels` | 0.14.4 | Statistical models |
| `xarray` | 2024.11.0 | Dependency for other packages |
| `python_dateutil` | 2.9.0.post0 | Dependency for other packages |

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/ondrejtobek/quantile-neural-networks.git
cd quantile-neural-networks
```

### 2. Create and activate `.venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the official PyTorch CUDA 12.1 wheel

GPU used is an `RTX 4090` on `Ubuntu 22.04.1`. Reinstall `torch` from the official CUDA 12.1 wheel index:

```bash
pip install --force-reinstall torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

This step requires a recent NVIDIA driver on Linux. The code uses CUDA automatically when available and otherwise falls back to CPU.

### 5. Register the virtual environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name quantile_nn --display-name "Python (.venv) quantile_nn"
```

You can then open `ChampionModel.ipynb` in Jupyter and select the `Python (.venv) quantile_nn` kernel.

---

## Data Description

### Input data

The input data is split into sharable and non-sharable datasets.

#### Sharable datasets included in the replication package

The following files are included in full under `Data/Inputs/` and can be used directly.

| File | Source | Status in package | Description |
|------|--------|-------------------|-------------|
| `CPIAUCNS.csv` | FRED | Full file included | Monthly US CPI time series `CPIAUCNS` downloaded from FRED. |
| `FF3.CSV` | Ken French Data Library | Full file included | Fama-French 3-factor returns. Only the risk-free rate is used in this project. |
| `AnomaliesMeta.xlsx` | Manual | Full file included | Static metadata on anomalies created manually by the authors. |
| `ExchangeMapping.xlsx` | Manual | Full file included | Mapping of exchanges and regions for the international sample. |
| `Exclude_DSCD.csv` | Manual | Full file included | List of DSCD identifiers filtered out after manual investigation. |

#### Non-sharable datasets required for full replication

The main empirical analysis also relies on proprietary historical data from WRDS and LSEG DataStream / Worldscope. The full vendor datasets are **not** included in this repository because of licence restrictions.

Files from these sources that are present under `Data/Inputs/` are provided only as reduced examples or schema references (for example top rows, extracts, or variable/header information). They document the expected file names, formats, and columns, but they are **not sufficient** to reproduce the full paper results. For full replication, these files must be replaced with the full licensed datasets obtained from the original providers.

| Provider | File(s) expected by the code | Status in package | Description |
|----------|-------------------------------|-------------------|-------------|
| WRDS | `crsp.msf_v2.csv` | Example extract only | Monthly CRSP time series data. |
| WRDS | `crsp.dsf_v2.sqlite` and `crsp.dsf_v2.csv` | Example extract only | Daily CRSP time series data. |
| WRDS | `comp.funda.csv` | Example extract only | Compustat fundamentals. |
| WRDS | `crsp.ccmxpf_linktable.csv` | Example extract only | CRSP-Compustat link table. |
| WRDS | `crsp.comphist.csv` | Example extract only | Compustat history table used for SIC. |
| WRDS | `crsp.comphead.csv` | Example extract only | Most recent Compustat snapshot used as SIC fallback. |
| WRDS | `crsp.stkissuerinfohist.csv` | Example extract only | CRSP issuer history used for universe filtering. |
| WRDS | `crsp.stksecurityinfohist.csv` | Example extract only | CRSP security history used for universe filtering. |
| WRDS | `ibes.statsumu_epsus.csv` | Example extract only | IBES summary table for the US. |
| WRDS | `ibes.statsumu_epsint.csv` | Example extract only | IBES summary table for international markets. |
| WRDS | `ibes.hdxrati.csv` | Example extract only | IBES exchange rates used for conversion to USD. |
| WRDS | `ibes.recddet.csv` | Example extract only | IBES detailed recommendation table. |
| WRDS | `wrdsapps.ibcrsphist.csv` | Example extract only | Mapping of IBES ticker to CRSP PERMNO. |
| LSEG DataStream / Worldscope | `DSTfundamental.csv` | Example extract only | Worldscope fundamental data. |
| LSEG DataStream / Worldscope | `DSTstatic_raw.csv` and `DSTstatic.csv` | Example extract only | Static DataStream security data before and after cleaning. |
| LSEG DataStream / Worldscope | `DSTstatic_all_DSCD.csv` | Example extract only | List of all DSCD identifiers used in the static download. |
| LSEG DataStream / Worldscope | `DSTd.csv` and `DSTd2.csv` | Example extract only | Daily market data downloaded from DataStream. |
| LSEG DataStream / Worldscope | `DST_holidays_ts.csv` and `DST_holidays_map.csv` | Example extract only | Holiday calendars and mapping files used in the international sample. |

The full non-sharable datasets can be obtained as follows:

- **WRDS:** CRSP, Compustat, and IBES historical data must be downloaded from WRDS. The original paper used an older WRDS table format covering `1926-2018`, which is no longer available. The current code expects updated WRDS tables covering `1926-2023`. The example files under `Data/Inputs/` show the required file names and column structure expected by the code.
- **LSEG DataStream / Worldscope:** Worldscope fundamentals and DataStream market data must be obtained from `LSEG`, typically via API, Excel plugin, or terminal. The international dataset covering `1980-2018` was originally downloaded in `2019` under a commercial licence. The example files under `Data/Inputs/` preserve the expected file names and downloaded variable codes.

### Intermediary data files

The following intermediary files are created during code run. The intermediary files are not shared due to licencing issues (the licence is not permitting sharing of derived data) and their big size.

| File(s) | Generated by | Description |
|---|---|---|
| `Data/ProcessedData/*.gzip` | `01_CreateData.py` | Processed raw input data |
| `Data/Features/*.gzip` | `01_CreateData.py` | Transformed input data used for estimation and analysis |
| `Data/Predict/*.gzip` | `02_QuantileNN_Estimation.py`, `03_GARCH_Estimation.py`, `06_RobustnessGBRT_RF.py` | Fitted model predictions |
| `Data/Output/*.gzip` | `04_DistributionMoments.py`, `05_Analytics.py`, `06_RobustnessSimulation.py`, `06_RobustnessGBRT_RF.py` | Tables and figures with results |
| `Data/Simulation/*.gzip` | `06_RobustnessSimulation.py` | Used for robustness test via simulations |

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
| Robustness - tree-based methods | 06_RobustnessGBRT_RF.py | ~ few days |
| Robustness - simulations | 06_RobustnessSimulation.py | ~ 2 weeks |
| Minimum working example | ChampionModel.ipynb | ~ 4 days |

---

## Paper Tables and Figures — Output Mapping

| Figure in pdf | File | Location in Code |
|---------------|------|------------------|
| Figure 1 | Manual in Latex | |
| Figure 2 | 05_Analytics.py | Microsoft2008Oct.png |
| Figure 3 | 05_Analytics.py | cdf_pdf_msft_corrected.pdf |
| Figure 4 | 04_DistributionMoments.py | Moment_hist_corrected_1_4.pdf |
| Figure B.1 | Manual in Latex | |
| Figure D.1 | Manual in Latex | |
| Figure E.1 | 05_Analytics.py | DirichletDensity.png |
| Figure H.1 | 06_RobustnessSimulation.py | Simulation_GARCH_param_hist.png |
| Figure J.1 | 05_Analytics.py | r_scale_adj.png |
| Table 1 | 05_Analytics.py | Observation_counts.txt |
| Table 2 | 04_DistributionMoments.py | Moment_summary_stat.txt |
| Table 3 | 04_DistributionMoments.py | VariableCorrelations.txt |
| Table 4 | 05_Analytics.py | TwoStageMotivation1.txt |
| Table 5 | 05_Analytics.py | MSEvsQuantile2.txt |
| Table 6 | 05_Analytics.py | VolaPredict.txt |
| Table 7 | 05_Analytics.py | QuantilePortfoliosUSA.txt |
| Table 8 | 05_Analytics.py | SingleSorts1.txt, SingleSorts2.txt, SingleSorts3.txt, SingleSorts4.txt, SingleSorts5.txt |
| Table A.1 | Manual in Latex | |
| Table B1 | 05_Analytics.py | GARCH_estim1.txt |
| Table C1 | 04_DistributionMoments.py | VariableCorrelationsEurope.txt, VariableCorrelationsJapan.txt, VariableCorrelationsAsiaPacific.txt |
| Table C2 | 05_Analytics.py | QuantilePortfoliosEurope.txt, QuantilePortfoliosJapan.txt, QuantilePortfoliosAsiaPacific.txt |
| Table C3 | 05_Analytics.py | SingleSorts1_VW.txt, SingleSorts2_VW.txt, SingleSorts3_VW.txt, SingleSorts4_VW.txt, SingleSorts5_VW.txt |
| Table C4 | 05_Analytics.py | QuantilePortfoliosBenchmarkUSA.txt |
| Table C5 | 05_Analytics.py | SingleSorts1_NN2.txt, SingleSorts3_NN2.txt, SingleSorts4_NN2.txt, SingleSorts5_NN2.txt |
| Table C6 | 05_Analytics.py | MSEvsQuantile2.txt |
| Table C7 | 05_Analytics.py | VolaPredict.txt |
| Table D1 | Manual in Latex | |
| Table D2 | 05_Analytics.py | HyperparameterSearch1.txt |
| Table D3 | 05_Analytics.py | HyperparameterSearch2.txt |
| Table E1 | 04_DistributionMoments.py | MomentAdjustment1.txt |
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
