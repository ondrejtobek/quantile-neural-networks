# Quantile neural networks
This repo describes python code and input data files for "Forecasting stock return distributions around the globe with quantile neural networks".

ReadMe.xlsx provides more details on:		
- Inputs: describes all the raw inputs to run the code
- Code description: describes the python code and how to run it
- Code to figure mapping: provides mapping of figures in the paper to code used to generate them

Minimum working example is provided in ChampionModel.ipynb which produces forecasts for distributions of stock returns using WRDS data in the US.

# How to run

The code was run on WSL 2 using Ubuntu 22.04.1 LTS and python version 3.12.8. Dependencies are in requirements.txt file.

The following hardware was used: 64GB RAM, fast 16 core CPU, and RTX 4090 GPU with 24GB VRAM. The true requirements for RAM are closer to 128GB and so 128GB swap on fast SSD was set up to compensate.

The environement can also be created using the following commands:

    conda create -n quantile_nn python=3.12 -y
    conda activate quantile_nn
    conda install numpy pandas ipykernel matplotlib tqdm seaborn scipy numba scikit-learn fastparquet statsmodels lightgbm pip pyarrow openpyxl xarray -y
    pip install arch scores
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Run time

Pure run time for all the scripts is more than 2 months, most of which is taken by robustness tests and hyperparameter search.
- data preparation ~ 2 weeks
- hyperparameter tuning ~ 3weeks
- estimation all NN specifications ~2 weeks
- GARCH estimation ~ 1 week
- analysis ~ few hours
- simulation for robustness ~ 2 weeks
