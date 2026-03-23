import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from EstimationFunctions.NN_Functions import *

# path to data
sPath = "./Data"

# file names
files = {
    "W_file_22d": "MLdata_W_22d.gzip",
    "W_file_22d_full": "MLdata_W_22d_full.gzip",
    "M_file": "MLdata.gzip",
    "M_file_full": "MLdata_full.gzip",
}


# define variables to be used generally
anomalies = get_anomalies_list(sPath)
vol_vars = (
    [f"EWMAVol{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMARange{i}" for i in [20, 10, 6, 4, 2, 1]]
    + [f"EWMAVolD{i}" for i in [20, 10, 6]]
    + ["TV3M", "TV6M", "TV12M"]
)
mkt_mean_vars = ["MktAvg10_EW", "MktAvg6_EW", "MktAvg4_EW", "MktAvg1_EW", "MktAvg0.1_EW"]
taus = (
    [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.075]
    + [0.925, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99995]
    + [i / 20 for i in range(1, 20)]
)
taus.sort()

####################################################################################################
########################################## Full Sample #############################################

### quantile regression hyperparameter search
sample_split = validation_logic(first_year=1990, last_year=1994, year_step=1)
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param_base = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [128, 128, 4, 128, 128],
    "hidden_sizes2": [8],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 10,
}
data = get_data(sPath, files["W_file_22d_full"], vol_vars, mkt_mean_vars, regions=["USA"])

# parameters search for one architecture
Settings = [
    [0.01, 0.0],
    [0.01, 0.2],
    [0.01, 0.4],
    [0.001, 0.0],
    [0.001, 0.2],
    [0.001, 0.4],
    [0.0001, 0.0],
    [0.0001, 0.2],
    [0.0001, 0.4],
    [0.0003, 0.2],
    [0.003, 0.2],
    [0.0003, 0.3],
    [0.0003, 0.1],
]
for LR, DR in Settings:
    print(f"LR: {LR}, DR: {DR}")
    param = param_base
    param["initial_lr"] = LR
    param["dropout_rate"] = DR
    pred, _ = train_loop(data, [], sample_split, param, inputs1, inputs2, pred_file=["W"])
    pred["Setting"] = f"LR: {LR}, DR: {DR}"
    DR_, LR_ = str(DR).replace("0.", ""), str(LR).replace("0.", "")
    pred.to_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_full_parameter_search_LR{LR_}_DR{DR_}.gzip"),
        compression="gzip",
    )

# architecture search
Settings = [
    ["s1_2x128_s2_0", [128, 128, 4, 128, 128], []],
    ["s1_2x128_s2_16", [128, 128, 4, 128, 128], [16]],
    ["s1_2x64_s2_8", [64, 64, 4, 64, 64], [8]],
    ["s1_2x256_s2_8", [256, 256, 4, 256, 256], [8]],
    ["s1_3x128_s2_8", [128, 128, 128, 4, 128, 128, 128], [8]],
    ["s1_1x128_s2_8", [128, 4, 128], [8]],
    ["s1_4x64_s2_8", [64, 64, 64, 64, 4, 64, 64, 64, 64], [8]],
]
param_base["initial_lr"] = 0.0003
param_base["dropout_rate"] = 0.2
for Spec, hs1, hs2 in Settings:
    print(f"Spec: {Spec}")
    param = param_base
    param["hidden_sizes"] = hs1
    param["hidden_sizes2"] = hs2
    pred, _ = train_loop(data, [], sample_split, param, inputs1, inputs2, pred_file=["W"])
    pred["Setting"] = Spec
    pred.to_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_full_parameter_search_{Spec}.gzip"),
        compression="gzip",
    )


### quantile regression two-stage with bottleneck
sample_split = validation_logic()
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [128, 128, 4, 128, 128],
    "hidden_sizes2": [8],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 10,
}
data = get_data(sPath, files["W_file_22d_full"], vol_vars, mkt_mean_vars, regions=["USA"])
data_m = get_data(sPath, files["M_file_full"], vol_vars, mkt_mean_vars)
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, inputs2)
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_M_full.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_M_full_m.gzip"), compression="gzip")
del pred, pred_m, data, data_m

### quantile regression two-stage linear
sample_split = validation_logic()
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [],
    "hidden_sizes2": [],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 10,
}
data = get_data(sPath, files["W_file_22d_full"], vol_vars, mkt_mean_vars, regions=["USA"])
data_m = get_data(sPath, files["M_file_full"], vol_vars, mkt_mean_vars)
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, inputs2)
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_NN0_M_full.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_NN0_M_full_m.gzip"), compression="gzip")
del pred, pred_m, data, data_m

### quantile regression directly one-stage with raw returns
sample_split = validation_logic()
inputs1 = anomalies + [i + "_raw" for i in vol_vars] + mkt_mean_vars
data = get_data(
    sPath,
    files["W_file_22d_full"],
    vol_vars,
    mkt_mean_vars,
    regions=["USA"],
    rescale_mean=False,
)
data_m = get_data(sPath, files["M_file_full"], vol_vars, mkt_mean_vars, rescale_mean=False)

## just linear regression
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 10,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN0_full_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN0_full_M_m.gzip"), compression="gzip")
del pred, pred_m

## NN with one layer
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [32],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 10,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN1_full_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN1_full_M_m.gzip"), compression="gzip")
del pred, pred_m

## NN with two layers
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 10,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN2_full_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN2_full_M_m.gzip"), compression="gzip")
del pred, pred_m

## NN2 MSE
param = {
    "loss_f": "mse",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": 1,
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 10,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_raw_full_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_raw_full_M_m.gzip"), compression="gzip")
del pred, pred_m


### mse forecast with cross-sectionally standardized returns
sample_split = validation_logic()
data = get_data(sPath, files["W_file_22d_full"], vol_vars, [], regions=["USA"], adjust_r="standardize")
data_m = get_data(sPath, files["M_file_full"], vol_vars, [], adjust_r="standardize")

## NN2 with volatility variables
inputs1 = anomalies + vol_vars
param = {
    "loss_f": "mse",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": 1,
    "initial_lr": 0.0003,
    "epochs": 100,
    "epoch_size": 3000000,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "num_networks": 10,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_full_vol_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_full_vol_M_m.gzip"), compression="gzip")
del pred, pred_m


####################################################################################################
########################################## Liquid Sample ###########################################

### quantile regression hyperparameter search
sample_split = validation_logic(first_year=1990, last_year=1994, year_step=1)
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param_base = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [128, 128, 4, 128, 128],
    "hidden_sizes2": [8],
    "output_size": len(taus),
    "initial_lr": 0.001,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 20,
}
data = get_data(sPath, files["W_file_22d"], vol_vars, mkt_mean_vars, regions=["USA"])

# parameters search for one architecture
Settings = [
    [0.01, 0.0],
    [0.01, 0.2],
    [0.01, 0.4],
    [0.001, 0.0],
    [0.001, 0.2],
    [0.001, 0.4],
    [0.0001, 0.0],
    [0.0001, 0.2],
    [0.0001, 0.4],
    [0.0003, 0.2],
    [0.003, 0.2],
    [0.001, 0.1],
    [0.001, 0.3],
]
for LR, DR in Settings:
    print(f"LR: {LR}, DR: {DR}")
    param = param_base
    param["initial_lr"] = LR
    param["dropout_rate"] = DR
    pred, _ = train_loop(data, [], sample_split, param, inputs1, inputs2, pred_file=["W"])
    pred["Setting"] = f"LR: {LR}, DR: {DR}"
    DR_, LR_ = str(DR).replace("0.", ""), str(LR).replace("0.", "")
    pred.to_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_parameter_search_LR{LR_}_DR{DR_}.gzip"),
        compression="gzip",
    )

# architecture search
Settings = [
    ["s1_2x128_s2_0", [128, 128, 4, 128, 128], []],
    ["s1_2x128_s2_16", [128, 128, 4, 128, 128], [16]],
    ["s1_2x64_s2_8", [64, 64, 4, 64, 64], [8]],
    ["s1_2x256_s2_8", [256, 256, 4, 256, 256], [8]],
    ["s1_3x128_s2_8", [128, 128, 128, 4, 128, 128, 128], [8]],
    ["s1_1x128_s2_8", [128, 4, 128], [8]],
    ["s1_4x64_s2_8", [64, 64, 64, 64, 4, 64, 64, 64, 64], [8]],
]
param_base["initial_lr"] = 0.001
param_base["dropout_rate"] = 0.2
for Spec, hs1, hs2 in Settings:
    print(f"Spec: {Spec}")
    param = param_base
    param["hidden_sizes"] = hs1
    param["hidden_sizes2"] = hs2
    pred, _ = train_loop(data, [], sample_split, param, inputs1, inputs2, pred_file=["W"])
    pred["Setting"] = Spec
    pred.to_parquet(
        os.path.join(sPath, "Predict", f"NN_clean_M_parameter_search_{Spec}.gzip"),
        compression="gzip",
    )


### quantile regression two-stage with bottleneck
sample_split = validation_logic()
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [128, 128, 4, 128, 128],
    "hidden_sizes2": [8],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 20,
}
data = get_data(sPath, files["W_file_22d"], vol_vars, mkt_mean_vars, regions=["USA"])
data_m = get_data(sPath, files["M_file"], vol_vars, mkt_mean_vars)
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, inputs2)
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_M_m.gzip"), compression="gzip")

### quantile regression two-stage linear
sample_split = validation_logic()
inputs1 = anomalies + vol_vars + mkt_mean_vars
inputs2 = [i + "_mean" for i in vol_vars]
param = {
    "tau": taus,
    "loss_f": "quantile_loss_two",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "input_size2": len(inputs2),
    "hidden_sizes": [],
    "hidden_sizes2": [],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "stage2_l1_lambda": 0.00001,
    "stage2_l2_lambda": 0.00001,
    "num_networks": 20,
}
data = get_data(sPath, files["W_file_22d"], vol_vars, mkt_mean_vars, regions=["USA"])
data_m = get_data(sPath, files["M_file"], vol_vars, mkt_mean_vars)
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, inputs2)
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_NN0_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_NN0_M_m.gzip"), compression="gzip")
del pred, pred_m, data, data_m


### quantile regression directly one-stage with raw returns
sample_split = validation_logic()
inputs1 = anomalies + [i + "_raw" for i in vol_vars] + mkt_mean_vars
data = get_data(sPath, files["W_file_22d"], vol_vars, mkt_mean_vars, regions=["USA"], rescale_mean=False)
data_m = get_data(sPath, files["M_file"], vol_vars, mkt_mean_vars, rescale_mean=False)

## just linear regression
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 20,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN0_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN0_M_m.gzip"), compression="gzip")

## NN with one layer
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [32],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 20,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN1_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN1_M_m.gzip"), compression="gzip")

## NN with two layers
param = {
    "tau": taus,
    "loss_f": "quantile_loss",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": len(taus),
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 20,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN2_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_onestage_NN2_M_m.gzip"), compression="gzip")

## NN2 MSE
param = {
    "loss_f": "mse",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": 1,
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.00001,
    "num_networks": 20,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r_raw", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_raw_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_raw_M_m.gzip"), compression="gzip")
del pred, pred_m


### mse forecast with cross-sectionally standardized returns
sample_split = validation_logic()
data = get_data(sPath, files["W_file_22d"], vol_vars, [], regions=["USA"], adjust_r="standardize")
data_m = get_data(sPath, files["M_file"], vol_vars, [], adjust_r="standardize")

## NN2 without volatility variables
inputs1 = anomalies
param = {
    "loss_f": "mse",
    "activation": "LeakyReLU",
    "input_size": len(inputs1),
    "hidden_sizes": [128, 128],
    "output_size": 1,
    "initial_lr": 0.0003,
    "epochs": 100,
    "batch_size": 10000,
    "dropout_rate": 0.2,
    "stage1_l1_lambda": 0.0001,
    "num_networks": 20,
}
pred, pred_m = train_loop(data, data_m, sample_split, param, inputs1, output="r", pred_type="OneStage")
pred.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_M.gzip"), compression="gzip")
pred_m.to_parquet(os.path.join(sPath, "Predict", "NN_clean_MSE_NN2_M_m.gzip"), compression="gzip")
