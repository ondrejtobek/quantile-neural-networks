import pandas as pd
import numpy as np
import os
from numba import njit
import xarray

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from scipy.interpolate import BSpline, splrep
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tqdm import tqdm
from scores.probability import crps_cdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN_quantile_regression(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate, activation="LeakyReLU"):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            setattr(self, "fc" + str(i), layer)  # add label to the layer to be able to add hooks
            self.layers.append(layer)
            if i != len(layer_sizes) - 1:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i]))
                if activation == "Sigmoid":
                    act = nn.Sigmoid()
                elif activation == "ReLU":
                    act = nn.ReLU()
                elif activation == "LeakyReLU":
                    act = nn.LeakyReLU()
                setattr(self, "fc_act" + str(i), act)
                self.layers.append(act)
                if layer_sizes[i] > 8:  # don't apply dropout for bottleneck layers
                    self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NN_quantile_regression_two(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        input_size2,
        hidden_sizes2,
        output_size,
        dropout_rate,
        activation="LeakyReLU",
    ):
        super().__init__()
        self.input_size = input_size
        self.input_size2 = input_size2

        if activation == "Sigmoid":
            act = nn.Sigmoid()
        elif activation == "ReLU":
            act = nn.ReLU()
        elif activation == "LeakyReLU":
            act = nn.LeakyReLU()

        ## first group of variable to fit quantiles of standardized distribution
        self.net1 = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.grp1_param = []
        for i in range(1, len(layer_sizes)):
            # linear layer
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            setattr(self, "fc" + str(i), layer)  # add label to the layer to be able to add hooks
            self.net1.append(layer)
            self.grp1_param += ["fc" + str(i)]
            # for all layers except the last one
            if i != len(layer_sizes) - 1:
                # batch norm
                layer = nn.BatchNorm1d(layer_sizes[i])
                setattr(self, "bn" + str(i), layer)  # add label to the layer to be able to add hooks
                self.net1.append(layer)
                self.grp1_param += ["bn" + str(i)]
                # activation
                self.net1.append(act)
                # dropout
                if layer_sizes[i] > 8:  # don't apply dropout for bottleneck layers
                    self.net1.append(nn.Dropout(dropout_rate))

        ## second group to fit raw returns
        self.net2 = nn.ModuleList()
        layer_sizes2 = [input_size2] + hidden_sizes2 + [1]
        self.grp2_param = []
        for i in range(1, len(layer_sizes2)):
            # linear layer
            layer = nn.Linear(layer_sizes2[i - 1], layer_sizes2[i])
            setattr(self, "fc2_" + str(i), layer)  # add label to the layer to be able to add hooks
            self.net2.append(layer)
            self.grp2_param += ["fc2_" + str(i)]
            # for all layers except the last one
            if i != len(layer_sizes2) - 1:
                # batch norm
                layer = nn.BatchNorm1d(layer_sizes2[i])
                setattr(self, "bn2_" + str(i), layer)  # add label to the layer to be able to add hooks
                self.net2.append(layer)
                self.grp2_param += ["bn2_" + str(i)]
                # activation
                self.net2.append(act)
                # dropout
                self.net2.append(nn.Dropout(dropout_rate))
            else:
                # initialize weights at 0 for the last layer
                torch.nn.init.zeros_(getattr(self, "fc2_" + str(i)).weight)
                torch.nn.init.ones_(getattr(self, "fc2_" + str(i)).bias)

    def forward(self, x):
        x1 = x[:, : self.input_size]
        for layer in self.net1:
            x1 = layer(x1)
        scaler = x[:, self.input_size + self.input_size2].unsqueeze(1)
        x1_scaled = x1 * scaler
        x1 = torch.where(x1_scaled < -1.0, -1.0 / scaler.repeat(1, x1.shape[1]), x1)  # clip at -100% return
        x2 = x[:, self.input_size : self.input_size + self.input_size2]
        for layer in self.net2:
            x2 = layer(x2)
        x2 = x1_scaled * x2
        x2 = torch.where(x2 < -1.0, -1.0, x2)  # clip at -100% return
        return torch.cat((x1.unsqueeze(2), (x2).unsqueeze(2)), 2)


class NN_quantile_regression_three(NN_quantile_regression_two):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        input_size2,
        hidden_sizes2,
        output_size,
        dropout_rate,
        stage3_bias=False,
        activation="LeakyReLU",
    ):
        super().__init__(
            input_size, hidden_sizes, input_size2, hidden_sizes2, output_size, dropout_rate, activation
        )

        ## third group to fit mse of raw returns
        self.net3 = nn.ModuleList()
        self.grp3_param = ["fc3_1"]
        layer = nn.Linear(output_size, 1, bias=stage3_bias)
        setattr(self, "fc3_1", layer)  # add label to the layer to be able to add hooks
        self.net3.append(layer)

    def forward(self, x):
        x1 = x[:, : self.input_size]
        for layer in self.net1:
            x1 = layer(x1)
        scaler = x[:, self.input_size + self.input_size2].unsqueeze(1)
        x1_scaled = x1 * scaler
        x1 = torch.where(x1_scaled < -1.0, -1.0 / scaler.repeat(1, x1.shape[1]), x1)  # clip at -100% return
        x2 = x[:, self.input_size : self.input_size + self.input_size2]
        for layer in self.net2:
            x2 = layer(x2)
        x2 = x1_scaled * x2
        x2 = torch.where(x2 < -1.0, -1.0, x2)  # clip at -100% return
        x3 = x2
        for layer in self.net3:
            x3 = layer(x3)
        return torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), 2), x3


class QuantileRegressionDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]
        return x, y


class RegressionNNEnsemble:
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        initial_lr,
        dropout_rate,
        num_networks,
        epochs,
        batch_size,
        input_size2=None,
        hidden_sizes2=None,
        tau: list = [0.5],
        momentum: tuple = (0.9, 0.999),
        loss_f: str = "quantile_loss",
        stage1_l1_lambda: float = 0.0001,
        stage2_l1_lambda: float = 0.00001,
        stage2_l2_lambda: float = 0.00001,
        stage3_l1_lambda: float = 0.00001,
        stage3_l2_lambda: float = 0.00001,
        l2_lambda: float = 0.0,
        seed: int = 42,
        epoch_size: int = 1500000,
        epoch_resize_f: float = 1.0,
        activation: str = "LeakyReLU",
        early_stopping: bool = True,
        early_stopping_patience: int = 2,
        early_stopping_validation_size: float = 0.2,
        decay_factor: float = 1.0,
        decay_step_size: int = 100,
        loss1_w: float = 1.0,
        loss2_w: float = 1.0,
        loss3_w: float = 0.05,
        stage3_bias: bool = False,
        huber_loss_param: float = 1.0,
        load_best_state: bool = True,
        filter_worst_forecast: bool = False,
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.input_size2 = input_size2
        self.hidden_sizes2 = hidden_sizes2
        self.output_size = output_size
        self.tau = tau
        self.num_networks = num_networks
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epoch_resize_f = epoch_resize_f
        self.stage1_l1_lambda = stage1_l1_lambda
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_validation_size = early_stopping_validation_size
        self.seed = seed
        self.initial_lr = initial_lr
        self.momentum = momentum
        self.decay_step_size = decay_step_size
        self.decay_factor = decay_factor
        self.loss_f = loss_f
        self.loss1_w = loss1_w
        self.loss2_w = loss2_w
        self.loss3_w = loss3_w
        self.stage2_l1_lambda = stage2_l1_lambda
        self.stage2_l2_lambda = stage2_l2_lambda
        self.stage3_l1_lambda = stage3_l1_lambda
        self.stage3_l2_lambda = stage3_l2_lambda
        self.load_best_state = load_best_state
        self.filter_worst_forecast = filter_worst_forecast

        if loss_f == "quantile_loss_two":
            self.models = [
                NN_quantile_regression_two(
                    input_size,
                    hidden_sizes,
                    input_size2,
                    hidden_sizes2,
                    output_size,
                    dropout_rate,
                    activation,
                )
                for _ in range(num_networks)
            ]
        elif loss_f == "quantile_loss_three":
            self.models = [
                NN_quantile_regression_three(
                    input_size,
                    hidden_sizes,
                    input_size2,
                    hidden_sizes2,
                    output_size,
                    dropout_rate,
                    stage3_bias,
                    activation,
                )
                for _ in range(num_networks)
            ]
        else:
            self.models = [
                NN_quantile_regression(input_size, hidden_sizes, output_size, dropout_rate, activation)
                for _ in range(num_networks)
            ]

        tau = torch.tensor(tau).to(device)
        if loss_f == "quantile_loss":
            self.loss_fn = self.quantile_loss(tau, loss1_w)
        elif loss_f == "huber_quantile_loss":
            self.loss_fn = self.quantile_huber_loss(tau, huber_loss_param, loss1_w)
        elif loss_f == "mse":
            self.loss_fn = self.MSE_loss(loss1_w)
        elif loss_f == "huber":
            self.loss_fn = self.huber_loss(huber_loss_param, loss1_w)
        elif loss_f == "quantile_loss_two":
            self.loss_fn = self.quantile_loss_two(tau, loss1_w, loss2_w)
        elif loss_f == "quantile_loss_three":
            self.loss_fn = self.quantile_loss_three(tau, loss1_w, loss2_w, loss3_w)
        else:
            ValueError("Loss function not supported.")

        self.define_optimizers()

    def quantile_loss(self, quantile, loss1_w: float = 1.0):
        def loss(y_pred, y_true):
            e = y_true.unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred
            return loss1_w * torch.where(e > 0, quantile * e, (quantile - 1) * e).mean()

        return loss

    def quantile_huber_loss(self, quantile, delta: float = 1.0, loss1_w: float = 1.0):
        def loss(y_pred, y_true):
            e = y_true.unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred
            abs_e = torch.abs(e)
            condition = abs_e <= delta
            squared_loss = 0.5 * e**2
            linear_loss = delta * (abs_e - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return loss1_w * torch.where(e > 0, quantile * loss, (1 - quantile) * loss).mean()

        return loss

    def MSE_loss(self, loss1_w: float = 1.0):
        def loss(y_pred, y_true):
            e = y_true.unsqueeze(1) - y_pred
            return loss1_w * (0.5 * e**2).mean()

        return loss

    def huber_loss(self, delta: float = 1.0, loss1_w: float = 1.0):
        def loss(y_pred, y_true):
            e = y_true.unsqueeze(1) - y_pred
            abs_e = torch.abs(e)
            condition = abs_e <= delta
            squared_loss = 0.5 * e**2
            linear_loss = delta * (abs_e - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return loss1_w * loss.mean()

        return loss

    def quantile_loss_two(self, quantile, loss1_w: float = 1.0, loss2_w: float = 1.0):
        def loss(y_pred, y_true):
            e1 = y_true[:, 0].unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred[:, :, 0]
            loss1 = torch.where(e1 > 0, quantile * e1, (quantile - 1) * e1).mean()
            e2 = y_true[:, 1].unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred[:, :, 1]
            loss2 = torch.where(e2 > 0, quantile * e2, (quantile - 1) * e2).mean()
            return loss1_w * loss1 + loss2_w * loss2

        return loss

    def quantile_loss_three(self, quantile, loss1_w: float = 1.0, loss2_w: float = 1.0, loss3_w: float = 0.05):
        def loss(y_pred, y_true):
            e1 = y_true[:, 0].unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred[0][:, :, 0]
            loss1 = torch.where(e1 > 0, quantile * e1, (quantile - 1) * e1).mean()
            e2 = y_true[:, 1].unsqueeze(1).repeat(1, quantile.shape[0]) - y_pred[0][:, :, 1]
            loss2 = torch.where(e2 > 0, quantile * e2, (quantile - 1) * e2).mean()
            e3 = y_true[:, 1].unsqueeze(1) - y_pred[1]
            loss3 = (0.5 * e3**2).mean()
            # scale down third loss so that it has smaller impact on previous layers
            return loss1_w * loss1 + loss2_w * loss2 + loss3_w * loss3

        return loss

    def define_optimizers(self):
        self.optimizers = [
            optim.Adam(model.parameters(), lr=self.initial_lr, betas=self.momentum) for model in self.models
        ]
        self.schedulers = [
            StepLR(optimizer, step_size=self.decay_step_size, gamma=self.decay_factor)
            for optimizer in self.optimizers
        ]

    def finetuning_init(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.define_optimizers()

    def fit(self, X, y):
        model_number = 1
        self.best_loss = []
        for model, optimizer, scheduler in zip(self.models, self.optimizers, self.schedulers):
            if self.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.early_stopping_validation_size, random_state=self.seed + model_number
                )
                val_dataset = QuantileRegressionDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            else:
                X_train, y_train = X, y

            train_dataset = QuantileRegressionDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            # rescale number of epochs based on sample size
            n_obs = X_train.shape[0]
            resize_f = self.epoch_resize_f * (self.epoch_size / n_obs - 1)
            batches_in_epoch = int(np.ceil(n_obs * (1 + resize_f) / self.batch_size))
            rescaled_epochs = int(np.ceil(self.epochs * (1 + resize_f)))

            model.to(device)
            model.train()

            best_loss = float("inf")
            best_model_state = None
            patience_counter = 0
            batch_counter = 0
            epoch_loss = 0
            epoch_obs = 0
            epoch_res = 0
            stop_early = False

            for epoch in range(rescaled_epochs):
                for _, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_fn(output, target)
                    l1_penalty = self._apply_l1_l2_penalization(model)
                    loss += l1_penalty
                    loss.backward()
                    optimizer.step()
                    batch_counter += 1
                    epoch_loss += loss.item() * data.shape[0]
                    epoch_obs += data.shape[0]
                    if (batch_counter % batches_in_epoch) == 0:
                        msg = f"Network number: {model_number}, Epoch: {epoch_res+1}/{self.epochs}"
                        train_loss = epoch_loss / epoch_obs
                        msg = msg + f", Train Loss: {train_loss:.5f}"
                        epoch_loss = 0
                        epoch_obs = 0
                        epoch_res += 1
                        if self.early_stopping:
                            # Evaluate the validation loss
                            val_loss = self._evaluate_validation_loss(val_loader, model)
                            if val_loss + 0.00001 < best_loss:
                                best_loss = val_loss
                                best_model_state = model.state_dict()
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                scheduler.step()  # Update learning rate
                            if patience_counter >= self.early_stopping_patience:
                                e = self.early_stopping_patience
                                print(f"Early stopping triggered. No improvement in {e} epochs.")
                                stop_early = True
                            msg = msg + f", Valid Loss: {val_loss:.5f}"
                        else:
                            scheduler.step()
                        print(msg)
                    if stop_early:
                        break
                if stop_early:
                    break

            if self.early_stopping:
                if self.load_best_state:
                    # Load the best model state
                    model.load_state_dict(best_model_state)
                    self.best_loss += [best_loss]
                else:
                    self.best_loss += [val_loss]

            model_number = model_number + 1

    def _evaluate_validation_loss(self, validation_loader, model):
        total_loss = 0.0
        num_samples = 0

        model.eval()
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item() * data.shape[0]
                num_samples += data.shape[0]
        model.train()

        average_loss = total_loss / num_samples
        return average_loss

    def _apply_l1_l2_penalization(self, model):
        loss_penalty = torch.tensor(0, dtype=torch.float32).to(device)
        # penalize just the first layer with l1 to do the selection
        loss_penalty += torch.norm(model.fc1.weight, 1) * self.stage1_l1_lambda
        if hasattr(model, "fc2_1"):
            loss_penalty += torch.norm(model.fc2_1.weight, 1) * self.stage2_l1_lambda
            loss_penalty += torch.norm(model.fc2_1.weight, 2) * self.stage2_l2_lambda
        if hasattr(model, "fc3_1"):
            loss_penalty += torch.abs(1 - model.fc3_1.weight.sum()) * self.stage3_l1_lambda
            loss_penalty += torch.norm(model.fc3_1.weight, 2) * self.stage3_l2_lambda
        for name, param in model.named_parameters():
            if name == "weight":
                loss_penalty += torch.norm(param, 2) * self.l2_lambda
        return loss_penalty

    def predict(self, X):
        if self.loss_f == "quantile_loss_two":
            return self._predict_two(X)
        if self.loss_f == "quantile_loss_three":
            return self._predict_three(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        # predictions for one stage approach
        test_data = torch.tensor(X.values, dtype=torch.float32).to(device)

        pred = np.empty((test_data.shape[0], len(self.tau), len(self.models)))
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred[:, :, i] = model(test_data).cpu().numpy()
        # discard predictions with the worst validation set fit
        if self.early_stopping and self.filter_worst_forecast:
            bl = np.array(self.best_loss)
            best_loss_cut = bl <= np.median(bl) + np.mean(np.abs(bl - np.median(bl)))
            pred = pred[:, :, best_loss_cut]
        if self.loss_f == "mse":
            output_cols = ["pred"]
        else:
            output_cols = [f"pred_{i}" for i in self.tau]
        return pd.DataFrame(pred.mean(axis=2), columns=output_cols)

    def _predict_two(self, X):
        # predictions for two stage approach
        test_data = torch.tensor(X.values, dtype=torch.float32).to(device)

        pred = np.empty((test_data.shape[0], len(self.tau), 2, len(self.models)))
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred[:, :, :, i] = model(test_data).cpu().numpy()
        # discard predictions with the worst validation set fit
        if self.early_stopping and self.filter_worst_forecast:
            bl = np.array(self.best_loss)
            best_loss_cut = bl <= np.median(bl) + np.mean(np.abs(bl - np.median(bl)))
            pred = pred[:, :, :, best_loss_cut]
        pred_norm = pd.DataFrame(pred[:, :, 0, :].mean(axis=2), columns=[f"pred_{i}" for i in self.tau])
        pred_raw = pd.DataFrame(pred[:, :, 1, :].mean(axis=2), columns=[f"pred_raw_{i}" for i in self.tau])
        return pd.concat((pred_norm, pred_raw), axis=1)

    def _predict_three(self, X):
        # predictions for two stage approach
        test_data = torch.tensor(X.values, dtype=torch.float32).to(device)

        pred = np.empty((test_data.shape[0], len(self.tau), 2, len(self.models)))
        pred2 = np.empty((test_data.shape[0], 1, len(self.models)))
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred_ = model(test_data)
                pred[:, :, :, i] = pred_[0].cpu().numpy()
                pred2[:, :, i] = pred_[1].cpu().numpy()
        # discard predictions with the worst validation set fit
        if self.early_stopping and self.filter_worst_forecast:
            bl = np.array(self.best_loss)
            best_loss_cut = bl <= np.median(bl) + np.mean(np.abs(bl - np.median(bl)))
            pred = pred[:, :, :, best_loss_cut]
            pred2 = pred2[:, :, best_loss_cut]
        pred_norm = pd.DataFrame(pred[:, :, 0, :].mean(axis=2), columns=[f"pred_{i}" for i in self.tau])
        pred_raw = pd.DataFrame(pred[:, :, 1, :].mean(axis=2), columns=[f"pred_raw_{i}" for i in self.tau])
        pred_mse = pd.DataFrame(pred2.mean(axis=2), columns=["pred_mse"])
        return pd.concat((pred_norm, pred_raw, pred_mse), axis=1)


def train_loop(
    data,
    data_m,
    sample_split,
    param,
    inputs1,
    inputs2=None,
    output=["r", "r_raw"],
    param_finetune={},
    finetune=True,
    pred_type="TwoStage",
    pred_file=["M", "W"],
    train_regions=[],
    output_bottleneck_act=False,
    output_forecast_wgts=False,
):
    if pred_type == "TwoStage":
        input_vars = inputs1 + inputs2 + ["r_scale"]
    else:
        input_vars = inputs1

    # function to attach hooks and get activations
    if output_bottleneck_act:
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

    extra_cols = [i for i in ["r", "r_raw", "r_scale"] if i in data.columns]

    preds = []
    preds_m = []
    bottleneck_acts = []
    forecast_wgts = []
    train_from_scratch = True
    for i in tqdm(range(len(sample_split))):
        split = sample_split.iloc[i].to_dict()

        # training sample
        train_df = data.loc[(data["date"] >= split["train_start"]) & (data["date"] <= split["train_end"]), :]
        if len(train_regions) > 0:
            train_df = train_df.loc[train_df["region"].isin(train_regions)]

        if train_from_scratch or not finetune:
            model = RegressionNNEnsemble(**param)
            model.fit(train_df[input_vars], train_df[output])
            train_from_scratch = False

        # finetune even in the first period as it can fix bad fit for some runs
        if finetune:
            model.finetuning_init(**param_finetune)
            model.fit(train_df[input_vars], train_df[output])

        # predict weekly
        if "W" in pred_file:
            test_df = data.loc[
                (data["date"] >= split["valid_start"]) & (data["date"] <= split["valid_end"]), :
            ].reset_index(drop=True)
            pred = model.predict(test_df[input_vars])
            for col in pred.columns:
                test_df[col] = pred[col]
            preds += [test_df[["DTID", "date"] + extra_cols + list(pred.columns)]]

        # predict monthly
        if "M" in pred_file:
            # get bottleneck activations
            if output_bottleneck_act:
                layers = param["hidden_sizes"]
                b_loc = np.argmin(layers) + 1
                b_n = min(layers)
                for i, model_ in enumerate(model.models):
                    getattr(model_, f"fc{b_loc}").register_forward_hook(get_activation(f"fc{b_loc}_{i}"))

            test_df_m = data_m.loc[
                (data_m["date"] >= split["valid_start"]) & (data_m["date"] <= split["valid_end"]), :
            ].reset_index(drop=True)

            pred = model.predict(test_df_m[input_vars])
            for col in pred.columns:
                test_df_m[col] = pred[col]
            preds_m += [test_df_m[["DTID", "date"] + extra_cols + list(pred.columns)]]

            if output_bottleneck_act:
                act = []
                for i in range(param["num_networks"]):
                    _act = activation[f"fc{b_loc}_{i}"].cpu().numpy()
                    _act = pd.DataFrame(_act, columns=[f"act{i}_{j+1}" for j in range(b_n)])
                    act += [_act]
                act = pd.concat(act, axis=1)
                act["DTID"] = test_df_m["DTID"]
                act["date"] = test_df_m["date"]
                bottleneck_acts += [act]

        # get weights for third stage forecast combining quantiles into mse forecast
        if output_forecast_wgts:
            for i, model_ in enumerate(model.models):
                if hasattr(model_, "fc3_1"):
                    for _name, _par in model_.fc3_1.named_parameters():
                        if _name == "weight":
                            par = _par.detach().cpu().numpy()
                    res_ = pd.DataFrame(par, columns=[f"p{i}" for i in param["tau"]], index=[0])
                    res_["model"] = i
                    res_["valid_start"] = split["valid_start"]
                    forecast_wgts += [res_]

    if len(preds) > 0:
        preds = pd.concat(preds).reset_index(drop=True)
    if len(preds_m) > 0:
        preds_m = pd.concat(preds_m).reset_index(drop=True)
    res = preds, preds_m
    if len(forecast_wgts) > 0:
        forecast_wgts = pd.concat(forecast_wgts).reset_index(drop=True)
        res += (forecast_wgts,)
    if len(bottleneck_acts) > 0:
        bottleneck_acts = pd.concat(bottleneck_acts).reset_index(drop=True)
        res += (bottleneck_acts,)
    return res


# define validation logic
def validation_logic(first_year: int = 1995, last_year: int = 2023, year_step: int = 1, r_horizon: str = "M"):
    sample_split = []
    for year in range(first_year, last_year + 1, year_step):
        valid_start = f"{year}-01-01"
        # need to shift training sample end otherwise there would be leakage of returns into predition sample
        if r_horizon == "M":  # M -> 22-d
            train_end = f"{year-1}-11-29"
        elif r_horizon == "Q":  # for quarterly
            train_end = f"{year-1}-09-30"
        elif r_horizon == "Y":  # for annual
            train_end = f"{year-2}-12-31"
        else:
            raise ValueError("Data frequency not supported.")
        sample_split += [
            pd.DataFrame(
                {
                    "train_start": "1973-01-01",
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": f"{year+year_step-1}-12-31",
                },
                index=[valid_start],
            )
        ]
    sample_split = pd.concat(sample_split)
    return sample_split


def get_data(
    sPath,
    feature_file,
    vol_vars,
    mkt_mean_vars,
    regions=[],
    rescale_mean=True,
    adjust_r="divide",
    r_scale=0.11,
):
    ## load data with features
    data = pd.read_parquet(os.path.join(sPath, "Features", feature_file))
    if len(regions) > 0:
        data = data.loc[data["region"].isin(regions)].copy()

    # rescale volatility variables that are not standardized
    data.reset_index(drop=True, inplace=True)
    for Var in vol_vars:
        data.loc[data[Var].isnull(), Var] = data.groupby(["date", "region"])[Var].transform("mean")
        data[Var + "_raw"] = data[Var] / 0.022
        data[Var] = data[Var] / data.groupby(["date", "region"])[Var].transform("mean")
    data["r_scale"] = data.groupby(["date", "region"])["EWMAVol6_raw"].transform("mean") * r_scale

    # normalize returns
    data["r_raw"] = data["r"]
    if adjust_r == "divide":
        data["r"] = data["r"] / data["r_scale"]
    elif adjust_r == "standardize":
        data["r"] = data["r"] - data.groupby(["date", "region"])["r"].transform("median")
        data["r_scale"] = data.groupby(["date", "region"])["r"].transform(lambda x: np.abs(x).mean()) / 1.4
        data["r"] = data["r"] / data["r_scale"]
    data = data.copy()

    # create cross-sectional mean volatility variables
    for Var in vol_vars:
        data[Var + "_mean"] = data.groupby(["date", "region"])[Var + "_raw"].transform("mean")

    # add historical average market returns
    MktMean = pd.read_parquet(
        os.path.join(sPath, "Features", "Mkt_mean.gzip"), columns=["date", "region"] + mkt_mean_vars
    )
    data["date"] = data["date"].astype("datetime64[ns]")  # got messed up date format with some package version...
    MktMean["date"] = MktMean["date"].astype("datetime64[ns]")
    MktMean.sort_values("date", inplace=True)
    data.sort_values("date", inplace=True)
    data = pd.merge_asof(data, MktMean, on="date", by="region")
    data.reset_index(drop=True, inplace=True)
    if rescale_mean:
        for Var in mkt_mean_vars:
            data[Var] = data[Var] / data["r_scale"]

    return data


def get_anomalies_list(sPath):
    return pd.read_excel(os.path.join(sPath, "Inputs", "AnomaliesMeta.xlsx"))["name_sc"].to_list()


@njit
def Integrate(x, y):
    m0, m1, m2, m3, m4 = 0, 0, 0, 0, 0
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        y1 = y[i]
        y2 = y[i + 1]
        b = (y2 - y1) / (x2 - x1)
        a = y1 - b * x1
        m0 += a * x2 + 1 / 2 * b * x2**2 - a * x1 - 1 / 2 * b * x1**2
        m1 += 1 / 2 * a * x2**2 + 1 / 3 * b * x2**3 - 1 / 2 * a * x1**2 - 1 / 3 * b * x1**3
        m2 += 1 / 3 * a * x2**3 + 1 / 4 * b * x2**4 - 1 / 3 * a * x1**3 - 1 / 4 * b * x1**4
        m3 += 1 / 4 * a * x2**4 + 1 / 5 * b * x2**5 - 1 / 4 * a * x1**4 - 1 / 5 * b * x1**5
        m4 += 1 / 5 * a * x2**5 + 1 / 6 * b * x2**6 - 1 / 5 * a * x1**5 - 1 / 6 * b * x1**6
    return m0, m1, m2, m3, m4


def DensityIntegration(x, y, grid_point_n: int = 100):
    # parameters
    min_density = 1e-5
    eps = 1e-4
    LinearFlag = False

    try:

        # reshape x to 1D
        if x.shape[0] == 1:
            x = x[0, :]

        # get limits to x
        x_min = x[1]
        x_max = x[-2]

        # cut quantiles in lower tail and keep only the highest quantile with -1 return
        cut_tail = 0
        for i in range(x.shape[0] - 1):
            if x[i + 1] == -1:
                cut_tail += 1
        x = x[cut_tail:]
        y = y[cut_tail:]

        # get 90th and 10th quantiles
        if y.min() < 0.05:
            x10 = x[y == 0.1]
        else:
            x10 = x[2]
        x90 = x[y == 0.9]

        # sort out possible inconsistencies
        for i in range(x.shape[0] - 1):
            if x[i + 1] < x[i] + eps:
                x[i + 1] = x[i] + eps

        # create grid of denser x values
        xnew = []
        for i in range(x.shape[0] - 3):
            x_l = x[i + 1]
            x_u = x[i + 2]
            xnew += [np.arange(x_l, x_u, (x_u - x_l) / grid_point_n)]
        xnew = np.concatenate(xnew)
        # xnew = np.arange(x_min, x_max, eps)
        xnew = np.unique(xnew)
        xnew.sort()

        # try cubic interpolation
        tck = splrep(x, y, k=3)
        yder = BSpline(*tck)(xnew, 1)

        # handle cases when the density is messed up
        if yder[(xnew >= x10) & (xnew <= x90)].min() < min_density:
            tck = splrep(x, y, k=1)
            yder = BSpline(*tck)(xnew, 1)
            LinearFlag = True

        # enforce minimum density
        yder[yder < min_density] = min_density

        # enforce monotonicity in the tails
        min_val = yder[(xnew <= x10)].min()
        min_loc = np.where((yder == min_val) & (xnew <= x10))[-1][0]
        yder[:min_loc] = min_val
        min_val = yder[(xnew >= x90)].min()
        min_loc = np.where((yder == min_val) & (xnew >= x90))[0][0]
        yder[min_loc:] = min_val

        # truncate at -1 if needed
        if xnew.min() < -1:
            yder = yder[xnew >= -1]
            xnew = xnew[xnew >= -1]
            x_min = -1

        # compute discrete probabilities at tails
        P_l = BSpline(*tck)(x_min)
        P_u = 1 - BSpline(*tck)(x_max)

        ## integrate
        m0, m1, m2, m3, m4 = Integrate(xnew, yder)

        # add discrete probabilities in the tails
        m0 += P_l + P_u
        m1 += P_l * x_min + P_u * x_max
        m2 += P_l * x_min**2 + P_u * x_max**2
        m3 += P_l * x_min**3 + P_u * x_max**3
        m4 += P_l * x_min**4 + P_u * x_max**4
        # normalize probability to 1
        m1 /= m0
        m2 /= m0
        m3 /= m0
        m4 /= m0

        res = pd.DataFrame(
            {"m0": m0, "m1": m1, "m2": m2, "m3": m3, "m4": m4, "LinearFlag": LinearFlag, "Error": False},
            index=[0],
        )

    except:

        res = pd.DataFrame(
            {
                "m0": np.nan,
                "m1": np.nan,
                "m2": np.nan,
                "m3": np.nan,
                "m4": np.nan,
                "LinearFlag": False,
                "Error": True,
            },
            index=[0],
        )

    return res


def ComputeMoments(dt, taus, grid_point_n: int = 100, pred_col="pred_raw"):
    y = np.array(taus)
    cols = [f"{pred_col}_{tau}" for tau in taus]
    res = dt.groupby(["date", "DTID"]).apply(
        lambda df: DensityIntegration(df.iloc[0][cols].values.astype(float), y, grid_point_n)
    )
    res = res.reset_index(-1, drop=True).reset_index()
    dt = dt.merge(res, on=["date", "DTID"], how="left")

    # compute central moments
    dt["var"] = dt["m2"] - dt["m1"] ** 2
    dt["std"] = np.sqrt(dt["var"])
    dt["skew"] = (dt["m3"] - 3 * dt["m1"] * dt["var"] - dt["m1"] ** 3) / dt["var"] ** (3 / 2)
    dt["kurtosis"] = (dt["m4"] - 4 * dt["m1"] * dt["m3"] + 6 * dt["m1"] ** 2 * dt["m2"] - 3 * dt["m1"] ** 4) / dt[
        "var"
    ] ** 2

    return dt


def AdjustMoments(res):
    res["e_k"] = res["kurtosis"] - 3
    res["var_adj"] = res["var"] * (1.00232494 - 0.0021291 * res["skew"] + 0.00219741 * res["e_k"])
    res["skew_adj"] = 0.99495005 * res["skew"] + 0.02609106 * res["skew"] ** 2 + 0.01066851 * res["e_k"]
    res["kurtosis_adj"] = 3 + 1.41849094 * res["e_k"] + 0.04657378 * res["e_k"] ** 2 - 0.73949787 * res["skew"]
    del res["e_k"]
    return res


def mean_quantile_loss(y_true, y_pred, sample_weight=None, alpha=0.5):
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff

    output_errors = np.average(loss, weights=sample_weight, axis=0)
    return np.average(output_errors)


def MaxDD(x):
    y = (1 + x).cumprod()
    RollMax = y.cummax()
    DailyDrawdown = y / RollMax - 1.0
    MaxDailyDrawdown = DailyDrawdown.cummin().iloc[-1]
    return MaxDailyDrawdown


def NW_std(x, lags: int = 12):
    """Compute standard errors with Newey-West adjustment

    Args:
        x (np.array): array with returns
        lags (int, optional): number of lags. Defaults to 12.
    """

    x = x[~np.isnan(x)]
    t = len(x)
    if t < max(12, lags + 2):
        return np.nan
    else:
        x_mean = x.mean()
        h = x - x_mean
        x_sd = (h**2).sum()
        if lags > 0:
            for z in range(1, lags + 1):
                x_sd += 2 * (1 - z / (lags + 1)) * (h[1 : (t - z)] * h[(z + 1) : t]).sum()
        x_sd = np.sqrt(x_sd / t)

        return x_sd


def GetPredictions(
    sPath, files, pred_file=None, horizon="M", full_sample: bool = True, keep_r: bool = False, region: list = []
):
    # add region for data file
    AddCols = ["DTID", "date", "region"]
    if horizon == "W":
        dt = pd.read_parquet(os.path.join(sPath, "Features", files["W_file_22d_full"]), columns=AddCols)
        dt_R = pd.read_parquet(os.path.join(sPath, "Features", files["W_file_22d"]), columns=AddCols)
        dt = pd.concat([dt, dt_R]).drop_duplicates(subset=["DTID", "date"])
    else:
        dt = pd.read_parquet(os.path.join(sPath, "Features", files["M_file_full"]), columns=AddCols)
        dt_R = pd.read_parquet(os.path.join(sPath, "Features", files["M_file"]), columns=AddCols)
        dt = pd.concat([dt, dt_R]).drop_duplicates(subset=["DTID", "date"])

    pred_type = "_full" if full_sample else ""
    if pred_file is None:
        pred_file = "NN_clean_M" + pred_type + "_m.gzip"
    pred = pd.read_parquet(os.path.join(sPath, "Predict", pred_file))
    dt = pred.merge(dt, on=["DTID", "date"], how="left")

    if not keep_r:
        dt.drop([i for i in pred.columns if i in ["r", "r_raw"]], axis=1, inplace=True)

    if len(region) > 0:
        dt = dt.loc[dt["region"].isin(region)].copy()

    return dt


def CreatePredSignalSimplified(dt, taus: list, Q: float = 0.1):
    dt["mean_v"] = 0
    for i in range(len(taus) - 1):
        dt["mean_v"] = dt["mean_v"] + (dt[f"pred_{taus[i+1]}"] + dt[f"pred_{taus[i]}"]) / 2 * (
            taus[i + 1] - taus[i]
        )
    dt["Var"] = dt["mean_v"]
    dt = dt[["date", "DTID", "region", "Var"]].copy()
    dt["long"] = dt["Var"] >= dt.groupby(["region", "date"])["Var"].transform("quantile", 1 - Q)
    dt["short"] = dt["Var"] <= dt.groupby(["region", "date"])["Var"].transform("quantile", Q)
    dt = dt.loc[dt["long"] | dt["short"], ["date", "DTID", "long", "short"]].dropna()
    return dt


def CreatePredSignal(dt, PredVar: str = "pred", Q: float = 0.1):
    dt["Var"] = dt[PredVar]
    dt = dt[["date", "DTID", "region", "Var"]].copy()
    dt["long"] = dt["Var"] >= dt.groupby(["region", "date"])["Var"].transform("quantile", 1 - Q)
    dt["short"] = dt["Var"] <= dt.groupby(["region", "date"])["Var"].transform("quantile", Q)
    dt = dt.loc[dt["long"] | dt["short"], ["date", "DTID", "long", "short"]].dropna()
    return dt


def CreatePredSignalSorts(dt, Var1: str, Var2=None, sort_n: int = 5, sort_type: str = "dependent"):
    if "Var" in dt.columns:
        dt.drop("Var", axis=1, inplace=True)
    GrpVars = ["region", "date"]
    if Var2 is None:
        for i in range(sort_n):
            dt.loc[
                (
                    (dt[Var1] >= dt.groupby(GrpVars)[Var1].transform("quantile", i / sort_n))
                    & (dt[Var1] <= dt.groupby(GrpVars)[Var1].transform("quantile", (i + 1) / sort_n))
                ),
                "Var",
            ] = f"Var1_{i}"
    elif sort_type == "independent":
        for i in range(sort_n):
            for j in range(sort_n):
                dt.loc[
                    (
                        (dt[Var1] >= dt.groupby(GrpVars)[Var1].transform("quantile", i / sort_n))
                        & (dt[Var1] <= dt.groupby(GrpVars)[Var1].transform("quantile", (i + 1) / sort_n))
                        & (dt[Var2] >= dt.groupby(GrpVars)[Var2].transform("quantile", j / sort_n))
                        & (dt[Var2] <= dt.groupby(GrpVars)[Var2].transform("quantile", (j + 1) / sort_n))
                    ),
                    "Var",
                ] = f"Var1_{i}_Var2_{j}"
    elif sort_type == "dependent":
        for i in range(sort_n):
            dt1 = dt.loc[
                (
                    (dt[Var1] >= dt.groupby(GrpVars)[Var1].transform("quantile", i / sort_n))
                    & (dt[Var1] <= dt.groupby(GrpVars)[Var1].transform("quantile", (i + 1) / sort_n))
                )
            ].copy()
            for j in range(sort_n):
                dt2 = dt1.loc[
                    (
                        (dt1[Var2] >= dt1.groupby(GrpVars)[Var2].transform("quantile", j / sort_n))
                        & (dt1[Var2] <= dt1.groupby(GrpVars)[Var2].transform("quantile", (j + 1) / sort_n))
                    )
                ].copy()
                dt2 = dt2[["date", "DTID", "region"]].copy()
                dt2["sort"] = f"Var1_{i}_Var2_{j}"
                dt = dt.merge(dt2, on=["date", "DTID", "region"], how="left")
                dt.loc[dt["sort"].notnull(), "Var"] = dt["sort"]
                del dt["sort"]
    else:
        raise ValueError("sort_type not supported")

    dt = dt[["date", "DTID", "Var"]].dropna()
    return dt


def ConstructPortfolios(
    dt, sPath, ret_type: str = "M", wgt_type: str = "EW", port_type: str = "LS", max_date=None
):
    # base the calculations on monthly or daily returns
    Cols = ["DTID", "date", "r", "region", "MC"]
    if ret_type == "M":
        ret = pd.read_parquet(os.path.join(sPath, "Features", "Monthly_ret.gzip"), columns=Cols)
    elif ret_type == "D":
        ret = pd.read_parquet(os.path.join(sPath, "Features", "Daily_ret.gzip"), columns=Cols)
    else:
        ValueError("Frequency not supported")

    if max_date is not None:
        ret = ret.loc[ret["date"] <= max_date].copy()

    ## merge portfolio signal on returns
    # create maping of dates for returns to rebalancing periods
    ret["date"] = ret["date"].astype("datetime64[ns]")
    dt["date"] = dt["date"].astype("datetime64[ns]")
    dates = ret[["date"]].drop_duplicates().sort_values("date")
    dates2 = dt[["date"]].drop_duplicates().sort_values("date")
    dates2["pred_date"] = dates2["date"]
    dates = pd.merge_asof(dates, dates2, on="date")

    # add returns
    dt = dt.rename({"date": "pred_date"}, axis=1).merge(dates, on="pred_date")
    dt = dt.merge(ret[["date", "DTID", "r"]], on=["date", "DTID"], how="left")
    dt["r"] = dt["r"].fillna(0)

    # add region and MC based on the first observation within the rebalancing period
    ret = ret.merge(dates, on="date")
    ret = ret.sort_values(["DTID", "date"]).groupby(["DTID", "pred_date"])[["region", "MC"]].first().reset_index()
    dt = dt.merge(ret, on=["pred_date", "DTID"], how="inner")

    # pred vars based on portfolio construction method
    if port_type == "LS":
        pred_vars = ["long", "short"]
    elif port_type == "sorts":
        pred_vars = ["Var"]

    # select weighting
    dt = dt.sort_values(["DTID", "date"])
    dt["first_obs"] = dt.groupby(["DTID", "pred_date"] + pred_vars)["date"].transform("first")
    if wgt_type == "EW":
        dt.loc[dt["first_obs"] == dt["date"], "wgt"] = 1
    elif wgt_type == "VW":
        dt.loc[dt["first_obs"] == dt["date"], "wgt"] = dt["MC"]
    else:
        ValueError("Weighting not supported")
    dt["wgt"] = dt["wgt"] / dt.groupby(["pred_date", "region"] + pred_vars)["wgt"].transform("sum")

    # fill the next time periods for returns within one rebalancing period
    max_rep = dt.groupby(["DTID", "pred_date"])["r"].count().max()
    for _ in range(max_rep - 1):
        dt.loc[dt["wgt"].isnull(), "wgt"] = (1 + dt.groupby("DTID")["r"].shift(1)) * dt.groupby("DTID")[
            "wgt"
        ].shift(1)
    dt["r_wgt"] = dt["r"] * dt["wgt"]

    # compute portfolio returns
    if port_type == "LS":
        long_r = dt.query("long == True").groupby(["region", "date"])["r_wgt"].sum()
        short_r = dt.query("short == True").groupby(["region", "date"])["r_wgt"].sum()
        long_r_global = long_r.groupby(["date"]).mean().reset_index()
        short_r_global = short_r.groupby(["date"]).mean().reset_index()
        long_r_global["region"] = "Global"
        short_r_global["region"] = "Global"
        long_r_global = long_r_global.set_index(["region", "date"])["r_wgt"]
        short_r_global = short_r_global.set_index(["region", "date"])["r_wgt"]
        long_r = pd.concat([long_r, long_r_global])
        short_r = pd.concat([short_r, short_r_global])
        ls_r = long_r - short_r
        return [long_r, short_r, ls_r]
    elif port_type == "sorts":
        long_r = dt.groupby(["region", "date"] + pred_vars)["r_wgt"].sum()
        return long_r


def PortfolioMetrics(long_r, short_r, ls_r, ret_type: str = "M"):
    if ret_type == "M":
        SR_scale = 12
    elif ret_type == "D":
        SR_scale = 260
    else:
        ValueError("Frequency not supported")

    res = pd.DataFrame(
        {
            "ls_mean": ls_r.groupby("region").mean() * 100,
            "ls_std": ls_r.groupby("region").apply(lambda x: NW_std(x.values)) * 100,
            "ls_SR": ls_r.groupby("region").mean() / ls_r.groupby("region").std() * np.sqrt(SR_scale),
            "ls_max_DD": ls_r.groupby("region").apply(lambda x: MaxDD(x) * 100),
            "long_mean": long_r.groupby("region").mean() * 100,
            "long_std": long_r.groupby("region").apply(lambda x: NW_std(x.values)) * 100,
            "long_SR": long_r.groupby("region").mean() / long_r.groupby("region").std() * np.sqrt(SR_scale),
            "long_max_DD": long_r.groupby("region").apply(lambda x: MaxDD(x) * 100),
            "short_mean": short_r.groupby("region").mean() * 100,
            "short_std": short_r.groupby("region").apply(lambda x: NW_std(x.values)) * 100,
        },
    ).reset_index()
    return res


def PortfolioMetricsClean(long_r, short_r, ls_r, ret_type: str = "M"):
    if ret_type == "M":
        SR_scale = 12
    elif ret_type == "D":
        SR_scale = 260
    else:
        ValueError("Frequency not supported")

    res = pd.DataFrame(
        {
            "ls_mean": ls_r.groupby("region").mean() * 100,
            "ls_t": ls_r.groupby("region").apply(lambda x: x.mean() / NW_std(x.values) * np.sqrt(x.count())),
            "ls_SR": ls_r.groupby("region").mean() / ls_r.groupby("region").std() * np.sqrt(SR_scale),
            "long_mean": long_r.groupby("region").mean() * 100,
            "long_t": long_r.groupby("region").apply(lambda x: x.mean() / NW_std(x.values) * np.sqrt(x.count())),
            "long_SR": long_r.groupby("region").mean() / long_r.groupby("region").std() * np.sqrt(SR_scale),
            "short_mean": short_r.groupby("region").mean() * 100,
            "short_t": short_r.groupby("region").apply(lambda x: x.mean() / NW_std(x.values) * np.sqrt(x.count())),
            "short_SR": short_r.groupby("region").mean() / short_r.groupby("region").std() * np.sqrt(SR_scale),
        },
    ).reset_index()

    res["ls"] = res["ls_mean"].apply(lambda x: f"{x:.2f}") + " (" + res["ls_t"].apply(lambda x: f"{x:.2f}") + ")"
    res["long"] = (
        res["long_mean"].apply(lambda x: f"{x:.2f}") + " (" + res["long_t"].apply(lambda x: f"{x:.2f}") + ")"
    )
    res["short"] = (
        res["short_mean"].apply(lambda x: f"{x:.2f}") + " (" + res["short_t"].apply(lambda x: f"{x:.2f}") + ")"
    )

    return res


def PortfolioMetricsCleanLS(ls_r, ret_type: str = "M"):
    if ret_type == "M":
        SR_scale = 12
    elif ret_type == "D":
        SR_scale = 260
    else:
        ValueError("Frequency not supported")

    res = pd.DataFrame(
        {
            "ls_mean": ls_r.groupby("region").mean() * 100,
            "ls_t": ls_r.groupby("region").apply(lambda x: x.mean() / NW_std(x.values) * np.sqrt(x.count())),
            "ls_SR": ls_r.groupby("region").mean() / ls_r.groupby("region").std() * np.sqrt(SR_scale),
        },
    ).reset_index()

    res["ls"] = res["ls_mean"].apply(lambda x: f"{x:.2f}") + " (" + res["ls_t"].apply(lambda x: f"{x:.2f}") + ")"

    return res


def PortfolioMetricsDoubleSorts(long_r, ret_type: str = "M", sort_n=5):
    if ret_type == "M":
        SR_scale = 12
    elif ret_type == "D":
        SR_scale = 260
    else:
        ValueError("Frequency not supported")

    # add results for LS portfolios
    ret = long_r.reset_index().set_index(["region", "date"])
    minq, maxq = 0, sort_n - 1
    AddPorts = []
    for i in range(minq, maxq + 1):
        Var1 = f"Var1_{i}_Var2_{minq}"
        Var2 = f"Var1_{i}_Var2_{maxq}"
        ret_ = ret.loc[ret["Var"] == Var2, "r_wgt"] - ret.loc[ret["Var"] == Var1, "r_wgt"]
        ret_ = ret_.reset_index()
        ret_["Var"] = f"Var1_{i}_Var2_{maxq+1}"
        AddPorts += [ret_]
        Var1 = f"Var1_{minq}_Var2_{i}"
        Var2 = f"Var1_{maxq}_Var2_{i}"
        ret_ = ret.loc[ret["Var"] == Var2, "r_wgt"] - ret.loc[ret["Var"] == Var1, "r_wgt"]
        ret_ = ret_.reset_index()
        ret_["Var"] = f"Var1_{maxq+1}_Var2_{i}"
        AddPorts += [ret_]
    AddPorts = pd.concat(AddPorts).set_index(["region", "date", "Var"])["r_wgt"]
    ret = pd.concat([long_r, AddPorts], axis=0)

    # compute portfoio statistics
    GrpVars = ["region", "Var"]
    res = pd.DataFrame(
        {
            "mean": ret.groupby(GrpVars).mean() * 100,
            "std": ret.groupby(GrpVars).apply(lambda x: NW_std(x.values) / np.sqrt(x.count())) * 100,
            "SR": ret.groupby(GrpVars).mean() / ret.groupby(GrpVars).std() * np.sqrt(SR_scale),
        },
    ).reset_index()

    # split back to individual sorting variables
    res["Var1"] = res["Var"].apply(lambda x: x.split("_")[1])
    if res["Var"].apply(lambda x: len(x.split("_"))).max() == 4:
        res["Var2"] = res["Var"].apply(lambda x: x.split("_")[3])

    # extra variables
    res["tstat"] = res["mean"] / res["std"]
    res["out"] = res["mean"].apply(lambda x: f"{x:.2f}") + " (" + res["tstat"].apply(lambda x: f"{x:.2f}") + ")"

    # format columns names
    res["Var1"] = "Q" + (res["Var1"].astype(int) + 1).astype(str)
    res["Var2"] = "Q" + (res["Var2"].astype(int) + 1).astype(str)
    res.loc[res["Var1"] == f"Q{maxq+2}", "Var1"] = f"Q{maxq+1}-Q1"
    res.loc[res["Var2"] == f"Q{maxq+2}", "Var2"] = f"Q{maxq+1}-Q1"

    return res


def PortfolioMetricsSorts(ret, ret_type: str = "M", sort_n=10):
    if ret_type == "M":
        SR_scale = 12
    elif ret_type == "D":
        SR_scale = 260
    else:
        ValueError("Frequency not supported")

    # add results for LS portfolios
    ret_ = ret.reset_index().set_index(["region", "date"])
    minq, maxq = 0, sort_n - 1
    Var1 = f"Var1_{minq}"
    Var2 = f"Var1_{maxq}"
    ret_ = ret_.loc[ret_["Var"] == Var2, "r_wgt"] - ret_.loc[ret_["Var"] == Var1, "r_wgt"]
    ret_ = ret_.reset_index()
    ret_["Var"] = f"Var1_{maxq+1}"
    ret_ = ret_.set_index(["region", "date", "Var"])["r_wgt"]
    ret = pd.concat([ret, ret_], axis=0)

    # compute portfoio statistics
    GrpVars = ["region", "Var"]
    res = pd.DataFrame(
        {
            "mean": ret.groupby(GrpVars).mean() * 100,
            "std": ret.groupby(GrpVars).apply(lambda x: NW_std(x.values) / np.sqrt(x.count())) * 100,
            "SR": ret.groupby(GrpVars).mean() / ret.groupby(GrpVars).std() * np.sqrt(SR_scale),
        },
    ).reset_index()

    # split back to individual sorting variables
    res["Var1"] = res["Var"].apply(lambda x: x.split("_")[1])
    if res["Var"].apply(lambda x: len(x.split("_"))).max() == 4:
        res["Var2"] = res["Var"].apply(lambda x: x.split("_")[3])

    # format columns names
    res["Var1_num"] = res["Var1"].astype(int) + 1
    res = res.sort_values("Var1_num")
    res["Var1"] = res["Var1_num"].astype(str)
    res.loc[res["Var1"] == f"{maxq+2}", "Var1"] = f"{maxq+1}-1"

    # extra variables
    res["tstat"] = res["mean"] / res["std"]
    res["out"] = res["mean"].apply(lambda x: f"{x:.2f}") + " (" + res["tstat"].apply(lambda x: f"{x:.2f}") + ")"

    return res


def ToLaTeX(df):
    first = True
    for col in range(df.shape[1]):
        if first:
            first = False
            out = df.iloc[:, col].astype(str)
        else:
            out = out + " & " + df.iloc[:, col].astype(str)
    out = out + "\\\\"
    return out


def ToLaTeX_list(l):
    return " & ".join(l) + "\\"


def Rsqrd(x, x_hat, rf, const=False):
    if const:
        return 1 - ((x - x_hat) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    else:
        return 1 - ((x - x_hat) ** 2).sum() / (x**2).sum()


def Rsqrd_df(dt, pred, actual="r", const=False):
    out = dt.groupby("region").apply(lambda x: Rsqrd(x[actual], x[pred], const))
    out_global = Rsqrd(dt[actual], dt[pred], const)
    out.loc["Global"] = out_global
    return out


def RMSE_df(dt, pred, actual):
    def RMSE(x, y):
        return np.sqrt(((x - y) ** 2).mean())

    out = dt.groupby("region").apply(lambda x: RMSE(x[actual], x[pred]))
    out_global = RMSE(dt[actual], dt[pred])
    out.loc["Global"] = out_global
    return out


def DieboldMariano_df(dt, pred1, pred2, actual, AddGlobal=False):
    def MSE(x, y):
        return ((x - y) ** 2).mean()

    out = pd.DataFrame(
        {
            "MSE1": dt.groupby(["region", "date"]).apply(lambda x: MSE(x[actual], x[pred1])),
            "MSE2": dt.groupby(["region", "date"]).apply(lambda x: MSE(x[actual], x[pred2])),
        }
    ).reset_index()
    if AddGlobal:
        out_global = pd.DataFrame(
            {
                "MSE1": dt.groupby(["date"]).apply(lambda x: MSE(x[actual], x[pred1])),
                "MSE2": dt.groupby(["date"]).apply(lambda x: MSE(x[actual], x[pred2])),
                "region": "Global",
            }
        ).reset_index()
        out = pd.concat([out, out_global])

    out["diff"] = out["MSE2"] - out["MSE1"]
    tstat = (
        out.groupby("region")["diff"].mean()
        / out.groupby("region")["diff"].apply(lambda x: NW_std(x.values))
        * np.sqrt(out.groupby("region")["diff"].count())
    )

    return tstat


def MAD_df(dt, pred, actual):
    def MAD(x, y):
        return (np.abs(x - y)).mean()

    out = dt.groupby("region").apply(lambda x: MAD(x[actual], x[pred]))
    out_global = MAD(dt[actual], dt[pred])
    out.loc["Global"] = out_global
    return out


def DensityIntegrationPlots(x, y, grid_point_n: int = 100):
    # parameters
    min_density = 1e-5
    eps = 1e-4

    # reshape x to 1D
    if x.shape[0] == 1:
        x = x[0, :]

    # cut quantiles in lower tail and keep only the highest quantile with -1 return
    cut_tail = 0
    for i in range(x.shape[0] - 1):
        if x[i + 1] == -1:
            cut_tail += 1
    x = x[cut_tail:]
    y = y[cut_tail:]

    # get 90th and 10th quantiles
    if y.min() < 0.05:
        x10 = x[y == 0.1]
    else:
        x10 = x[2]
    x90 = x[y == 0.9]

    # sort out possible inconsistencies
    for i in range(x.shape[0] - 1):
        if x[i + 1] < x[i] + eps:
            x[i + 1] = x[i] + eps

    # create grid of denser x values
    xnew = []
    for i in range(x.shape[0] - 3):
        x_l = x[i + 1]
        x_u = x[i + 2]
        xnew += [np.arange(x_l, x_u, (x_u - x_l) / grid_point_n)]
    xnew = np.concatenate(xnew)
    xnew = np.unique(xnew)
    xnew.sort()

    # try cubic interpolation
    tck = splrep(x, y, k=3)
    yder = BSpline(*tck)(xnew, 1)

    # handle cases when the density is messed up
    if yder[(xnew >= x10) & (xnew <= x90)].min() < min_density:
        tck = splrep(x, y, k=1)
        yder = BSpline(*tck)(xnew, 1)
    ynew = BSpline(*tck)(xnew)

    # enforce minimum density
    yder[yder < min_density] = min_density

    # enforce monotonicity in the tails
    min_val = yder[(xnew <= x10)].min()
    min_loc = np.where((yder == min_val) & (xnew <= x10))[-1][0]
    yder[:min_loc] = min_val
    min_val = yder[(xnew >= x90)].min()
    min_loc = np.where((yder == min_val) & (xnew >= x90))[0][0]
    yder[min_loc:] = min_val

    # truncate at -1 if needed
    if xnew.min() < -1:
        yder = yder[xnew >= -1]
        ynew = ynew[xnew >= -1]
        xnew = xnew[xnew >= -1]

    res = pd.DataFrame({"x": xnew, "cdf": ynew, "density": yder})

    return res


def DistScoring(x, y, ret_act, grid_point_n: int = 100):
    # parameters
    min_density = 1e-5
    eps = 1e-4
    LinearFlag = False

    try:

        # reshape x to 1D
        if x.shape[0] == 1:
            x = x[0, :]

        # cut quantiles in lower tail and keep only the highest quantile with -1 return
        cut_tail = 0
        for i in range(x.shape[0] - 1):
            if x[i + 1] == -1:
                cut_tail += 1
        x = x[cut_tail:]
        y = y[cut_tail:]

        # get 90th and 10th quantiles
        if y.min() < 0.05:
            x10 = x[y == 0.1]
        else:
            x10 = x[2]
        x90 = x[y == 0.9]

        # sort out possible inconsistencies
        for i in range(x.shape[0] - 1):
            if x[i + 1] < x[i] + eps:
                x[i + 1] = x[i] + eps

        # create grid of denser x values
        xnew = []
        for i in range(x.shape[0] - 3):
            x_l = x[i + 1]
            x_u = x[i + 2]
            xnew += [np.arange(x_l, x_u, (x_u - x_l) / grid_point_n)]
        xnew = np.concatenate(xnew)
        # xnew = np.arange(x_min, x_max, eps)
        xnew = np.unique(xnew)
        xnew.sort()

        # try cubic interpolation
        tck = splrep(x, y, k=3)
        yder = BSpline(*tck)(xnew, 1)

        # handle cases when the density is messed up
        if yder[(xnew >= x10) & (xnew <= x90)].min() < min_density:
            tck = splrep(x, y, k=1)
            LinearFlag = True
        yder = BSpline(*tck)(xnew, 0)

        # enforce monotonicity in the tails
        min_val = yder[(xnew <= x10)].min()
        min_loc = np.where((yder == min_val) & (xnew <= x10))[-1][0]
        yder[:min_loc] = min_val
        max_val = yder[(xnew >= x90)].max()
        max_loc = np.where((yder == max_val) & (xnew >= x90))[0][0]
        yder[max_loc:] = max_val

        # truncate at -1 if needed
        if xnew.min() < -1:
            yder = yder[xnew >= -1]
            xnew = xnew[xnew >= -1]

        obs_array = xarray.DataArray(ret_act)
        fcst_array = xarray.DataArray(coords={"cdf": xnew}, data=yder)
        score = crps_cdf(fcst_array, obs_array, threshold_dim="cdf").total.values

        res = pd.DataFrame(
            {"score": score, "LinearFlag": LinearFlag, "Error": False},
            index=[0],
        )

    except:

        res = pd.DataFrame(
            {
                "score": np.nan,
                "LinearFlag": False,
                "Error": True,
            },
            index=[0],
        )

    return res


def ComputeScoring(dt, taus, grid_point_n: int = 100):
    y = np.array(taus)
    if "pred_raw_0.5" in dt.columns:
        cols = [f"pred_raw_{tau}" for tau in taus]
    else:
        cols = [f"pred_{tau}" for tau in taus]
    res = dt.groupby(["date", "DTID"]).apply(
        lambda df: DistScoring(df.iloc[0][cols].values.astype(float), y, df.iloc[0]["r"], grid_point_n)
    )
    res = res.reset_index(-1, drop=True).reset_index()

    return res


def GetMoments(x, y):
    dt = DensityIntegration(x, y)
    dt["var"] = dt["m2"] - dt["m1"] ** 2
    dt["std"] = np.sqrt(dt["var"])
    dt["skew"] = (dt["m3"] - 3 * dt["m1"] * dt["var"] - dt["m1"] ** 3) / dt["var"] ** (3 / 2)
    dt["kurtosis"] = (dt["m4"] - 4 * dt["m1"] * dt["m3"] + 6 * dt["m1"] ** 2 * dt["m2"] - 3 * dt["m1"] ** 4) / dt[
        "var"
    ] ** 2
    return dt
