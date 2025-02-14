"""
Main file for the evaluation of the rating rgnn model.
used to log metrics to the MLFLOW server for later evaluation
"""

import argparse
import os
import pickle
from datetime import timedelta
from typing import Optional
import pathlib as pl

import numpy as np
import optuna
import mlflow
import pandas as pd
import torch

from trainer import Trainer
from gnn import RatingRGNN
from data import DataTransformation

from loguru import logger

# basic parser definition ----------------------------------------------------------
parser = argparse.ArgumentParser()

DATASETS = ["extraliga", "nba", "nfl", "plusliga", "premier_league", "svenska_superligan", "wimbledon"]
RATINGS = ["elo", "berrar", "pi", "no_rating"]

leagues_with_draws = ["premier_league"]

parser.add_argument("-d", "--dataset", type=str, choices=DATASETS, default="extraliga",
                    help=f"Dataset to evaluate (choices: {', '.join(DATASETS)})", required=True)

parser.add_argument("-n", "--ntrials", type=int, default=10,
                    help="Number of optuna trials", required=False)

parser.add_argument("-v", "--verbosity_level", default=0, type=int,)
# -----------------------------------------------------------------------------------

def _rating_rgnn_to_device(model, device):
    model.to(device)  # Move the model itself to the device
    for elem in model.gconv_layers:
        elem.to(device)
    for elem in model.linear_layers:
        elem.to(device)

def _print(string: str, verbosity_level: int):
    if verbosity_level == 0:
        return
    print(string)

class Evaluator:
    # fixed hyperparams:
    dropout_rate = 0.2
    rgnn = "GCONV_GRU"
    epochs = 20  # early stopping is employed, so it pretty much does not matter
    train_val_ratio = 0.9
    train_test_ratio = 0.8
    activation = "lrelu"
    normalization = "sym"

    @staticmethod
    def log_mlflow_array(name: str, array: list):
        for i, value in enumerate(array):
            mlflow.log_metric(name, value, step=i)

    @staticmethod
    def objective(trial: optuna.Trial, dataset, team_count, run_name, bidirectional_graph, target_dimension, verbosity_level):
        # rgnn fixed to be GCONV_GRU
        # rating and dataset comes from CLI args -> forms run name

        # Hyperparams for grid search optimization
        rating = trial.suggest_categorical("rating", ['elo', 'berrar', 'pi', None])
        K = trial.suggest_int("K", 2, 4) # message passing neighbourhood size
        lr_hyper = trial.suggest_float("lr_hyper", 0.0005, 0.02)
        lr_rating = trial.suggest_float("lr_rating", 0.5, 8)
        embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
        embed_dim = int(embed_dim)
        correction = trial.suggest_categorical("correction", [True, False])
        correction = bool(correction)
        discount = trial.suggest_float("discount", 0.75, 0.99)
        dense_layers = trial.suggest_int("dense_layers", 0, 3)
        conv_layers = trial.suggest_int("conv_layers", 1, 5)

        _print(f"[INFO] starting to train test validate on {run_name} run", verbosity_level)
        (snapshot_trn_acc, snapshot_val_acc, snapshot_trn_loss, snapshot_val_loss,
         train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, model, final_epoch) = (
            Evaluator.train_test_validate(dataset, team_count, lr_hyper, lr_rating, Evaluator.epochs,
                                          embed_dim, discount, correction, Evaluator.activation, K,
                                          Evaluator.rgnn, None, rating, dense_layers, conv_layers,
                                          Evaluator.dropout_rate, bidirectional_graph, target_dimension,
                                          Evaluator.normalization, Evaluator.train_test_ratio, verbosity_level,))

        # 2) LOG EVERYTHING to the MLFlow server
        # 2.a) save the artifacts (arrays), model
        os.makedirs(f"mlruns/artifacts/{run_name}", exist_ok=True)

        snapshot_train_acc_path = f"mlruns/artifacts/{run_name}/snapshot_train_acc_{run_name}_{trial.number}.npy"
        snapshot_val_acc_path = f"mlruns/artifacts/{run_name}/snapshot_val_acc_{run_name}_{trial.number}.npy"
        snapshot_train_loss_path = f"mlruns/artifacts/{run_name}/snapshot_train_loss_{run_name}_{trial.number}.npy"
        snapshot_val_loss_path = f"mlruns/artifacts/{run_name}/snapshot_val_loss_{run_name}_{trial.number}.npy"

        epoch_train_acc_path = f"mlruns/artifacts/{run_name}/epoch_train_acc_{run_name}_{trial.number}.npy"
        epoch_val_acc_path = f"mlruns/artifacts/{run_name}/epoch_val_acc_{run_name}_{trial.number}.npy"
        epoch_train_loss_path = f"mlruns/artifacts/{run_name}/epoch_train_loss_{run_name}_{trial.number}.npy"
        epoch_val_loss_path = f"mlruns/artifacts/{run_name}/epoch_val_loss_{run_name}_{trial.number}.npy"

        np.save(snapshot_train_acc_path, np.array(snapshot_trn_acc))
        np.save(snapshot_val_acc_path, np.array(snapshot_val_acc))
        np.save(snapshot_train_loss_path, np.array(snapshot_trn_loss))
        np.save(snapshot_val_loss_path, np.array(snapshot_val_loss))

        np.save(epoch_train_acc_path, np.array(train_acc))
        np.save(epoch_val_acc_path, np.array(val_acc))
        np.save(epoch_train_loss_path, np.array(train_loss))
        np.save(epoch_val_loss_path, np.array(val_loss))

        model_path = f"mlruns/artifacts/{run_name}/model_{run_name}_{trial.number}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # 2.b) log everything (including artifacts) to the MLFlow server
        with mlflow.start_run(nested=True, run_name=run_name + f"_trial_{trial.number}"):
            # Log hyperparameters to MLflow
            # 2.b.1) log even the fixed hyperparams for easier analysis
            mlflow.log_param("droupout_rate", Evaluator.dropout_rate)
            mlflow.log_param("RGNN", Evaluator.rgnn)
            mlflow.log_param("train_to_val_ratio", Evaluator.train_val_ratio)
            mlflow.log_param("train_to_test_ratio", Evaluator.train_test_ratio)
            mlflow.log_param("activation", Evaluator.activation)
            mlflow.log_param("normalization", Evaluator.normalization)

            # 2.b.2) log optimized hyperparameters
            mlflow.log_param("embed_dim", embed_dim)
            mlflow.log_param("K", K)
            mlflow.log_param("rating", rating)
            mlflow.log_param("dense_layers", dense_layers)
            mlflow.log_param("conv_layers", conv_layers)
            mlflow.log_param("lr_hyper", lr_hyper)
            mlflow.log_param("lr_rating", lr_rating)
            mlflow.log_param("correction", correction)
            mlflow.log_param("final_epoch", final_epoch)
            mlflow.log_param("discount", discount)

            # 2.b.3) Log the score to MLflow (test loss, test accuracy)
            mlflow.log_metric("test loss", test_loss)
            mlflow.log_metric("test accuracy", test_acc)

            # 2.b.4) log artifacts
            mlflow.log_artifact(snapshot_train_acc_path)
            mlflow.log_artifact(snapshot_val_acc_path)
            mlflow.log_artifact(snapshot_train_loss_path)
            mlflow.log_artifact(snapshot_val_loss_path)
            mlflow.log_artifact(epoch_train_acc_path)
            mlflow.log_artifact(epoch_val_acc_path)
            mlflow.log_artifact(epoch_train_loss_path)
            mlflow.log_artifact(epoch_val_loss_path)
            mlflow.log_artifact(model_path)

            # 2.b.5) log artifacts also as metric arrays for easier analysis within MLFlow UI
            Evaluator.log_mlflow_array("snapshot_train_acc", snapshot_trn_acc)
            Evaluator.log_mlflow_array("snapshot_val_acc", snapshot_val_acc)
            Evaluator.log_mlflow_array("snapshot_train_loss", snapshot_trn_loss)
            Evaluator.log_mlflow_array("snapshot_val_loss", snapshot_val_loss)
            Evaluator.log_mlflow_array("epoch_train_loss", train_loss)
            Evaluator.log_mlflow_array("epoch_val_loss", val_loss)
            Evaluator.log_mlflow_array("epoch_train_acc", train_acc)
            Evaluator.log_mlflow_array("epoch_val_acc", val_acc)

        # 3) Return evaluation metric
        # We use TEST LOSS as an evaluation metric
        return test_loss

    @staticmethod
    def train_test_validate(
            dataset,
            team_count: int,
            lr_hyperparams: float,
            lr_rating: float,
            epochs: int,
            embed_dim: int,
            discount: float,
            correction: bool,
            activation: str,
            K: int,
            rgnn_conv: str,
            graph_conv: Optional[str],
            rating: Optional[str],
            dense_layers: int,
            conv_layers: int,
            dropout_rate: float,
            bidirectional: bool,
            target_dimension: int,
            normalization,
            train_val_ratio: float,
            verbosity_level: int,
            **rating_kwargs,
    ) -> tuple[list[float], list[float], list[float], list[float],
                list[float], list[float], float, list[float], list[float], float,
                torch.nn.Module, int]:
        # 1) create the model --------------------------------------------------------------------------
        model = RatingRGNN(
            team_count,
            embed_dim,
            target_dim=target_dimension,
            discount=discount,
            correction=correction,
            activation=activation,
            K=K,
            rgnn_conv=rgnn_conv,
            graph_conv=graph_conv,
            rating=rating,
            dense_layers=dense_layers,
            conv_layers=conv_layers,
            dropout_rate=dropout_rate,
            normalization=normalization,
            **rating_kwargs,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _rating_rgnn_to_device(model, device)

        # 2) train and validate the model -----------------------------------------------------------------
        trainer = Trainer(
            dataset, model, lr_hyperparams, lr_rating,
            train_ratio=Evaluator.train_test_ratio, bidirectional_edges=bidirectional
        )
        snapshot_trn_acc, snapshot_val_acc, snapshot_trn_loss, snapshot_val_loss = trainer.train(
            epochs,
            train_val_ratio=train_val_ratio,
            verbosity_level=verbosity_level,
            train_on_validation=True,
            early_stopping=True,
        )

        # 3) test the model and return metrics for evaluation --------------------------------------------
        trainer.test(verbosity_level=verbosity_level)

        # MAIN OPTIMIZATION METRIC IS << test_loss >>
        return (snapshot_trn_acc, snapshot_val_acc, snapshot_trn_loss, snapshot_val_loss,
                trainer.train_loss, trainer.val_loss, trainer.test_loss,
                trainer.train_acc, trainer.val_acc, trainer.test_acc,
                trainer.model, trainer.final_epoch)

    @staticmethod
    def evaluate(raw_dataset: pd.DataFrame, n_trials: int, run_name: str, experiment_name: str, bidirectional: bool,
                 target_dimension: int, drop_draws: bool, verbosity_level: int=0):
        transform = DataTransformation(raw_dataset, snapshot_duration=timedelta(days=365))
        dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True, drop_draws=drop_draws)

        # Setup MLFlow experiment
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Set up an Optuna study
            study = optuna.create_study(direction="minimize")

            _print("[INFO] study started", verbosity_level=verbosity_level)

            study.optimize(
                lambda trial: Evaluator.objective(
                    trial, dataset, transform.num_teams, run_name, bidirectional, target_dimension, verbosity_level
                ),
                n_trials=n_trials,
            )

            # Log the best hyperparameters to MLflow
            mlflow.log_params(study.best_params)
            mlflow.log_metric(f"Best score", study.best_value)
            mlflow.log_metric("Best trial", study.best_trial.number)

if __name__ == '__main__':
    args = parser.parse_args()

    from utils import logger, setup_logger
    setup_logger(f"logs/{args.dataset}.log", int(args.verbosity_level))

    print("RatingRGNN EVALUATOR v1.0.0")
    print(f"Running with these arguments: {vars(args)}")

    _print("starting evaluation", args.verbosity_level)

    # Load dataset
    root_data_path = pl.Path(__file__).resolve().parent.parent / 'resources'
    raw_data = pd.read_csv(root_data_path / f"{args.dataset}.csv")
    raw_data["DT"] = pd.to_datetime(raw_data["DT"], format="%Y-%m-%d %H:%M:%S")
    raw_data = raw_data.sort_values(by="DT", ascending=False)

    experiment_name = "ratingRGNN"
    run_name = args.dataset
    bidirectional = args.dataset not in leagues_with_draws
    drop_draws = args.dataset not in leagues_with_draws
    target_dimension = 2 if args.dataset not in leagues_with_draws else 3


    Evaluator.evaluate(raw_data, args.ntrials, run_name, experiment_name, bidirectional,
                       target_dimension, drop_draws, int(args.verbosity_level))


