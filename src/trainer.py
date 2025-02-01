import copy
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal

import numpy as np
from loguru import logger

from .gnn import RecurrentGNN

def no_grad_when_validating(fcn):
    """
    If keyword 'validation' present and set to True we call the decorated function fcn with torch.no_grad() context
    """
    def cond_no_grad(*args, **kwargs):
        if kwargs.get("validation", False):
            with torch.no_grad():
                fcn(*args, **kwargs)
        else:
            fcn(*args, **kwargs)

    return cond_no_grad

class Trainer:
    def __init__(
            self,
            dataset: DynamicGraphTemporalSignal,
            model: RecurrentGNN,
            lr_weights: float = 0.001,
            lr_rating: float = 3.0,
            train_ratio: float = 0.8,
            bidirectional_edges: bool = False,
            output_classes_weights: Optional[Tensor] = None,
    ):
        # Dataset
        trn, tst = temporal_signal_split(dataset, train_ratio=train_ratio)
        self._train_val_dataset, self._test_dataset = trn, tst

        # Model, lr, loss fn
        self._model: nn.Module = model
        self._lr_weights: float = lr_weights
        self._lr_rating: float = lr_rating
        self._loss = CrossEntropyLoss(weight=output_classes_weights)

        # Optimizer
        self._optim = torch.optim.SGD(
            [
                {"params": model.embedding.parameters(), "lr": self._lr_rating},
            ],
            lr=self._lr_weights,
        )
        for m in model.gconv_layers:
            self._optim.add_param_group({"params": m.parameters(), "lr": self._lr_weights})
        if model.linear_layers is not None:
            for m in model.linear_layers:
                self._optim.add_param_group({"params": m.parameters(), "lr": self._lr_weights})
        if model.rating is not None:
            self._optim.add_param_group({"params": model.rating.parameters(), "lr": self._lr_weights})

        # Caching training data for MLFlow
        self._val_acc: list[float] = [] # length = number of epochs
        self._val_loss: list[float] = []
        self._train_acc: list[float] = []
        self._train_loss: list[float] = []
        self._test_acc: float = 0.
        self._test_loss: float = 0.

        # use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._verbose: bool = False
        self._bidirectional_edges: bool = bidirectional_edges

    @property
    def val_acc(self) -> list[float]:
        return self._val_acc

    @property
    def val_loss(self) -> list[float]:
        return self._val_loss

    @property
    def train_acc(self) -> list[float]:
        return self._train_acc

    @property
    def train_loss(self) -> list[float]:
        return self._train_loss

    @property
    def test_acc(self) -> float:
        return self._test_acc

    @property
    def test_loss(self) -> float:
        return self._test_loss

    def _log(self, fstring: str, level: str = 'info', *args, **kwargs):
        if not self._verbose:
            return
        if level == 'info':
            logger.info(fstring, *args, **kwargs)
        elif level == 'debug':
            logger.debug(fstring, *args, **kwargs)
        elif level == 'error':
            logger.error(fstring, *args, **kwargs)

    def _create_edge_index_and_weight(self, match, y, validation) -> tuple[Tensor, Optional[Tensor]]:
        outcome = torch.argmax(y).item()
        match = match.unsqueeze(1)

        if len(y) == 2:
            # draws not used -> bidirectional edges possible
            if outcome == 0:
                # away win
                if self._bidirectional_edges:
                    index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                    weight = torch.tensor([+1, -1], dtype=torch.float)
                else:
                    index = match
                    weight = torch.tensor([1.0], dtype=torch.float)
            else:
                # home win
                if self._bidirectional_edges:
                    index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                    weight = torch.tensor([-1, +1], dtype=torch.float)
                else:
                    index = torch.flip(match, dims=[0])
                    weight = torch.tensor([1.0], dtype=torch.float)
        else:
            # draws used
            if outcome == 0:
                # away win
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([+1, -1], dtype=torch.float)
            elif outcome == 1:
                # draw
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([0.5, 0.5], dtype=torch.float)
            else:
                # home win
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([-1, +1], dtype=torch.float)

        if validation:
            weight = None
        else:
            weight *= 0.1
            weight = weight.to(self._device)

        index = index.to(self._device)
        return index, weight

    def _resolve_y(self, home_pts, away_pts, outcomes, m):
        if self._model.rating_str == "berrar":
            y = torch.cat((away_pts.unsqueeze(0), home_pts.unsqueeze(0)), dim=0).to(self._device, torch.float)
            # rescale between 0 and 1
            max_abs_value = torch.max(torch.abs(y))
            y = y / max_abs_value
        elif self._model.rating_str == "pi":
            g_d_home = home_pts - away_pts
            g_d_away = away_pts - home_pts
            y = torch.cat((g_d_away.unsqueeze(0), g_d_home.unsqueeze(0)), dim=0).to(self._device, torch.float)
            # rescale between 0 and 1
            max_abs_value = torch.max(torch.abs(y))
            y = y / max_abs_value
        else:
            # elo, None
            y = outcomes[m, :].to(torch.float)

        return y

    @no_grad_when_validating
    def _train_validate_snapshot(self, matches, outcomes, match_points, validation=False, verbose=False):
        if validation:
            # validation part
            self._model.eval()
            self._model.store_index(False)
        else:
            # training part
            self._model.train()
            self._model.store_index(True)

        accuracy, loss_acc = 0, 0
        for m in range(matches.shape[1]):
            self._optim.zero_grad()

            home, away = match = matches[:, m]
            home_pts, away_pts = match_points[m, :]

            outcome_y = outcomes[m, :].to(torch.float)
            edge_index, edge_weight = self._create_edge_index_and_weight(match, outcome_y, validation)
            y = self._resolve_y(home_pts, away_pts, outcome_y, m)

            # forward pass
            y_hat = self._model(edge_index, home, away, edge_weight, home_pts, away_pts)

            target = torch.argmax(outcomes[m, :])
            prediction = torch.argmax(y_hat)

            accuracy += int(prediction == target)

            loss = self._loss(y_hat, y)
            loss.retains_grad_ = True
            loss_acc += loss.item()

            self._log(f"[{'VAL' if validation else 'TRN'}] Match {m}:"
                      f" correct: {'YES' if (prediction == target) else 'NO'} loss: {loss:.4f}",
                      level='debug')

            if validation:
                continue

            # Backward pass (only during training)
            loss.backward()

            self._optim.step()

        return accuracy, loss_acc

    def train(self, epochs: int, train_val_ratio: float = 0.9, verbose: bool = False,
              eval_callback: Optional[Callable] = None, train_on_validation: bool = False,
              early_stopping: bool = True, early_stopping_delta: float = 1e-3):
        self._verbose = verbose

        snapshot_trn_acc = []
        snapshot_val_acc = []
        snapshot_trn_loss = []
        snapshot_val_loss = []

        for epoch in range(epochs):
            trn_acc, trn_loss, trn_count = 0, 0, 0
            val_acc, val_loss, val_count = 0, 0, 0
            self._model.reset_index()

            snapshot_trn_acc_i = []
            snapshot_val_acc_i = []
            snapshot_trn_loss_i = []
            snapshot_val_loss_i = []

            last_model = copy.deepcopy(self._model.state_dict())

            for time, snapshot in enumerate(self._train_val_dataset):
                matches = snapshot.edge_index.to(self._device)
                outcomes = snapshot.edge_attr.to(self._device)  # edge weight encodes the match outcome
                match_points = snapshot.match_points.to(self._device)

                # split train / val data ----------------------------------------------------------
                matches_count = matches.shape[1]
                trn_size = np.ceil(matches_count * train_val_ratio).astype(int)

                matches_train = matches[:, :trn_size]
                y_train = outcomes[:trn_size, :]
                match_pts_trn = match_points[:trn_size, :]
                trn_count_i = matches_train.shape[1]
                trn_count += trn_count_i

                matches_val = matches[:, trn_size:]
                y_val = outcomes[trn_size:, :]
                match_pts_val = match_points[trn_size:, :]
                val_count_i = matches_val.shape[1]
                val_count += val_count_i
                # ----------------------------------------------------------------------------------

                # train
                trn_acc_i, trn_loss_i = self._train_validate_snapshot(matches_train, y_train, match_pts_trn,
                                                                      validation=False, verbose=verbose)
                trn_acc += trn_acc_i
                trn_loss += trn_loss_i

                # validate
                val_acc_i, val_loss_i = self._train_validate_snapshot(matches_val, y_val, match_pts_val,
                                                                      validation=True, verbose=verbose)
                val_acc += val_acc_i
                val_loss += val_loss_i

                # train on the validation subset afterward if needed
                if train_on_validation:
                    trn_acc_val_i, trn_loss_val_i = self._train_validate_snapshot(matches_val, y_val, match_pts_val,
                                                                          validation=False, verbose=verbose)
                    trn_acc += trn_acc_val_i
                    trn_loss += trn_loss_val_i
                    trn_count_i += val_count_i
                # ---------------------------------------------------

                # collect snapshot results
                snapshot_trn_acc_i.append(trn_acc_i / trn_count_i)
                snapshot_val_acc_i.append(val_acc_i / val_count_i)
                snapshot_trn_loss_i.append(trn_loss_i)
                snapshot_val_loss_i.append(val_loss_i)

            # SNAPSHOT LOOP END -----------------------------------------------------

            # collect results from whole epoch
            snapshot_trn_acc.append(snapshot_trn_acc_i)
            snapshot_val_acc.append(snapshot_val_acc_i)
            snapshot_trn_loss.append(snapshot_trn_loss_i)
            snapshot_val_loss.append(snapshot_val_loss_i)

            train_count_all = trn_count if not train_on_validation else trn_count + val_count

            self._train_acc.append(trn_acc / train_count_all)
            self._train_loss.append(trn_loss)
            self._val_acc.append(val_acc / max(val_count, 1))
            self._val_loss.append(val_loss)

            # early stopping
            if early_stopping and (len(self._val_loss) > 1) and (self._val_loss[-1] - self._val_loss[-2] > early_stopping_delta):
                self._model.load_state_dict(last_model)
                break


    def get_eval_metric(self, metric: str = "val_acc"):
        """
        possible eval metrics:
            "val_acc",
            "test_acc",
            "val_loss",
            "test_loss",
            "train_loss",
            "train_acc",
        cached from all train and test runs
        """
        assert metric in [
            "val_acc",
            "test_acc",
            "val_loss",
            "test_loss",
            "train_loss",
            "train_acc",
        ]
        return getattr(self, "_" + metric)