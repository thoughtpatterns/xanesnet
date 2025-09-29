"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy
import random
import mlflow
import torch
import time
import pickle
import numpy as np

# import optuna

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch_geometric
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from xanesnet.utils.switch import (
    OptimSwitch,
    LossSwitch,
    LRSchedulerSwitch,
    LossRegSwitch,
    KernelInitSwitch,
    BiasInitSwitch,
)

# from xanesnet.param_optuna import ParamOptuna
# from xanesnet.param_freeze import Freeze


class Learn(ABC):
    """Base class for model training procedures"""

    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.recon_flag = 0  # Set to 1 for AELearn or AEGANLearn

        # Unpack kwargs
        model_config = kwargs.get("model_config")
        hyper_params = kwargs.get("hyper_params")
        kfold_params = kwargs.get("kfold_params")
        bootstrap_params = kwargs.get("bootstrap_params")
        ensemble_params = kwargs.get("ensemble_params")
        scheduler_params = kwargs.get("scheduler_params")

        # model parameter set
        self.model_type = model_config.get("type")
        self.model_params = model_config.get("params", {})
        self.weights_params = model_config.get("weights", {})

        # hyperparameter set
        self.hyper_params = hyper_params
        self.batch_size = hyper_params.get("batch_size", 32)
        self.epochs = hyper_params.get("epochs", 100)
        self.lr = hyper_params.get("lr", 0.001)
        self.optimizer = hyper_params.get("optimizer", "adam")
        self.model_eval = hyper_params.get("model_eval", False)
        self.loss = hyper_params.get("loss", "mse")
        self.loss_reg = hyper_params.get("loss_reg", "None")
        self.loss_lambda = hyper_params.get("loss_lambda", 0.0001)
        self.seed = hyper_params.get("seed", random.randrange(1000))

        # kfold parameter set
        self.n_splits = kfold_params.get("n_splits", 3)
        self.n_repeats = kfold_params.get("n_repeats", 1)
        self.seed_kfold = kfold_params.get("seed", random.randrange(1000))

        # bootstrap parameter set
        self.n_boot = bootstrap_params.get("n_boot", 3)
        self.n_size = bootstrap_params.get("n_size", 1.0)
        self.weight_seed_boot = bootstrap_params.get(
            "weight_seed", random.sample(range(1000), 3)
        )

        # ensemble parameter set
        self.n_ens = ensemble_params.get("n_ens", 3)
        self.weight_seed_ens = ensemble_params.get(
            "weight_seed", random.sample(range(1000), 3)
        )
        # learning rate scheduler
        self.lr_scheduler = kwargs.get("lr_scheduler")
        self.scheduler_type = scheduler_params.get("type")
        self.scheduler_params = {
            k: v for k, v in scheduler_params.items() if k != "type"
        }

        # mlflow and tensorboard
        self.mlflow_flag = kwargs.get("mlflow")
        self.tb_flag = kwargs.get("tensorboard")
        self.writer = None

        # Initialise tensorboard writer with custom layout
        if self.tb_flag:
            layout = self.tensorboard_layout()
            self.writer = self.setup_writer(layout)

        # Initialise mlflow experiment
        if self.mlflow_flag:
            self.setup_mlflow()

    @abstractmethod
    def tensorboard_layout(self):
        pass

    @abstractmethod
    def train(self, model, dataset):
        pass

    @abstractmethod
    def train_std(self):
        pass

    @abstractmethod
    def train_kfold(self):
        pass

    def train_bootstrap(self):
        """
        Trains multiple models on bootstrap resamples of the provided dataset.
        """

        model_list = []
        n_samples = len(self.dataset)

        # Size of each bootstrap sample
        sample_size = int(n_samples * self.n_size)

        for i in range(self.n_boot):
            rng = np.random.default_rng(self.weight_seed_boot[i])

            # Generate all random indices at once
            bootstrap_indices = rng.choice(n_samples, size=sample_size, replace=True)

            # Create the bootstrap sample in a single, fast indexing operation
            dataset_boot = self.dataset[bootstrap_indices]

            # Deep copy model and re-initialise model weight using bootstrap seeds
            model = copy.deepcopy(self.model)
            self.weights_params["seed"] = self.weight_seed_boot[i]
            model = self._init_model_weights(model, **self.weights_params)

            # Train the model on the bootstrap sample
            model, _ = self.train(model, dataset_boot)

            model_list.append(model)

        return model_list

    def train_ensemble(self):
        """
        Train multiple models with ensemble learning
        """
        model_list = []

        for i in range(self.n_ens):
            # Deep copy model and re-initialise model weight using ensemble seeds
            model = copy.deepcopy(self.model)

            self.weights_params["seed"] = self.weight_seed_ens[i]
            model = self._init_model_weights(model, **self.weights_params)

            model, _ = self.train(model, self.dataset)

            model_list.append(model)

        return model_list

    def setup_components(self, model):
        """Initializes optimizer, loss function, and LR scheduler."""
        # --- Initialise Optimizer ---
        optim_fn = OptimSwitch().get(self.optimizer)
        optimizer = optim_fn(model.parameters(), self.lr)

        # --- Initialise loss functions ---
        criterion = LossSwitch().get(self.loss)

        # --- Regularizer ---
        regularizer = LossRegSwitch()

        # --- LR schedulers (optional) ---
        scheduler = None
        if self.lr_scheduler:
            scheduler = LRSchedulerSwitch(
                optimizer,
                self.scheduler_type,
                self.scheduler_params,
            )

        return optimizer, criterion, regularizer, scheduler

    def setup_dataloaders(self, dataset):
        """
        Splits data and creates DataLoaders.
        """
        indices = dataset.indices

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio, valid_ratio, eval_ratio = 0.75, 0.15, 0.10

            # First split: train vs (test + eval)
            train_idx, valid_idx = train_test_split(
                indices, test_size=1 - train_ratio, random_state=self.seed
            )

            # Second split: test vs eval
            test_size = eval_ratio / (eval_ratio + valid_ratio)
            valid_idx, eval_idx = train_test_split(
                valid_idx, test_size=test_size, random_state=self.seed
            )
        else:
            train_idx, valid_idx = train_test_split(
                indices, test_size=0.2, random_state=self.seed
            )

            eval_idx = None

        train_loader = self._create_loader(
            dataset[train_idx], shuffle=True, drop_last=True
        )
        valid_loader = self._create_loader(dataset[valid_idx])
        eval_loader = self._create_loader(dataset[eval_idx]) if eval_idx else None

        return train_loader, valid_loader, eval_loader

    # Helper function to create loaders
    def _create_loader(self, dataset, shuffle: bool = False, drop_last: bool = False):
        if self.model.gnn_flag:
            dataloader_cls = torch_geometric.data.DataLoader
        else:
            dataloader_cls = torch.utils.data.DataLoader

        return dataloader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )

    def setup_mlflow(self):
        experiment_name = self.model.__class__.__name__
        mlflow.set_experiment(experiment_name)

        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(self.hyper_params)
        mlflow.log_param("n_epoch", self.epochs)

    def setup_writer(self, layout: dict) -> SummaryWriter:
        # setup tensorboard stuff
        writer = SummaryWriter(f"tensorboard/{int(time.time())}")
        writer.add_custom_scalars(layout)

        return writer

    def log_mlflow(self, model):
        # Log the model as an artifact of the MLflow run.
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model", pickle_module=pickle
        )

        mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))

    def log_loss(self, name: str, value: float, epoch: int):
        # Log loss to MLflow if enabled
        if self.mlflow_flag:
            mlflow.log_metric(name, value, step=epoch)

        # Log loss to TensorBoard if enabled
        if self.tb_flag:
            self.writer.add_scalar(name, value, epoch)

    def log_close(self):
        if self.tb_flag:
            log_dir = self.writer.log_dir  # Get TensorBoard log directory
            self.writer.close()
            print(f"\nTensorBoard logs saved at: file://{Path(log_dir).resolve()}")

        if self.mlflow_flag:
            run_url = mlflow.get_artifact_uri()  # Get the MLflow run URL
            mlflow.end_run()
            print(f"\nMLflow run saved at: {run_url}")

    # TODO
    # def evaluate(self, model, loaders):
    #     """Performs final model evaluation and logs results to MLflow."""
    #     train_loader, valid_loader, eval_loader = loaders
    #     eval_test = create_eval_scheme(
    #         self.model_name,
    #         model,
    #         train_loader,
    #         valid_loader,
    #         eval_loader,
    #         self.X.shape[1],
    #         self.y.shape[1],
    #     )
    #     eval_results = eval_test.eval()
    #
    #     if self.mlflow_flag:
    #         print("Logging evaluation results to MLflow...")
    #         for k, v in eval_results.items():
    #             mlflow.log_dict(v, f"{k}.yaml")

    def _init_model_weights(self, model, **kwargs):
        """
        Initialise model weights and biases
        """
        kernel = kwargs.get("kernel", "xavier_uniform")
        bias = kwargs.get("bias", "zeros")
        seed = kwargs.get("seed", random.randrange(1000))

        # set seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

        kernel_init_fn = KernelInitSwitch().get(kernel)
        bias_init_fn = BiasInitSwitch().get(bias)

        # nested function to apply to each module
        def _init_fn(m):
            # Initialise Conv and Linear layers
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                kernel_init_fn(m.weight)
                if m.bias is not None:
                    bias_init_fn(m.bias)

        model.apply(_init_fn)

        return model
