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

import mlflow
import torch

from abc import ABC, abstractmethod
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from src.model_utils import WeightInitSwitch, weight_bias_init


class Learn(ABC):
    """
    Base class for model training

    """

    def __init__(
        self,
        model,
        x_data,
        y_data,
        hyperparams,
        kfold,
        kfoldparams,
        bootstrap_params,
        ensemble_params,
        model_name,
        schedular=None,
    ):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.hyperparams = hyperparams
        self.kfold = kfold
        self.model_name = model_name
        self.schedular = schedular

        # kfold parameter set
        self.n_splits = kfoldparams["n_splits"]
        self.n_repeats = kfoldparams["n_repeats"]
        self.seed_kfold = kfoldparams["seed"]

        # bootstrap parameter set
        self.n_boot = bootstrap_params["n_boot"]
        self.weight_seed_boot = bootstrap_params["weight_seed"]
        self.n_size = bootstrap_params["n_size"]

        # ensemble parameter set
        self.n_ens = ensemble_params["n_ens"]
        self.weight_seed_ens = ensemble_params["weight_seed"]

        self.writer = None
        self.score = None
        self.train_loader = None
        self.valid_loader = None
        self.eval_loader = None
        self.weight_seed = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_writer()

        # setup mlflow experiment info
        exp_name = f"{model_name}"
        self.exp_time = f"run_{datetime.today()}"
        try:
            self.exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        except:
            self.exp_id = mlflow.create_experiment(exp_name)

    def setup_dataloader(self):
        # split dataset and setup train/valid/test dataloader
        x_data = torch.from_numpy(self.x_data)
        y_data = torch.from_numpy(self.y_data)

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio = 0.75
            test_ratio = 0.15
            eval_ratio = 0.10

            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=1 - train_ratio, random_state=42
            )

            x_test, x_eval, y_test, y_eval = train_test_split(
                x_test, y_test, test_size=eval_ratio / (eval_ratio + test_ratio)
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42
            )

        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_set = torch.utils.data.TensorDataset(x_test, y_test)
        self.valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if self.model_eval:
            eval_set = torch.utils.data.TensorDataset(x_eval, y_eval)
            self.eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            self.eval_loader = None

    def setup_weight(self):
        torch.cuda.manual_seed(
            self.weight_seed
        ) if torch.cuda.is_available() else torch.manual_seed(self.weight_seed)

        # Initialise model weight & bias
        kernel_init = WeightInitSwitch().fn(self.kernel)
        bias_init = WeightInitSwitch().fn(self.bias)

        self.model.apply(
            lambda m: weight_bias_init(
                m=m, kernel_init_fn=kernel_init, bias_init_fn=bias_init
            )
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_kfold(self):
        pass

    @abstractmethod
    def train_bootstrap(self):
        pass

    @abstractmethod
    def train_ensemble(self):
        pass
