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

import os
import pickle
import tempfile
import random
import mlflow
import time
import mlflow.pytorch
import numpy as np
import torch

from sklearn.model_selection import RepeatedKFold
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from .base_learn import Learn
from src.creator import create_eval_scheme
from src.model_utils import (
    loss_reg_fn,
    WeightInitSwitch,
    OptimSwitch,
    LossSwitch,
    weight_bias_init,
)


class NNLearn(Learn):
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
        label=None,
        schedular=None,
    ):
        # Call the constructor of the parent class
        super().__init__(
            model,
            x_data,
            y_data,
            hyperparams,
            kfold,
            kfoldparams,
            bootstrap_params,
            ensemble_params,
            label,
            schedular,
        )

        # hyperparameter set
        self.loss_fn = hyperparams["loss"]["loss_fn"]
        self.loss_args = hyperparams["loss"]["loss_args"]
        self.loss_reg_type = hyperparams["loss"]["loss_reg_type"]
        self.lambda_reg = hyperparams["loss"]["loss_reg_param"]

        self.n_epoch = hyperparams["epochs"]
        self.kernel = hyperparams["kernel_init"]
        self.bias = hyperparams["bias_init"]
        self.optim_fn = hyperparams["optim_fn"]
        self.batch_size = hyperparams["batch_size"]
        self.lr = hyperparams["lr"]
        self.model_eval = hyperparams["model_eval"]
        self.weight_seed_hyper = hyperparams["weight_seed"]
        self.seed = hyperparams["seed"]

        self.weight_seed = self.weight_seed_hyper

    def setup_writer(self):
        # setup tensorboard stuff
        layout = {
            "Multi": {
                "loss": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        self.writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
        self.writer.add_custom_scalars(layout)

    def train(self):
        # Move model to the available device
        self.model.to(self.device)
        self.setup_dataloader()
        self.setup_weight

        # Initialise optimizer using specified optimization function and learning rate
        optim_fn = OptimSwitch().fn(self.optim_fn)
        optimizer = optim_fn(self.model.parameters(), lr=self.lr)

        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        with mlflow.start_run(experiment_id=self.exp_id, run_name=self.exp_time):
            mlflow.log_params(self.hyperparams)
            mlflow.log_param("n_epoch", self.n_epoch)

            for epoch in range(self.n_epoch):
                print(f">>> epoch = {epoch}")
                self.model.train()
                running_loss = 0
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs, labels = inputs.float(), labels.float()

                    optimizer.zero_grad()
                    logps = self.model(inputs)

                    loss = criterion(logps, labels)

                    if self.loss_reg_type is not None:
                        l_reg = loss_reg_fn(self.model, self.loss_reg_type, self.device)
                        loss += self.lambda_reg * l_reg

                    loss.mean().backward()
                    optimizer.step()

                    running_loss += loss.item()

                valid_loss = 0
                self.model.eval()

                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs, labels = inputs.float(), labels.float()

                    target = self.model(inputs)

                    loss = criterion(target, labels)
                    valid_loss += loss.item()

                if self.schedular is not None:
                    before_lr = optimizer.param_groups[0]["lr"]
                    self.schedular.step()
                    after_lr = optimizer.param_groups[0]["lr"]
                    print(
                        "Epoch %d: Adam lr %.5f -> %.5f" % (epoch, before_lr, after_lr)
                    )

                print("Training loss:", running_loss / len(self.train_loader))
                print("Validation loss:", valid_loss / len(self.valid_loader))

                self.log_scalar(
                    "loss/train", (running_loss / len(self.train_loader)), epoch
                )
                self.log_scalar(
                    "loss/validation", (valid_loss / len(self.valid_loader)), epoch
                )

            self.write_log()

        # Perform model evaluation using invariance tests
        if self.model_eval:
            eval_test = create_eval_scheme(
                self.model_name,
                self.model,
                self.train_loader,
                self.valid_loader,
                self.eval_loader,
                self.x_data[0].size,
                self.y_data.shape[1],
            )

            eval_results = eval_test.eval()

            # Log results
            for k, v in eval_results.items():
                mlflow.log_dict(v, f"{k}.yaml")

        summary(self.model)
        self.writer.close()
        self.score = running_loss / len(self.train_loader)

        return self.model

    def train_kfold(self):
        # K-fold Cross Validation model evaluation
        prev_score = 1e6
        fit_time = []
        train_score = []
        test_score = []

        kfold_spooler = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        for fold, (train_index, test_index) in enumerate(
            kfold_spooler.split(self.x_data)
        ):
            print(">> fitting neural net...")
            start = time.time()

            model = self.train()
            train_score.append(self.score)
            fit_time.append(time.time() - start)
            # Testing
            model.eval()
            x_test = torch.from_numpy(self.x_data[test_index]).float().to(self.device)
            pred_xanes = model(x_test)
            pred_score = criterion(
                torch.tensor(self.y_data[test_index]).to(self.device), pred_xanes
            ).item()
            test_score.append(pred_score)

            if pred_score < prev_score:
                best_model = model
            prev_score = pred_score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_score": test_score,
        }

        self._print_kfold_result(result)

        return best_model

    def train_bootstrap(self):
        model_list = []
        for i in range(self.n_boot):
            self.weight_seed = self.weight_seed_boot[i]
            random.seed(self.weight_seed)

            new_x = []
            new_y = []

            for _ in range(int(self.x_data.shape[0] * self.n_size)):
                idx = random.randint(0, self.x_data.shape[0] - 1)
                new_x.append(self.x_data[idx])
                new_y.append(self.y_data[idx])

                self.x_data = np.asarray(new_x)
                self.y_data = np.asarray(new_y)

            if self.kfold:
                model = self.train_kfold()
            else:
                model = self.train()

            model_list.append(model)

        return model_list

    def train_ensemble(self):
        model_list = []
        for i in range(self.n_ens):
            self.weight_seed = self.weight_seed_ens[i]

            if self.kfold:
                model = self.train_kfold()
            else:
                model = self.train()

            model_list.append(model)

        return model_list

    def _print_kfold_result(self, scores: dict):
        # prints a summary table of the scores from k-fold cross validation;
        # summarises the elapsed time and train/test metric scores for each k-fold
        # with overall k-fold cross validation statistics (mean and std. dev.)
        # using the `scores` dictionary returned from `cross_validate`

        print("")
        print(">> summarising scores from k-fold cross validation...")
        print("")

        print("*" * 16 * 3)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test"))
        print("*" * 16 * 3)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}"
        for kf, (t, train, test) in enumerate(
            zip(scores["fit_time"], scores["train_score"], scores["test_score"])
        ):
            print(fmt.format(kf, t, np.absolute(train), np.absolute(test)))

        print("*" * 16 * 3)
        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("std. dev.", *stdevs_))

        print("*" * 16 * 3)

        print("")

    def log_scalar(self, name, value, epoch):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, epoch)
        mlflow.log_metric(name, value)

    def write_log(self):
        # # Create a SummaryWriter to write TensorBoard events locally
        output_dir = tempfile.mkdtemp()

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(
            self.model, artifact_path="pytorch-model", pickle_module=pickle
        )
        print(
            "\nThe model is logged at:\n%s"
            % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )

        mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))
