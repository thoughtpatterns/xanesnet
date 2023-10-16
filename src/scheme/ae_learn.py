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
import time
import torch
import mlflow.pytorch
import numpy as np

from sklearn.model_selection import RepeatedKFold
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from .nn_learn import NNLearn
from src.creator import create_eval_scheme
from src.model_utils import OptimSwitch, LossSwitch


class AELearn(NNLearn):
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
                "recon_loss": [
                    "Multiline",
                    ["recon_loss/train", "recon_loss/validation"],
                ],
                "pred_loss": ["Multiline", ["pred_loss/train", "pred_loss/validation"]],
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
                loss_r = 0
                loss_p = 0

                total_step_train = 0
                for inputs, labels in self.train_loader:
                    inputs, labels = (
                        inputs.to(self.device),
                        labels.to(self.device),
                    )
                    inputs, labels = (
                        inputs.float(),
                        labels.float(),
                    )

                    optimizer.zero_grad()

                    recon_input, outputs = self.model(inputs)

                    loss_recon = criterion(recon_input, inputs)
                    loss_pred = criterion(outputs, labels)

                    loss = loss_recon + loss_pred
                    loss.backward()

                    optimizer.step()
                    running_loss += loss.mean().item()
                    loss_r += loss_recon.item()
                    loss_p += loss_pred.item()

                    total_step_train += 1

                valid_loss = 0
                valid_loss_r = 0
                valid_loss_p = 0
                self.model.eval()
                total_step_valid = 0

                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs, labels = inputs.float(), labels.float()

                    recon_input, outputs = self.model(inputs)

                    loss_recon = criterion(recon_input, inputs)
                    loss_pred = criterion(outputs, labels)

                    loss = loss_recon + loss_pred

                    valid_loss = loss.item()
                    valid_loss_r += loss_recon.item()
                    valid_loss_p += loss_pred.item()

                    total_step_valid += 1

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
                    "total_loss/train", (running_loss / len(self.train_loader)), epoch
                )
                self.log_scalar(
                    "total_loss/validation",
                    (valid_loss / len(self.valid_loader)),
                    epoch,
                )

                self.log_scalar(
                    "recon_loss/train", (loss_r / len(self.train_loader)), epoch
                )
                self.log_scalar(
                    "recon_loss/validation",
                    (valid_loss_r / len(self.train_loader)),
                    epoch,
                )

                self.log_scalar(
                    "pred_loss/train", (loss_p / len(self.train_loader)), epoch
                )
                self.log_scalar(
                    "pred_loss/validation",
                    (valid_loss_p / len(self.valid_loader)),
                    epoch,
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
        test_recon_score = []
        test_pred_score = []

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
            # Training
            start = time.time()
            # Training
            model = self.train()
            train_score.append(self.score)
            fit_time.append(time.time() - start)
            # Testing
            model.eval()
            x_test = torch.from_numpy(self.x_data[test_index]).float().to(self.device)
            y_test = torch.from_numpy(self.y_data[test_index]).float().to(self.device)
            recon_x, pred_y = model(x_test)
            recon_score = criterion(x_test, recon_x).item()
            pred_score = criterion(y_test, pred_y).item()
            test_recon_score.append(recon_score)
            test_pred_score.append(pred_score)
            mean_score = np.mean([recon_score, pred_score])

            if mean_score < prev_score:
                best_model = model
            prev_score = mean_score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_recon_score": test_recon_score,
            "test_pred_score": test_pred_score,
        }
        self._print_kfold_result(result)

        return best_model

    def _print_kfold_result(self, scores: dict):
        # prints a summary table of the scores from k-fold cross validation;
        # summarises the elapsed time and train/test metric scores for each k-fold
        # with overall k-fold cross validation statistics (mean and std. dev.)
        # using the `scores` dictionary returned from `cross_validate`
        print("*" * 16 * 4)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test recon", "test pred"))
        print("*" * 16 * 4)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        for kf, (t, train, test_recon, test_pred) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_score"],
                scores["test_pred_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(test_recon),
                    np.absolute(test_pred),
                )
            )

        print("*" * 16 * 4)

        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))

        print("*" * 16 * 4)

        print("")
