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
import random
import mlflow
import time
import mlflow.pytorch
import numpy as np
import torch
from datetime import datetime

from sklearn.model_selection import RepeatedKFold
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from .base_learn import Learn

from src.model_utils import LRScheduler, LossSwitch, loss_reg_fn, OptimSwitch
from src.creator import create_eval_scheme


class AEGANLearn(Learn):
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
        self.batch_size = hyperparams["batch_size"]
        self.weight_seed_hyper = hyperparams["weight_seed"]
        self.kernel = hyperparams["kernel_init"]
        self.bias = hyperparams["bias_init"]
        self.n_epoch = hyperparams["epochs"]
        self.model_eval = hyperparams["model_eval"]
        self.seed = hyperparams["seed"]
        self.loss_fn = hyperparams["loss_gen"]["loss_fn"]
        self.loss_args = hyperparams["loss_gen"]["loss_args"]

        # Regularisation of gen loss function
        self.loss_gen_reg_type = self.hyperparams["loss_gen"]["loss_reg_type"]
        self.loss_gen_reg = True if self.loss_gen_reg_type is not None else False
        self.lambda_gen_reg = self.hyperparams["loss_gen"]["loss_reg_param"]

        # Regularisation of dis loss function
        self.loss_dis_reg_type = self.hyperparams["loss_dis"]["loss_reg_type"]
        self.loss_dis_reg = True if self.loss_gen_reg_type is not None else False
        self.lambda_dis_reg = self.hyperparams["loss_dis"]["loss_reg_param"]

    def setup_writer(self):
        # setup tensorboard stuff
        layout = {
            "Multi": {
                "total_loss": ["multiline", ["total_loss"]],
                "recon_loss": ["Multiline", ["loss/x", "loss/y"]],
                "pred_loss": ["Multiline", ["loss_p/x", "loss_p/y"]],
            },
        }
        self.writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
        self.writer.add_custom_scalars(layout)

    def setup_scheduler(self, gen_opt, dis_opt):
        scheduler_gen = LRScheduler(
            gen_opt,
            scheduler_type=self.scheduler_param["scheduler_type"],
            params=self.scheduler_param["scheduler_param"],
        )
        scheduler_dis = LRScheduler(
            dis_opt,
            scheduler_type=self.scheduler_param["scheduler_type"],
            params=self.scheduler_param["scheduler_param"],
        )
        return scheduler_gen, scheduler_dis

    def train(self):
        # Move model to the available device
        self.model.to(self.device)
        self.setup_dataloader()
        self.setup_weight

        gen_opt, dis_opt = self.model.get_optimizer()
        # setup learning rate scheduler
        if self.schedular is not None:
            scheduler_gen, scheduler_dis = self.setup_scheduler(gen_opt, dis_opt)

        self.model.train()

        # Select loss function
        # Select running loss function as generative loss function
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        train_total_loss = [None] * self.n_epoch
        train_loss_x_recon = [None] * self.n_epoch
        train_loss_y_recon = [None] * self.n_epoch
        train_loss_x_pred = [None] * self.n_epoch
        train_loss_y_pred = [None] * self.n_epoch

        with mlflow.start_run(experiment_id=self.exp_id, run_name=self.exp_time):
            mlflow.log_params(self.hyperparams)
            mlflow.log_param("n_epoch", self.n_epoch)

            for epoch in range(self.n_epoch):
                self.model.train()
                running_loss_recon_x = 0
                running_loss_recon_y = 0
                running_loss_pred_x = 0
                running_loss_pred_y = 0
                running_gen_loss = 0
                running_dis_loss = 0

                for inputs_x, inputs_y in self.train_loader:
                    inputs_x, inputs_y = inputs_x.to(self.device), inputs_y.to(
                        self.device
                    )
                    inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

                    self.model.gen_update(inputs_x, inputs_y)
                    self.model.dis_update(inputs_x, inputs_y)

                    (
                        recon_x,
                        recon_y,
                        pred_x,
                        pred_y,
                    ) = self.model.reconstruct_all_predict_all(inputs_x, inputs_y)

                    # Track running losses
                    running_loss_recon_x += criterion(recon_x, inputs_x)
                    running_loss_recon_y += criterion(recon_y, inputs_y)
                    running_loss_pred_x += criterion(pred_x, inputs_x)
                    running_loss_pred_y += criterion(pred_y, inputs_y)

                    loss_gen_total = (
                        running_loss_recon_x
                        + running_loss_recon_y
                        + running_loss_pred_x
                        + running_loss_pred_y
                    )
                    loss_dis_total = self.model.loss_dis_total

                    # Regularisation of gen loss
                    if self.loss_gen_reg:
                        gen_a_l_reg = loss_reg_fn(
                            self.model.gen_a, self.loss_gen_reg_type, self.device
                        )
                        gen_b_l_reg = loss_reg_fn(
                            self.model.gen_b, self.loss_gen_reg_type, self.device
                        )
                        loss_gen_total += self.lambda_gen_reg * (
                            gen_a_l_reg + gen_b_l_reg
                        )

                    # Regularisation of dis loss
                    if self.loss_dis_reg:
                        dis_a_l_reg = loss_reg_fn(
                            self.model.dis_a, self.loss_dis_reg_type, self.device
                        )
                        dis_b_l_reg = loss_reg_fn(
                            self.model.dis_b, self.loss_dis_reg_type, self.device
                        )
                        loss_dis_total += self.lambda_dis_reg * (
                            dis_a_l_reg + dis_b_l_reg
                        )

                    running_gen_loss += loss_gen_total.item()
                    running_dis_loss += loss_dis_total.item()

                if self.schedular is not None:
                    before_lr_gen = gen_opt.param_groups[0]["lr"]
                    scheduler_gen.step()
                    after_lr_gen = gen_opt.param_groups[0]["lr"]
                    print(
                        "Epoch %d: Adam lr %.5f -> %.5f"
                        % (epoch, before_lr_gen, after_lr_gen)
                    )

                    before_lr_dis = dis_opt.param_groups[0]["lr"]
                    scheduler_dis.step()
                    after_lr_dis = dis_opt.param_groups[0]["lr"]
                    print(
                        "Epoch %d: Adam lr %.5f -> %.5f"
                        % (epoch, before_lr_dis, after_lr_dis)
                    )

                running_gen_loss = running_gen_loss / len(self.train_loader)
                running_dis_loss = running_dis_loss / len(self.train_loader)

                running_loss_recon_x = running_loss_recon_x.item() / len(
                    self.train_loader
                )
                running_loss_recon_y = running_loss_recon_y.item() / len(
                    self.train_loader
                )
                running_loss_pred_x = running_loss_pred_x.item() / len(
                    self.train_loader
                )
                running_loss_pred_y = running_loss_pred_y.item() / len(
                    self.train_loader
                )

                self.log_scalar("gen_loss", running_gen_loss, epoch)
                self.log_scalar("dis_loss", running_dis_loss, epoch)
                self.log_scalar("recon_x_loss", running_loss_recon_x, epoch)
                self.log_scalar("recon_y_loss", running_loss_recon_y, epoch)
                self.log_scalar("pred_x_loss", running_loss_pred_x, epoch)
                self.log_scalar("pred_y_loss", running_loss_pred_y, epoch)

                train_loss_x_recon[epoch] = running_loss_recon_x
                train_loss_y_recon[epoch] = running_loss_recon_y
                train_loss_x_pred[epoch] = running_loss_pred_x
                train_loss_y_pred[epoch] = running_loss_pred_y

                train_total_loss[epoch] = running_gen_loss

                valid_loss_recon_x = 0
                valid_loss_recon_y = 0
                valid_loss_pred_x = 0
                valid_loss_pred_y = 0
                valid_loss_total = 0

                self.model.eval()

                for inputs_x, inputs_y in self.valid_loader:
                    inputs_x, inputs_y = inputs_x.to(self.device), inputs_y.to(
                        self.device
                    )
                    inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

                    (
                        recon_x,
                        recon_y,
                        pred_x,
                        pred_y,
                    ) = self.model.reconstruct_all_predict_all(inputs_x, inputs_y)

                    valid_loss_recon_x += criterion(recon_x, inputs_x)
                    valid_loss_recon_y += criterion(recon_y, inputs_y)
                    valid_loss_pred_x += criterion(pred_x, inputs_x)
                    valid_loss_pred_y += criterion(pred_y, inputs_y)

                    valid_loss = (
                        valid_loss_recon_x
                        + valid_loss_recon_y
                        + valid_loss_pred_x
                        + valid_loss_pred_y
                    )

                    valid_loss_total += valid_loss.item()

                valid_loss_recon_x = valid_loss_recon_x.item() / len(self.valid_loader)
                valid_loss_recon_y = valid_loss_recon_y.item() / len(self.valid_loader)
                valid_loss_pred_x = valid_loss_pred_x.item() / len(self.valid_loader)
                valid_loss_pred_y = valid_loss_pred_y.item() / len(self.valid_loader)

                self.log_scalar("valid_recon_x_loss", valid_loss_recon_x, epoch)
                self.log_scalar("valid_recon_y_loss", valid_loss_recon_y, epoch)
                self.log_scalar("valid_pred_x_loss", valid_loss_pred_x, epoch)
                self.log_scalar("valid_pred_y_loss", valid_loss_pred_y, epoch)

                print(f">>> Epoch {epoch}...")

                print(f">>> Training loss (recon x)   = {running_loss_recon_x:.4f}")
                print(f">>> Training loss (recon y)   = {running_loss_recon_y:.4f}")
                print(f">>> Training loss (pred x)    = {running_loss_pred_x:.4f}")
                print(f">>> Training loss (pred y)    = {running_loss_pred_y:.4f}")

                print(f">>> Validation loss (recon x) = {valid_loss_recon_x:.4f}")
                print(f">>> Validation loss (recon y) = {valid_loss_recon_y:.4f}")
                print(f">>> Validation loss (pred x)  = {valid_loss_pred_x:.4f}")
                print(f">>> Validation loss (pred y)  = {valid_loss_pred_y:.4f}")

                losses = {
                    "train_loss": train_total_loss,
                    "loss_x_recon": train_loss_x_recon,
                    "loss_y_recon": train_loss_y_recon,
                    "loss_x_pred": train_loss_x_pred,
                    "loss_y_pred": train_loss_y_pred,
                }

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
        self.score = losses

        return self.model

    def train_kfold(self):
        # K-fold Cross Validation model evaluation
        prev_score = 1e6
        fit_time = []
        train_score = []
        test_recon_xyz_score = []
        test_recon_xanes_score = []
        test_pred_xyz_score = []
        test_pred_xanes_score = []

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

            train_score.append(self.score["train_loss"][-1])
            fit_time.append(time.time() - start)
            # Testing
            model.eval()
            xyz_test = torch.from_numpy(self.x_data[test_index]).float()
            xanes_test = torch.from_numpy(self.y_data[test_index]).float()
            (
                recon_xyz,
                recon_xanes,
                pred_xyz,
                pred_xanes,
            ) = model.reconstruct_all_predict_all(xyz_test, xanes_test)

            recon_xyz_score = criterion(xyz_test, recon_xyz).item()
            recon_xanes_score = criterion(xanes_test, recon_xanes).item()
            pred_xyz_score = criterion(xyz_test, pred_xyz).item()
            pred_xanes_score = criterion(xanes_test, pred_xanes).item()

            test_recon_xyz_score.append(recon_xyz_score)
            test_recon_xanes_score.append(recon_xanes_score)
            test_pred_xyz_score.append(pred_xyz_score)
            test_pred_xanes_score.append(pred_xanes_score)

            mean_score = np.mean(
                [
                    recon_xyz_score,
                    recon_xanes_score,
                    pred_xyz_score,
                    pred_xanes_score,
                ]
            )
            if mean_score < prev_score:
                best_model = model
            prev_score = mean_score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_recon_xyz_score": test_recon_xyz_score,
            "test_recon_xanes_score": test_recon_xanes_score,
            "test_pred_xyz_score": test_pred_xyz_score,
            "test_pred_xanes_score": test_pred_xanes_score,
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
        print("*" * (16 + 18 * 5))
        fmt = "{:<10s}{:>6s}{:>18s}{:>18s}{:>18s}{:>18s}{:>18s}"
        print(
            fmt.format(
                "k-fold",
                "time",
                "train",
                "test recon xyz",
                "test recon xanes",
                "test pred xyz",
                "test pred xanes",
            )
        )
        print("*" * (16 + 18 * 5))

        fmt = "{:<10.0f}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        for kf, (t, train, recon_xyz, recon_xanes, pred_xyz, pred_xanes) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_xyz_score"],
                scores["test_recon_xanes_score"],
                scores["test_pred_xyz_score"],
                scores["test_pred_xanes_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(recon_xyz),
                    np.absolute(recon_xanes),
                    np.absolute(pred_xyz),
                    np.absolute(pred_xanes),
                )
            )

        print("*" * (16 + 18 * 5))

        fmt = "{:<10s}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))
        print("*" * (16 + 18 * 5))

        print("")

    def log_scalar(self, name, value, epoch):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, epoch)
        mlflow.log_metric(name, value)
