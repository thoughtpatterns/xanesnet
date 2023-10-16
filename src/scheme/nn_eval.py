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

import torch

from .base_eval import Eval


class NNEval(Eval):
    def eval(self):
        print(f"{'='*20} Running Model Evaluation Tests {'='*20}")

        test_results = {}
        l0 = self.get_true_loss()

        li1 = self.get_loss_input_shuffle()
        lo1 = self.get_loss_output_shuffle()

        li2 = self.get_loss_input_mean_train()
        lo2 = self.get_loss_output_mean_train()

        li3 = self.get_loss_input_mean_sd_train()
        lo3 = self.get_loss_output_mean_sd_train()

        li4 = self.get_loss_input_random_valid()
        lo4 = self.get_loss_output_random_valid()

        test_results["Shuffle Input"] = Eval.loss_ttest(l0, li1)
        test_results["Shuffle Output"] = Eval.loss_ttest(l0, lo1)

        test_results["Mean Train Input"] = Eval.loss_ttest(l0, li2)
        test_results["Mean Train Output"] = Eval.loss_ttest(l0, lo2)

        test_results["Mean Std. Train Input"] = Eval.loss_ttest(l0, li3)
        test_results["Mean Std. Train Output"] = Eval.loss_ttest(l0, lo3)

        test_results["Random Valid Input"] = Eval.loss_ttest(l0, li4)
        test_results["Random Valid Output"] = Eval.loss_ttest(l0, lo4)

        for k, v in test_results.items():
            print(f">>> {k:25}: {v}")

        test_results = {"ModelEvalResults-Prediction": test_results}

        print(f"{'='*19} MLFlow: Evaluation Results Logged {'='*18}")
        return test_results

    def get_true_loss(self):
        true_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()
            target = self.model(inputs)
            loss = Eval.functional_mse(target, labels)
            true_loss.extend(loss)

        return true_loss

    def get_loss_input_shuffle(self):
        other_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(inputs.shape[0])
            inputs = inputs[idx]

            target = self.model(inputs)

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_output_shuffle(self):
        other_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(labels.shape[0])
            labels = labels[idx]

            target = self.model(inputs)

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_input_mean_train(self):
        other_loss = []

        target_output = self.model(self.mean_input)

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()
            target = target_output.repeat(labels.shape[0], 1)
            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_output_mean_train(self):
        other_loss = []

        target_output = self.mean_output

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()
            target = target_output.repeat(labels.shape[0], 1)
            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_input_mean_sd_train(self):
        other_loss = []

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()

            mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.n_in], device=self.device),
                self.std_input.to(self.device),
            )

            target = self.model(mean_sd_input)

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_output_mean_sd_train(self):
        other_loss = []

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()

            target = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.out_dim], device=self.device),
                self.std_output.to(self.device),
            )

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_input_random_valid(self):
        other_loss = []

        it = iter(self.valid_loader)

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()
            alt_inputs, _ = next(it)
            alt_inputs = alt_inputs.to(self.device).float()
            if labels.shape[0] < alt_inputs.shape[0]:
                alt_inputs = alt_inputs[: labels.shape[0], :]

            target = self.model(alt_inputs)

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss

    def get_loss_output_random_valid(self):
        other_loss = []

        it = iter(self.valid_loader)

        for _, labels in self.eval_loader:
            labels = labels.to(self.device)
            labels = labels.float()
            _, target = next(it)
            target = target.to(self.device).float()
            if labels.shape[0] < target.shape[0]:
                target = target[: labels.shape[0], :]

            loss = Eval.functional_mse(target, labels)
            other_loss.extend(loss)

        return other_loss
