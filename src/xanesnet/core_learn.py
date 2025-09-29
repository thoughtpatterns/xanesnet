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

import logging
import random
import sys
import torch
import time

from datetime import timedelta
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch import nn
from torchinfo import summary

from xanesnet.models.pre_trained import PretrainedModels
from xanesnet.utils.mode import Mode, get_mode
from xanesnet.utils.switch import KernelInitSwitch, BiasInitSwitch
from xanesnet.utils.io import (
    save_models,
    load_pretrained_descriptors,
    load_pretrained_model,
)
from xanesnet.creator import (
    create_learn_scheme,
    create_descriptors,
    create_model,
    create_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        # logging.FileHandler("train.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)


def train(config, args):
    """
    Train ML model based on the provided configuration and arguments.
    """
    logging.info(f">> Training mode: {args.mode}")
    mode = get_mode(args.mode)

    # Setup descriptors from inputscript or pretrained model
    descriptor_list = _setup_descriptors(config)

    # Load, encode, and preprocess data
    dataset = _setup_datasets(config, mode, descriptor_list)

    # Setup model from inputscript or pretrained model
    model = _setup_model(config, dataset)

    # Setup training scheme
    scheme = _setup_scheme(config, args, model, dataset)

    # Run model training
    model_list, scheme_type, train_time = _train_models(config, scheme)

    # Print trained model summary
    _summary_model(model_list[0], dataset)

    # Print training time
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model, encoded data and config to disk
    if args.save:
        metadata = {
            "mode": args.mode,
            "dataset": dataset.config,
            "model": model.config,
            "descriptors": [desc.config for desc in descriptor_list],
            "scheme": scheme_type,
            "standardscaler": config["standardscaler"],
        }

        save_models(Path("models"), model_list, metadata)


def _setup_descriptors(config):
    """Initialises or loads descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading descriptors from pretrained model: {model_type}")
        descriptor_list = load_pretrained_descriptors(model_type)
    else:
        descriptor_config = config["descriptors"]
        descriptor_types = ", ".join(d["type"] for d in descriptor_config)
        logging.info(f">> Initialising descriptors: {descriptor_types}")
        descriptor_list = create_descriptors(config=descriptor_config)

    return descriptor_list


def _setup_datasets(config, mode, descriptor_list):
    dataset_type = config["dataset"]["type"]

    logging.info(f">> Initialising training datasets: {dataset_type}")
    # Pack kwargs
    kwargs = {
        "root": config["dataset"]["root_path"],
        "xyz_path": config["dataset"]["xyz_path"],
        "xanes_path": config["dataset"]["xanes_path"],
        "mode": mode,
        "descriptors": descriptor_list,
        **config["dataset"].get("params", {}),
    }

    dataset = create_dataset(dataset_type, **kwargs)

    # Log dataset summary
    logging.info(
        f">> Dataset Summary: # of samples = {len(dataset)}, feature(X) size = {dataset.x_size}, label(y) size = {dataset.y_size}"
    )

    return dataset


def _setup_model(config, dataset):
    """Initialises or loads the model and its descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model: {model_type}")
        model_params = config["model"].get("params", {})
        model = load_pretrained_model(model_type, **model_params)
    else:
        logging.info(f">> Initialising model: {model_type}")
        model_params = config["model"].get("params", {})

        # Add additional model parameters
        model_params["in_size"] = dataset.x_size
        model_params["out_size"] = dataset.y_size

        model = create_model(model_type, **model_params)

        weights_config = config["model"].get("weights", {})
        weight_kernel = weights_config.get("kernel", "xavier_uniform")
        logging.info(f">> Initialising model weights: {weight_kernel}")
        model = _init_model_weights(model, **weights_config)

    return model


def _init_model_weights(model, **kwargs):
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


def _setup_scheme(config, args, model, dataset):
    model_type = config["model"].get("type")

    logging.info(">> Initialising training scheme")
    # Pack kwargs
    kwargs = {
        "model_config": config.get("model"),
        "hyper_params": config.get("hyperparams", {}),
        "kfold_params": config.get("kfold_params", {}),
        "bootstrap_params": config.get("bootstrap_params", {}),
        "ensemble_params": config.get("ensemble_params", {}),
        "lr_scheduler": config.get("lr_scheduler", False),
        "scheduler_params": config.get("scheduler_params", {}),
        "mlflow": args.mlflow,
        "tensorboard": args.tensorboard,
    }

    scheme = create_learn_scheme(model_type, model, dataset, **kwargs)

    return scheme


def _train_models(config, scheme):
    model_list = []
    start_time = time.time()
    if config["bootstrap"]:
        logging.info(">> Training model using bootstrap resampling...\n")
        scheme_type = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        logging.info(">> Training model using ensemble learning...\n")
        scheme_type = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        logging.info(">> Training model using kfold cross-validation...\n")
        scheme_type = "kfold"
        model_list.append(scheme.train_kfold())
    else:
        logging.info(">> Training model using standard training procedure...\n")
        scheme_type = "std"
        model_list.append(scheme.train_std())

    train_time = time.time() - start_time

    return model_list, scheme_type, train_time


def _summary_model(model, dataset):
    logging.info("\n--- Model Summary ---")

    if model.aegan_flag:
        dummy_x = torch.randn(1, dataset.x_size)
        dummy_y = torch.randn(1, dataset.y_size)
        input_data = (dummy_x, dummy_y)
    elif model.batch_flag:
        input_data = None
    else:
        dummy_x = torch.randn(1, dataset.x_size)
        input_data = dummy_x

    summary(model, input_data=input_data)
