"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
import pickle as pickle
import tqdm as tqdm

from pathlib import Path

from inout import load_xyz
from inout import load_xanes
from inout import save_xanes
from utils import unique_path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES

import torch
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(model_dir: str, x_path: str, y_path: str):
    """
    PREDICT. The model state is restored from a model directory containing
    serialised scaling/pipeline objects and the serialised model, .xyz (X)
    data are loaded and transformed, and the model is used to predict XANES
    spectral (Y) data; convolution of the Y data is also possible if
    {conv_params} are provided (see xanesnet/convolute.py).

    Args:
        model_dir (str): The path to a model.[?] directory created by
            the LEARN routine.
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
    """

    model_dir = Path(model_dir)

    x_path = Path(x_path)
    y_path = Path(y_path)

    ids = list(set(list_filestems(x_path)) & set(list_filestems(y_path)))

    ids.sort()

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)
    n_x_features = descriptor.get_len()
    n_y_features = linecount(y_path / f"{ids[0]}.txt") - 2

    x = np.full((n_samples, n_x_features), np.nan)
    print(">> preallocated {}x{} array for X data...".format(*x.shape))
    y = np.full((n_samples, n_y_features), np.nan)
    print(">> preallocated {}x{} array for Y data...".format(*y.shape))
    print(">> ...everything preallocated!\n")

    print(">> loading data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(ids)):
        with open(x_path / f"{id_}.xyz", "r") as f:
            atoms = load_xyz(f)
        x[i, :] = descriptor.transform(atoms)
        with open(y_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
            e, y[i, :] = xanes.spectrum
    print(">> ...loaded!\n")

    model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
    model.eval()
    print("Loaded model from disk")

    x = torch.from_numpy(x)
    x = x.float()

    print(">> predicting Y data with neural net...")

    y_predict = model(x)
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    print(mean_squared_error(y, y_predict.detach().numpy()))

    predict_dir = unique_path(Path("."), "predictions")
    predict_dir.mkdir()

    with open(model_dir / "dataset.npz", "rb") as f:
        e = np.load(f)["e"]

    print(">> saving Y data predictions...")

    total_y = []
    total_y_pred = []
    for id_, y_predict_, y_ in tqdm.tqdm(zip(ids, y_predict, y)):
        sns.set()
        plt.figure()
        plt.plot(y_predict_.detach().numpy(), label="prediction")
        plt.plot(y_, label="target")
        plt.legend(loc="upper right")
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        with open(predict_dir / f"{id_}.txt", "w") as f:
            save_xanes(f, XANES(e, y_predict_.detach().numpy()))
            plt.savefig(predict_dir / f"{id_}.pdf")

        plt.close()

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)

    # plotting the average loss
    sns.set_style("dark")
    plt.figure()

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)
    plt.plot(mean_y, label="target")

    plt.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)
    plt.plot(mean_y_pred, label="prediction")
    plt.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    plt.legend(loc="best")
    plt.grid()
    plt.savefig(predict_dir / "plot.pdf")

    plt.show()

    print("...saved!\n")

    return 0
