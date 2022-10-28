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

import os
import numpy as np
import pickle as pickle
import tqdm as tqdm

from pathlib import Path
from glob import glob

from inout import load_xyz
from inout import load_xanes
from inout import save_xanes
from utils import unique_path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES
from mlp_pytorch import MLP
import torch
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from pyemd import emd_samples
from sklearn import preprocessing


###############################################################################
################################ ELEMENT CLASS ACCURACY #######################
###############################################################################


def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)

###############################################################################
################################ CONFUSION MATRIX #############################
###############################################################################

def plot_error_matrix(cm, classes, cmap=plt.cm.Blues):
    """Plot the error matrix for the neural network models"""

    from sklearn.metrics import confusion_matrix
    import itertools

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()


def CCM(cnf_labels, cnf_predictions):

    class_names = ["2", "3", "4"]
    # Compute confusion matrix
    # print(cnf_labels)
    # print(cnf_predictions)
    # cnf_label -- [0 0 0 0 ... 1 1 1 1]
    # size: cnf_labels (966,), cnf_predictions (966, )
    # type: cnf_labels <class 'tuple'>, cnf_predictions <class 'numpy.ndarray'>

    cnf_matrix = confusion_matrix(cnf_labels, cnf_predictions)
    np.set_printoptions(precision=2)

    # Normalise
    cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({"font.size": 16})

    # Plot normalized confusion matrix
    plt.figure()
    plot_error_matrix(cnf_matrix, classes=class_names)
    plt.tight_layout()
    filename = "alexnet_nweight.pdf"
    plt.savefig(filename, format="PDF", bbox_inches="tight")
    plt.show()


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(aemode: str, model_dir: str, x_path: str, y_path: str):
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
    print(model_dir)

    xyz_path = [Path(p) for p in glob(x_path)]
    xanes_path = [Path(p) for p in glob(y_path)]

    predict_dir = unique_path(Path("."), "predictions")
    predict_dir.mkdir()
    print(predict_dir)
    print(xyz_path)

    for n_element in range(0, len(xyz_path)):

        element_label = []
        element_name = str(xyz_path[n_element]).split("/")[-3]
        print(element_name)

        ids = list(
            set(list_filestems(xyz_path[n_element]))
            & set(list_filestems(xanes_path[n_element]))
        )

        ids.sort()

        with open(model_dir / "descriptor.pickle", "rb") as f:
            descriptor = pickle.load(f)

        n_samples = len(ids)
        n_x_features = descriptor.get_len()
        n_y_features = linecount(xanes_path[n_element] / f"{ids[0]}.txt") - 2

        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}x{} array for Y data...".format(*xanes_data.shape))
        print(">> ...everything preallocated!\n")

        print(">> loading data into array(s)...")
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            element_label.append(element_name)
            with open(xyz_path[n_element] / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
            xyz_data[i, :] = descriptor.transform(atoms)
            with open(xanes_path[n_element] / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
            e, xanes_data[i, :] = xanes.spectrum
        print(">> ...loaded!\n")

        model = torch.load(
            model_dir / "train_xanes_model.pt", map_location=torch.device("cpu")
        )
        model.eval()
        print("Loaded model from disk")

        le = preprocessing.LabelEncoder()
        element_label = le.fit_transform(element_label)
        element_label = torch.as_tensor(element_label + n_element)

        if aemode == "predict_xyz":
            print("predict xyz structure")

            n_sample = xanes_data.shape[0]
            print(n_sample)
            xanes = torch.from_numpy(xanes_data)
            xanes = xanes.float()

            print(">> predicting Y data with neural net...")
            # recon_xanes, pred_xyz, class_pred = model(xanes)

            class_pred = model(xanes)

            ps = torch.exp(class_pred)
            top_p, top_class = ps.topk(1, dim=1)
            accuracy = get_accuracy(element_label, top_class)
            print(element_label)
            print(torch.t(top_class))
            print(f"Accuracy: {accuracy/n_sample*100}")

            # print(mean_squared_error(xanes, recon_xanes.detach().numpy()))
            # print(mean_squared_error(xyz_data, pred_xyz.detach().numpy()))

        elif aemode == "train_xanes":
            print("training xanes spectrum")

            xyz = torch.from_numpy(xyz_data)
            xyz = xyz.float()

            print(">> predicting Y data with neural net...")
            recon_xyz, pred_xanes, class_pred = model(xyz)
            

            print(mean_squared_error(xyz, recon_xyz.detach().numpy()))
            print(mean_squared_error(xanes, pred_xanes.detach().numpy()))

            # with open(model_dir / 'dataset.npz', 'rb') as f:
            #     e = np.load(f)['e']

            print(">> saving Y data predictions...")

        # os.makedirs(os.path.join(predict_dir, element_name))

        # total_y = []
        # total_y_pred = []
        # total_x = []
        # total_x_recon = []

        # for id_, y_predict_, y_, x_recon_, x_ in tqdm.tqdm(
        #     zip(ids, pred_xyz, xyz_data, recon_xanes, xanes_data)
        # ):
        #     # sns.set()
        #     # plt.figure()
        #     # plt.plot(y_predict_.detach().numpy(), label="prediction")
        #     # plt.plot(y_, label="target")
        #     # plt.legend(loc="upper right")
        #     total_y.append(y_)
        #     total_y_pred.append(y_predict_.detach().numpy())

        #     total_x.append(x_)
        #     total_x_recon.append(x_recon_.detach().numpy())

        #     # with open(predict_dir / f'{id_}.txt', 'w') as f:
        #     #     save_xanes(f, XANES(e, y_predict_.detach().numpy()))
        #     #     plt.savefig(predict_dir / f'{id_}.pdf')
        #     # plt.close()
        # total_y = np.asarray(total_y)
        # total_y_pred = np.asarray(total_y_pred)
        # total_x = np.asarray(total_x)
        # total_x_recon = np.asarray(total_x_recon)

        # # plotting the average loss
        # sns.set_style("dark")
        # fig, (ax1, ax2) = plt.subplots(2)

        # mean_y = np.mean(total_y, axis=0)
        # stddev_y = np.std(total_y, axis=0)

        # ax1.plot(mean_y, label="target")
        # ax1.fill_between(
        #     np.arange(mean_y.shape[0]),
        #     mean_y + stddev_y,
        #     mean_y - stddev_y,
        #     alpha=0.4,
        #     linewidth=0,
        # )

        # mean_y_pred = np.mean(total_y_pred, axis=0)
        # stddev_y_pred = np.std(total_y_pred, axis=0)

        # ax1.plot(mean_y_pred, label="prediction")
        # ax1.fill_between(
        #     np.arange(mean_y_pred.shape[0]),
        #     mean_y_pred + stddev_y_pred,
        #     mean_y_pred - stddev_y_pred,
        #     alpha=0.4,
        #     linewidth=0,
        # )

        # ax1.legend(loc="best")
        # ax1.grid()

        # mean_x = np.mean(total_x, axis=0)
        # stddev_x = np.std(total_x, axis=0)

        # ax2.plot(mean_x, label="target")
        # ax2.fill_between(
        #     np.arange(mean_x.shape[0]),
        #     mean_x + stddev_x,
        #     mean_x - stddev_x,
        #     alpha=0.4,
        #     linewidth=0,
        # )

        # mean_x = np.mean(total_x_recon, axis=0)
        # stddev_x = np.std(total_x_recon, axis=0)

        # ax2.plot(mean_x, label="reconstruction")
        # ax2.fill_between(
        #     np.arange(mean_x.shape[0]),
        #     mean_x + stddev_x,
        #     mean_x - stddev_x,
        #     alpha=0.4,
        #     linewidth=0,
        # )

        # ax2.legend(loc="best")
        # ax2.grid()

        # plt.savefig(predict_dir / element_name / "plot.pdf")

        # plt.show()
        # fig.clf()
        # plt.close(fig)

        # print("...saved!\n")

    return 0
