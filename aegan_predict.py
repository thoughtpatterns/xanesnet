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
# from inout import save_xyz
from inout import load_xanes
from inout import save_xanes
# from inout import load_pipeline
# from inout import save_pipeline
from utils import unique_path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES
# from tensorflow.keras.models import model_from_json

import torch
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from pyemd import emd_samples

import AEGAN

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################
 
def main(
    model_dir: str,
    x_path: str,
    y_path:str
):
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
        y_path (str): The path to the .xyz (X) data; expects a directory
            containing xanes files.
    """

    model_dir = Path(model_dir)

    x_path = Path(x_path)
    y_path = Path(y_path)

    ids = list(
            set(list_filestems(x_path)) & set(list_filestems(y_path))
        )

    ids.sort()

    with open(model_dir / 'descriptor.pickle', 'rb') as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)
    n_x_features = descriptor.get_len()
    n_y_features = linecount(y_path / f'{ids[0]}.txt') - 2

    x = np.full((n_samples, n_x_features), np.nan)
    print('>> preallocated {}x{} array for X data...'.format(*x.shape))
    y = np.full((n_samples, n_y_features), np.nan)
    print('>> preallocated {}x{} array for Y data...'.format(*y.shape))
    print('>> ...everything preallocated!\n')

    print('>> loading data into array(s)...')
    for i, id_ in enumerate(tqdm.tqdm(ids)):
        with open(x_path / f'{id_}.xyz', 'r') as f:
            atoms = load_xyz(f)
        x[i,:] = descriptor.transform(atoms)
        with open(y_path / f'{id_}.txt', 'r') as f:
            xanes = load_xanes(f)
            # print(xanes.spectrum)
            e, y[i,:] = xanes.spectrum
    print('>> ...loaded!\n')


    # pipeline = load_pipeline(
    #     model_dir / 'net.keras',
    #     model_dir / 'pipeline.pickle'
    # )

    # load the model
    # model = MLP()
    model_file = open(model_dir / 'model.pt', 'r')
    # # loaded_model = torch.load(model_file)
    # model.load_state_dict(torch.load(model_file))

    model = torch.load(model_dir / 'model.pt', map_location=torch.device('cpu'))
    model.eval()
    print("Loaded model from disk")
    # print(model)


    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    print('>> Reconstructing and predicting data with neural net...')
    x_recon, y_recon, x_pred, y_pred  = model.reconstruct_all_predict_all(x,y)
    print('>> ...done!\n')


    print(f">>> MSE xyz-xyz     {mean_squared_error(x,x_recon.detach().numpy())}")
    print(f">>> MSE xanes-xanes {mean_squared_error(y,y_recon.detach().numpy())}")
    print(f">>> MSE xyz-xanes   {mean_squared_error(y,y_pred.detach().numpy())}")
    print(f">>> MSE xanes-xyz   {mean_squared_error(x,x_pred.detach().numpy())}")

    predict_dir = unique_path(Path('.'), 'predictions')
    predict_dir.mkdir()

    # with open(model_dir / 'dataset.npz', 'rb') as f:
    #     e = np.load(f)['e']

    print('>> Saving reconstructions and predictions...')
    for id_, x_, y_, x_recon_, y_recon_, x_pred_, y_pred_ in tqdm.tqdm(zip(ids, x, y, x_recon, y_recon, x_pred, y_pred)):
        sns.set()
        fig, (ax1, ax2,ax3, ax4) = plt.subplots(4,figsize=(20,20))

        ax1.plot(x_recon_.detach().numpy(), label="Reconstruction")
        ax1.set_title(f'Structure Reconstruction (MSE {mean_squared_error(x_,x_recon_.detach().numpy())})')
        ax1.plot(x_, label="target")
        ax1.legend(loc="upper left")

        ax2.plot(y_recon_.detach().numpy(), label="Reconstruction")
        ax2.set_title(f'Spectrum Reconstruction (MSE {mean_squared_error(y_,y_recon_.detach().numpy())})')
        ax2.plot(y_, label="target")
        ax2.legend(loc="upper left")

        ax3.plot(y_pred_.detach().numpy(), label="Prediction")
        ax3.set_title(f'Spectrum Prediction (MSE {mean_squared_error(y_,y_pred_.detach().numpy())})')
        ax3.plot(y_, label="target")
        ax3.legend(loc="upper left")
        
        ax4.plot(x_pred_.detach().numpy(), label="Prediction")
        ax4.set_title(f'Structure Prediction (MSE {mean_squared_error(x_,x_pred_.detach().numpy())})')
        ax4.plot(x_, label="target")
        ax4.legend(loc="upper left")

        # with open(predict_dir / f'{id_}.txt', 'w') as f:
            # save_xanes(f, XANES(e, y_predict_.detach().numpy()))
        plt.savefig(predict_dir / f'{id_}.pdf')
        fig.clf()
        plt.close(fig)
    print('...saved!\n')
        
    return 0