import seaborn as sns
import matplotlib.pyplot as plt
import tqdm as tqdm
import numpy as np

from inout import save_xanes
from spectrum.xanes import XANES

def plot_predict(ids, y, y_predict, e, predict_dir, mode):
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

        if mode == 'predict_xanes':
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == 'predict_xyz':
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        plt.close()

    print(">> saving Y data predictions...")

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
    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()

def plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode):
    total_y = []
    total_y_pred = []
    total_x = []
    total_x_recon = []

    for id_, y_predict_, y_, x_recon_, x_ in tqdm.tqdm(zip(ids, y_predict, y, x_recon, x)):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(y_predict_.detach().numpy(), label="prediction")
        ax1.set_title("prediction")
        ax1.plot(y_, label="target")
        ax1.legend(loc="upper right")

        ax2.plot(x_recon_.detach().numpy(), label="prediction")
        ax2.set_title("reconstruction")
        ax2.plot(x_, label="target")
        ax2.legend(loc="upper right")
        # print(type(x_))
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        total_x.append(x_.detach().numpy())
        total_x_recon.append(x_recon_.detach().numpy())
        if mode == 'predict_xanes':
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == 'predict_xyz':
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        fig.clf()
        plt.close(fig)
        
    print(">> saving Y data predictions...")

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)
    total_x = np.asarray(total_x)
    total_x_recon = np.asarray(total_x_recon)

    # plotting the average loss
    sns.set_style("dark")
    fig, (ax1, ax2) = plt.subplots(2)

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)

    ax1.plot(mean_y, label="target")
    ax1.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)

    ax1.plot(mean_y_pred, label="prediction")
    ax1.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    ax1.legend(loc="best")
    ax1.grid()

    mean_x = np.mean(total_x, axis=0)
    stddev_x = np.std(total_x, axis=0)

    ax2.plot(mean_x, label="target")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    mean_x = np.mean(total_x_recon, axis=0)
    stddev_x = np.std(total_x_recon, axis=0)

    ax2.plot(mean_x, label="reconstruction")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    ax2.legend(loc="best")
    ax2.grid()

    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()
    fig.clf()
    plt.close(fig)