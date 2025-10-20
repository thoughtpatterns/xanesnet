"""Implement `train()`."""

# ruff: noqa: B008, E501, E731, ISC003
# pyright: reportCallInDefaultInitializer=false

from pathlib import Path

from typer import Option

from fit.cli import cli


@cli.command()
def train(  # noqa: D103
    tfeatures: Path = Option(
        ...,
        "--tfeatures",
        help="path to a directory of train feature `.npy` files",
        exists=True,
    ),
    ttargets: Path = Option(
        ...,
        "--ttargets",
        help="path to a directory of train target `.npy` files",
        exists=True,
    ),
    vfeatures: Path = Option(
        ...,
        "--vfeatures",
        help="path to a directory of validation feature `.npy` files",
        exists=True,
    ),
    vtargets: Path = Option(
        ...,
        "--vtargets",
        help="path to a directory of validation target `.npy` files",
        exists=True,
    ),
    output: Path = Option(
        ...,
        "--output",
        help="path to save the trained model, as a `.pt` file",
    ),
) -> None:
    from copy import deepcopy
    from typing import Final

    from torch import cuda, inf, no_grad, save
    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from fit.cli import console
    from fit.dataset import Dataset
    from xanesnet.models.mlp import MLP as Mlp  # noqa: N811

    # Define non-model hyperparameters.
    batch: Final = 32
    lr: Final = 1e-4
    epochs: Final = 512

    # Define datasets and loaders.
    loader = lambda d, s: DataLoader(d, batch_size=batch, shuffle=s)
    tloader = loader(tdataset := Dataset(tfeatures, ttargets), s=True)
    vloader = loader(Dataset(vfeatures, vtargets), s=False)

    # Define other hyperparameters.
    in_size, out_size = tdataset.dimensions
    params: Final = {
        "in_size": in_size,
        "out_size": out_size,
        "hidden_size": 256,
        "dropout": 0.1,
        "num_hidden_layers": 3,
        "shrink_rate": 1.0,
        "activation": "relu",
    }

    # Define and print CUDA availability.
    device = "cuda" if cuda.is_available() else "cpu"

    # Define model.
    model = Mlp(**params).to(device)  # pyright: ignore[reportArgumentType]
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # :: Define helpers. ::

    # (1) Forward pass our `i` tensor through the model's layers, which yields a tensor,
    # (2) then evaluate our output tensor against our truth, `t`, via our loss function,
    # (3) then return our loss tensor.
    loss = lambda i, t: criterion(model(i.to(device)), t.to(device))

    # (1) Calculate loss for the batch,
    # (2) perform backpropagation,
    # (3) update the model's weights,
    # (4) then return the scalar from the loss tensor.
    step = lambda i, t: (s := loss(i, t), s.backward(), optimizer.step(), s.item())[3]

    # (1) Take an optimizer,
    # (2) reset the model gradients to zero,
    # (3) then return `tloader`.
    reset = lambda p: (p.zero_grad(), tloader)[1]

    # :: Start train procedure. ::

    console.print(
        "train procedure started"
        + ", with device '[bold cyan]{device}[/bold cyan]'...",
    )

    best_vloss = inf
    best_weights = None

    for epoch in range(epochs):
        # Step, and get train loss.
        _ = model.train()
        tloss = sum(step(inputs, targets) for inputs, targets in reset(optimizer))

        # Evaluate for validation loss.
        _ = model.eval()
        with no_grad():
            vloss = sum(loss(inputs, targets).item() for inputs, targets in vloader)

        # :: Print a summary of our epoch. ::

        mean_tloss = tloss / len(tloader)
        mean_vloss = vloss / len(vloader)

        console.print(
            f"epoch {epoch + 1:05d} of {epochs}"
            + f", train loss: {mean_tloss:.6f}"
            + f", validation loss: {mean_vloss:.6f}",
        )

        # :: Update our best model & loss cache, if necessary. ::

        if mean_vloss >= best_vloss:
            continue

        best_vloss = mean_vloss
        best_weights = deepcopy(model.state_dict())
        console.print("    -> [bold green]new minimum reached[/bold green]")

    if best_weights:
        save({"model_params": params, "model_state_dict": best_weights}, output)
        console.print(
            "\n[bold green]train procedure completed[/bold green]"
            + f", model saved to [cyan]{output}[/cyan]",
        )
    else:
        console.print(
            "\n[bold yellow]train procedure completed"
            + ", but no suitable model was found to save[/bold yellow]",
        )
