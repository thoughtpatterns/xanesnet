"""Implement `train()`."""

# ruff: noqa: B008, E501, E731, ISC003
# pyright: reportCallInDefaultInitializer=false

from pathlib import Path

from typer import Option

from fitter.cli import cli


@cli.command()
def train(  # noqa: D103, PLR0915
    modes: Path = Option(
        ...,
        "--modes",
        help='path to a 2D ".npy" file of modes',
        exists=True,
    ),
    spectra: Path = Option(
        ...,
        "--spectra",
        help='path to a 2D ".npy" file of spectra',
        exists=True,
    ),
    output: Path = Option(
        ...,
        "--output",
        help='path to save the trained model, as a ".pt" file',
    ),
) -> None:
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from fitter.cli import console

    p = lambda: Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

    # Import within the command, to save time. Print a message as we import.
    with p() as progress:
        _ = progress.add_task(
            description="train procedure imports called...",
            total=None,
        )

        from copy import deepcopy
        from typing import Final

        from torch import cuda, inf, no_grad, save
        from torch.nn import MSELoss
        from torch.optim import Adam
        from torch.utils.data import DataLoader

        from fitter.dataset import Dataset
        from xanesnet.models.mlp import MLP as Mlp  # noqa: N811

    # Define non-model hyperparameters.
    batch: Final = 32
    lr: Final = 1e-4
    epochs: Final = 512

    # :: Define datasets and loaders. ::
    loader = lambda d, s: DataLoader(d, batch_size=batch, shuffle=s)

    tdataset, vdataset = Dataset.from_npy(modes, spectra)
    tloader = loader(tdataset, s=True)
    vloader = loader(vdataset, s=False)

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

    # Define CUDA availability.
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
        + f", [cyan]with{'' if device == 'cuda' else 'out'}[/cyan] CUDA...",
        end="\n---\n",
    )

    best_vloss = inf
    weights = None

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
            f"epoch {epoch + 1:0{len(str(epochs))}d} of {epochs}"
            + f", train loss: {mean_tloss:.8f}"
            + f", validation loss: {mean_vloss:.8f}",
            end="",  # Leave space for an asterisk, if we've reached a new minimum.
        )

        # :: Update our best model & loss cache, if necessary. ::

        if mean_vloss >= best_vloss:
            console.print()  # We did not print an asterisk --- simply print the newline.
            continue

        best_vloss = mean_vloss
        weights = deepcopy(model.state_dict())
        console.print(" [bold green]*[/bold green]")

    console.print("---")

    if weights:
        save({"params": params, "weights": weights}, output)
        console.print(
            "train procedure [green]succeeded[/green]"
            + f', with model saved to [cyan]"{output}"[/cyan]',
        )
    else:
        console.print(
            "train procedure [yellow]failed[/yellow]"
            + ", as no acceptable model was reached",
        )
