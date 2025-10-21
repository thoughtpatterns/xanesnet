"""Implement `fit()`."""

# ruff: noqa: B008, E501, E731, ISC003
# pyright: reportCallInDefaultInitializer=false

from pathlib import Path

from typer import Option

from fitter.cli import cli


@cli.command()
def fit(  # noqa: D103
    model_: Path = Option(
        ...,
        "--model",
        help='path to a pre-trained ".pt" model file',
        exists=True,
    ),
    spectrum_: Path = Option(
        ...,
        "--spectrum",
        help='path to the target ".npy" spectrum file',
        exists=True,
    ),
    output: Path = Option(
        ...,
        "--output",
        help='path to save the output modes, as an ".npy" file',
    ),
    bounds_: float | None = Option(
        None,
        "--bounds",
        help="symmetric bound for each amplitude",
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
            description="fit procedure imports called...",
            total=None,
        )

        from typing import Final

        from numpy import load as load_npy
        from numpy import mean, zeros
        from numpy import save as save_npy
        from numpy.typing import NDArray
        from scipy.optimize import minimize
        from torch import float32, load, no_grad, tensor
        from typer import Exit

        from xanesnet.models.mlp import MLP as Mlp  # noqa: N811

    # :: Load model and data. ::

    device = "cpu"  # SciPy and NumPy run on the CPU.

    try:
        checkpoint = load(model_, map_location=device)
        params = checkpoint["params"]
        weights = checkpoint["weights"]
    except KeyError as e:
        console.print(
            f'[red]fatal:[/red] model [cyan]"{model_}"[/cyan]'
            + ' must contain keys [cyan]"params"[/cyan] and [cyan]"weights"[/cyan]',
        )
        raise Exit(1) from e

    # :: Initialize model and spectra. ::

    model = Mlp(**params)
    _ = model.load_state_dict(weights)
    _ = model.eval()

    spectrum = load_npy(spectrum_)

    def objective(amplitudes: NDArray[float32]) -> float32:  # pyright: ignore[reportInvalidTypeForm]
        """Compute the MSE between predicted and target spectra."""
        features = tensor(amplitudes, dtype=float32).unsqueeze(0)
        with no_grad():
            predicted = model(features).squeeze().numpy()
        return mean((predicted - spectrum) ** 2)

    # Define fit parameters.
    fit_params: Final = {
        "x0": zeros(params["in_size"]),
        "bounds": [(-bounds_, bounds_)] * params["in_size"] if bounds_ else None,
        "method": "L-BFGS-B",
        "options": {"max_iter": 256},
    }

    # Run fit procedure.
    console.print("fit procedure started...")
    result = minimize(objective, **fit_params)

    # :: Save our modes, if successful. ::

    if result.success:
        save_npy(output, result.x)
        console.print(
            "fit procedure [green]succeeded[/green]"
            + f", with an MSE of {result.fun:.8f}"
            + f', and with modes saved to [cyan]"{output}"[/cyan]',
        )
    else:
        console.print(
            "fit procedure [yellow]failed[/yellow]"
            + f', with message "{result.message}"',
        )
