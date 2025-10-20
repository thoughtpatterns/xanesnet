# <strong>XANESNET</strong>

See [the upstream project](https://github.com/NewcastleRSE/xray-spectroscopy-ml).

## Installation

This project uses Python 3.10, and is managed via
[`uv`](https://docs.astral.sh/uv). To install `uv`,
* on Windows, run
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* or on Unix, run
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
or see
[the installation page](https://docs.astral.sh/uv/getting-started/installation)
for package manager availability.

The project also depends on,
* `cmake (3.31.7)`,
* `gfortran (14.3.0)`,
* and `pkg-config (0.29.2)`.

The program is developed with the parenthesized versions, but lower versions may
work.

Then, to clone the project, and to install a virtual environment which will run
the project, run
```bash
# We store structure files and FDMNES output in a submodule at `/data/`, so
# we must supply `--recurse-submodules`.
git clone --recurse-submodules https://github.com/thoughtpatterns/xanesnet
cd xanesnet

# If you use a non-POSIX-compliant shell, `uv` will offer the correct command
# with which to source the virtual environment, i.e., "Activate with: <command>".
uv venv
source .venv/bin/activate # For POSIX-compliant shells.

uv sync # Installs each requisite package to the virtual environment.
```

If you use [`nix`](https://nixos.wiki/wiki/Nix_package_manager) with
[`direnv`](https://direnv.net), a `flake.nix` is provided, which pulls
`python310`, `uv`, and the library dependencies listed above, then sources the
virtual environment as you enter the project directory.

## Usage

To train a model, run

```bash
xanesnet --mode train_xyz --in_file <input> --save
```
where `<input>` is a `.yaml` file — see `/inputs/` for examples. Models will be
placed into `/models/`.

To test a model, run

```bash
xanesnet --mode predict_xanes --in_file <input> --in_model <model> --save
```
where `<model>` is a model directory within `/models/`. If specified in
`<input>`, plots and raw text files will be placed in `/outputs/`, within a
directory named for the input `<model>`.

Structure and spectrum files are placed into `/data/cocl2en2/`, and are
separated into,
- `lhs/`, for structures computed via Latin-hypercube sampling,
- and `modes/`, for structures computed via normal mode amplitude variance.

Each of these is further separated into `cis/` and `trans/`, then into names
for each dataset, and lastly, into,
- `xyz/`, for structure files,
- `txt/`, for FDMNES-computed spectrum files,
- and for some sets,
  - `unconv/`, for unconvolved spectrum files,
  - or `pickle/`, which stores normal mode amplitudes for use with the `amps`
    descriptor.

See the example input files in `/input/` for how to point XANESNET to a particular
dataset.

## License

This project is licensed under the GPL-3.0 License — see the LICENSE file for
details.
