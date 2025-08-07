{
  description = "Developer shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # Required for `cudatoolkit`.
        };

        python = pkgs.python310;

        commonArgs = {
          buildInputs = with pkgs; [
            cmake
            gfortran
            llvmPackages.openmp
            pkg-config
            python
            uv
          ];

          NIX_CFLAGS_COMPILE = "-ftemplate-depth=2048"; # Required for `dscribe`.

          shellHook = ''
            if ! [ -d .venv ]
            then uv venv -p ${python}/bin/python
            fi

            unset VIRTUAL_ENV
            . .venv/bin/activate

            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH-}"
          '';
        };

        cudaPackages = with pkgs; [
          cudatoolkit_11
          linuxPackages.nvidia_x11
          ncurses5
        ];

        cudaShellHook =
          commonArgs.shellHook
          + ''
            export CUDA_PATH="${pkgs.cudatoolkit_11}"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaPackages}:/usr/lib/wsl/lib:''${LD_LIBRARY_PATH-}"
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
      in
      {
        devShells = {
          default = pkgs.mkShell commonArgs;
          cuda = pkgs.mkShell (
            commonArgs
            // {
              buildInputs = commonArgs.buildInputs ++ cudaPackages;
              shellHook = cudaShellHook;
            }
          );
        };
      }
    );
}
