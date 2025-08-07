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
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python310;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cmake
            gfortran
            llvmPackages.openmp
            pkg-config
            python
            uv
          ];

          NIX_CFLAGS_COMPILE = "-ftemplate-depth=2048"; # Needed for `dscribe`.

          shellHook = ''
            if ! [ -d .venv ]
            then uv venv -p ${python}/bin/python
            fi

            unset VIRTUAL_ENV
            . .venv/bin/activate
          '';
        };
      }
    );
}
