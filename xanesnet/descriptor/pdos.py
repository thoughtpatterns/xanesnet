"""
XANESNET

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
from pyscf import scf, gto

from ase import Atoms
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

from xanesnet.descriptor import WACSF

###############################################################################
################################## CLASSES ####################################
###############################################################################


class PDOS(WACSF):
    """
    A class for transforming a molecular system into a project density of
    states representation.
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 6.0,
        e_min: float = -20.0,
        e_max: float = 20.0,
        sigma: float = 0.7,
        orb_type: str = "p",
        num_points: float = 200,
        basis: str = "3-21g",
        init_guess: str = "minao",
        max_scf_cycles: float = 0,
        use_wacsf=False,
        n_g2: int = 0,
        n_g4: int = 0,
        l: list = [1.0, -1.0],
        z: list = [1.0],
        g2_parameterisation: str = "shifted",
        g4_parameterisation: str = "centred",
        use_charge=False,
        use_spin=False,
    ):
        """
        Args:
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            n_g2 (int, optional): The number of G2 symmetry functions to use
                for encoding.
                Defaults to 0.
            n_g4 (int, optional): The number of G4 symmetry functions to use
                for encoding.
                Defaults to 0.
            l (list, optional): List of lambda values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0, -1.0].
            z (list, optional): List of zeta values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0].
            g2_parameterisation (str, optional): The strategy to use for G2
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
            g4_parameterisation (str, optional): The strategy to use for G4
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'centred'.
            e_min (float, optional): The minimum energy grid point for the pDOS (in eV)
                Default: -20.0 eV.
            e_max (float, optional): The maximum energy grid point for the pDOS (in eV)
                Default: 20.0 eV.
            sigma (float, optional): This is the FWHM of the Gaussian function used to
                broaden the pDOS obtained from pySCF.
                Default: 0.7 eV.
            num_points (float, optional): This is the number of point over which the broadened
                pDOS is projected.
                Default: 200.
            basis (string, optional): This is the basis set used by pySCF during developing
                the pDOS.
                Default: 3-21G
            basis (string, optional): Defines the method of the initial guess used by pySCF
                during generation of the pDOS.
                Default: minao
            max_scf_cycles (float, optional): This is the number of SCF cycles used by pySCF
                during develop the pDOS. Smaller numbers will be closer to the raw guess, while
                larger number will take longer to load.
                Note, the warnings are suppressed and so it will not tell you if the SCF is
                converged. Larger numbers make this more likely, but do not gurantee it.
                Default: 0
            use_wacsf (bool): If True, the wacsf descriptor for the structure is also generated
                and concatenated onto the end after the pDOS descriptor.
                Defaults to False.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """
        if use_wacsf:
            super().__init__(
                r_min,
                r_max,
                n_g2,
                n_g4,
                l,
                z,
                g2_parameterisation,
                g4_parameterisation,
                use_charge,
                use_spin,
            )

        self.e_min = e_min
        self.e_max = e_max
        self.num_points = num_points
        self.max_scf_cycles = max_scf_cycles
        self.basis = basis
        self.sigma = sigma
        self.init_guess = init_guess
        self.orb_type = orb_type
        self.use_wacsf = use_wacsf

    def transform(self, system: Atoms) -> np.ndarray:
        # Move molecular geometry from ASE environment to pySCF format
        mol = gto.Mole()
        mol.atom = atoms_to_pyscf(system)
        mol.basis = self.basis
        #       charge=0
        #       spin=0
        mol.build()

        # Create a SCF (Self-Consistent Field) object with a specific max_cycle value
        max_scf_cycles = self.max_scf_cycles
        mf = scf.RHF(mol)
        mf.init_guess = self.init_guess
        mf.max_cycle = max_scf_cycles

        # Perform the SCF calculation, suppress warnings as we know SCF isn't converged!
        mf.verbose = 0
        mf.kernel()

        # Get the atomic orbital coefficients for molecular orbitals
        ao_coefficients = mf.mo_coeff
        ao_labels = mol.ao_labels()

        # Get the orbital energies
        orbital_energies = mf.mo_energy
        pdos = np.zeros_like(orbital_energies)

        # Calculate the squared magnitude of coefficients and convert to percentages
        coeff_magnitude_squared = np.square(ao_coefficients)
        coeff_percentage = coeff_magnitude_squared / np.sum(
            coeff_magnitude_squared, axis=0
        )

        # Find the index of the first atom's (absorbing atom) atomic orbital labels
        first_atom_index = ao_labels.index(
            [label for label in ao_labels if label.split()[0] == "1"][0]
        )

        # Find coeff_percent for absorbing atom
        for i, coeff_percent in enumerate(
            coeff_percentage.T
        ):  # Transpose for easier iteration
            p_contribution = 0
            for j, percent in enumerate(coeff_percent):
                if j < first_atom_index:
                    label = ao_labels[j]
                    atomic_num, orb_type = label.split()[1], label.split()[2]
                    if self.orb_type in orb_type:
                        p_contribution = p_contribution + percent
            pdos[i] = p_contribution

        # Convert orbital energies from atomic units to eV
        orbital_energies = np.multiply(orbital_energies, 27.211324570273)
        # Filter out the occupied orbitals
        unoccupied_orbital_energies = orbital_energies[mol.nelectron // 2 :]
        unoccupied_pdos = pdos[mol.nelectron // 2 :]

        # Generate a grid and broaden pDOS
        x = np.linspace(self.e_min, self.e_max, num=self.num_points, endpoint=True)
        sigma = self.sigma
        gE = spectrum(unoccupied_orbital_energies, unoccupied_pdos, sigma, x)

        pdos_gauss = gE
        #       pdos_gauss = np.multiply(pdos_gauss,20)

        if self.use_wacsf:
            pdos_gauss = np.append(pdos_gauss, super().transform(system))

        if self.use_spin:
            pdos_gauss = np.append(system.info["S"], pdos_gauss)

        if self.use_charge:
            pdos_gauss = np.append(system.info["q"], pdos_gauss)

        return pdos_gauss

    def get_number_of_features(self) -> int:
        if self.use_wacsf:
            return int(
                self.num_points
                + 1
                + self.n_g2
                + self.n_g4
                + self.use_charge
                + self.use_spin
            )
        else:
            return int(self.num_points + self.use_charge + self.use_spin)


def atoms_to_pyscf(atoms):
    # This converts system in ASE format into atom format for pySCF
    atom_list = []
    for atom in atoms:
        symbol = atom.symbol
        position = atom.position
        atom_list.append((symbol, tuple(position)))
    return atom_list


def spectrum(E, osc, sigma, x):
    # This Gaussian broadens the partial density of states over a defined
    # energy range and grid spacing.
    gE = []
    for Ei in x:
        tot = 0
        for Ej, os in zip(E, osc):
            tot += os * np.exp(-((((Ej - Ei) / sigma) ** 2)))
        gE.append(tot)
    return gE
