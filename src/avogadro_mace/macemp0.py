#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""MACE-MP-0 energy and gradient calculator for inorganic crystals."""

import json
import sys

import numpy as np
from ase import Atoms
from ase.cell import Cell
import torch
from mace.calculators import mace_mp

from .energy import EnergyServer

# don't use MPS on Apple yet
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# eV → kJ/mol (energy and gradient use the same conversion factor)
_EV_TO_KJ_MOL = 96.4853321


def run():
    bootstrap = json.loads(sys.stdin.buffer.readline())
    mol_cjson = bootstrap["cjson"]

    atom_numbers = np.array(mol_cjson["atoms"]["elements"]["number"])
    coord_list = mol_cjson["atoms"]["coords"]["3d"]
    coordinates = np.array(coord_list, dtype=float).reshape(-1, 3)
    # num_atoms must match what C++ sends in each binary frame
    num_atoms = len(atom_numbers)

    # Build a boolean mask of the unique (non-image) atoms.
    # Avogadro may send periodic image atoms whose fractional coordinates are
    # ~1.0 (equivalent to 0.0 under PBC).  Keep only atoms where every
    # fractional coordinate is strictly less than 1 - tol.
    keep = np.ones(num_atoms, dtype=bool)

    if "unitCell" in mol_cjson:
        frac_list = mol_cjson["atoms"]["coords"].get("3dFractional", [])
        if frac_list:
            frac = np.array(frac_list, dtype=float).reshape(-1, 3)
            keep = ~np.any(np.abs(frac - 1.0) < 1e-4, axis=1)

        cell = mol_cjson["unitCell"]
        lattice = np.array(cell["cellVectors"]).reshape(3, 3)
        atoms = Atoms(
            atom_numbers[keep], coordinates[keep],
            cell=Cell(lattice), pbc=True
        )
    else:
        atoms = Atoms(atom_numbers, coordinates)
        atoms.pbc = False

    calc = mace_mp(default_dtype="float32", device=_DEVICE)
    atoms.calc = calc

    with EnergyServer(sys.stdin.buffer, sys.stdout.buffer, num_atoms) as server:
        for request in server.requests():
            # request.coords has shape (num_atoms, 3) — full molecule
            atoms.set_positions(request.coords[keep])

            if request.wants_gradient:
                # ASE forces = -gradient; shape (n_unique, 3)
                forces = atoms.get_forces()
                # Expand back to full atom count; image atoms get zero gradient
                grad_full = np.zeros((num_atoms, 3), dtype=np.float64)
                grad_full[keep] = forces * -_EV_TO_KJ_MOL
                request.send_gradient(grad_full)
            else:
                energy = float(atoms.get_potential_energy()) * _EV_TO_KJ_MOL
                request.send_energy(energy)
