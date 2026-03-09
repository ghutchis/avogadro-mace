#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""MACE-MP-0 energy and gradient calculator for inorganic crystals."""

import json
import sys

import torch
from mace.calculators import mace_mp

from ._mace_server import run_mace_server

# don't use MPS on Apple yet
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    bootstrap = json.loads(sys.stdin.buffer.readline())
    calc = mace_mp(default_dtype="float32", device=_DEVICE)
    run_mace_server(bootstrap["cjson"], calc)
