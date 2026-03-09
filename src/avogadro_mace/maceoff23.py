#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""MACE-OFF23 energy and gradient calculator for organic molecules."""

import json
import sys

import torch
from mace.calculators import mace_off

from ._mace_server import run_mace_server

# don't use MPS on Apple yet
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    bootstrap = json.loads(sys.stdin.buffer.readline())
    calc = mace_off(model="medium", default_dtype="float32", device=_DEVICE)
    run_mace_server(bootstrap["cjson"], calc)
