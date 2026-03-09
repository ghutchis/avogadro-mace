#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Entry point for the avogadro-mace plugin.

Avogadro calls this as:
    avogadro-mace <identifier> [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin (one compact JSON line).
"""

import argparse


def main():
    parser = argparse.ArgumentParser("avogadro-mace")
    parser.add_argument("feature")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    args = parser.parse_args()

    match args.feature:
        case "MACE-MP-0":
            from .macemp0 import run
            run()
        case "MACE-OFF23":
            from .maceoff23 import run
            run()
