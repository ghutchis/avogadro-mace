"""Protocol tests for avogadro-mace.

These run the real entry point in a subprocess and speak the same binary
protocol Avogadro does. The point is to catch anything written to stdout
(mace prints notices on import and while loading weights) ahead of or between
response frames: Avogadro requires every response to start with the AVB1 magic
and does not resynchronize. The first run downloads the model weights.
"""

import io
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from avogadro.energy import (
    FLAG_REQUEST_ENERGY_AND_GRADIENT,
    FLAG_REQUEST_GRADIENT,
    read_frame,
    write_frame,
)

# python -c leaves sys.argv[1:] as the arguments after the code.
_LAUNCH = [sys.executable, "-c", "from avogadro_mace import main; main()"]

_WATER = np.array(
    [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
)


def _bootstrap():
    cjson = {
        "atoms": {
            "elements": {"number": [8, 1, 1]},
            "coords": {"3d": _WATER.reshape(-1).tolist()},
        }
    }
    return {"cjson": cjson, "charge": 0, "spin": 1}


def _request(coords, flags=0):
    buffer = io.BytesIO()
    write_frame(
        buffer, flags, len(coords), np.ascontiguousarray(coords, "<f8").tobytes()
    )
    return buffer.getvalue()


@pytest.mark.parametrize("identifier", ["MACE-OFF23", "MACE-MP-0"])
def test_stdout_carries_only_frames_and_gradient_matches_energy(identifier):
    step = 0.01
    shifted = []
    for sign in (+1, -1):
        coords = _WATER.copy()
        coords[1, 1] += sign * step
        shifted.append(coords)

    stdin = json.dumps(_bootstrap()).encode() + b"\n"
    stdin += _request(shifted[0])
    stdin += _request(shifted[1])
    stdin += _request(_WATER, FLAG_REQUEST_GRADIENT)
    stdin += _request(_WATER, FLAG_REQUEST_ENERGY_AND_GRADIENT)

    process = subprocess.run(
        _LAUNCH + [identifier, "--protocol", "binary-v1"],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", "replace")

    # The very first bytes must be a frame, not a notice from mace.
    assert process.stdout[:4] == b"AVB1", process.stdout[:200]

    stream = io.BytesIO(process.stdout)
    plus = struct.unpack("<d", read_frame(stream).payload)[0]
    minus = struct.unpack("<d", read_frame(stream).payload)[0]
    gradient = np.frombuffer(read_frame(stream).payload, "<f8").reshape(3, 3)
    fused = read_frame(stream).payload
    energy = struct.unpack("<d", fused[:8])[0]
    fused_gradient = np.frombuffer(fused[8:], "<f8").reshape(3, 3)
    assert not stream.read(), "unexpected bytes after the last frame"

    assert np.isfinite([plus, minus, energy]).all()
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(fused_gradient, gradient, atol=1e-3)

    # Avogadro wants dE/dx in kJ/mol/Angstrom; compare with finite differences.
    numeric = (plus - minus) / (2 * step)
    assert gradient[1, 1] == pytest.approx(numeric, rel=0.1, abs=5.0)
