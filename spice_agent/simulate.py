import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

import numpy as np


class SimulationResult(TypedDict):
    frequency_hz: np.ndarray
    gain_db: np.ndarray
    phase_deg: np.ndarray
    fc_measured_hz: float | None
    fc_theoretical_hz: float


def _is_unmeasurable_cutoff(line: str) -> bool:
    """ngspice reports a cutoff outside the AC sweep as a measure 'error'.

    That is not a failed simulation: the response is still valid, the -3 dB point
    just lies outside the swept range, so fc_measured_hz is returned as None.
    """
    lowered = line.lower()
    return "measure" in lowered and "out of interval" in lowered


def simulate_rc(r_ohms: float, c_farads: float) -> SimulationResult:
    """Simulate one RC low-pass filter and return its AC response."""
    if not math.isfinite(r_ohms) or r_ohms <= 0:
        raise ValueError("r_ohms must be a finite positive number")
    if not math.isfinite(c_farads) or c_farads <= 0:
        raise ValueError("c_farads must be a finite positive number")

    ngspice = shutil.which("ngspice")
    if ngspice is None:
        raise RuntimeError("ngspice isn't installed or isn't on PATH")

    netlist = Path(__file__).resolve().parents[1] / "circuits" / "rc_lowpass.cir"
    if not netlist.is_file():
        raise FileNotFoundError(f"Couldn't find RC low-pass netlist: {netlist}")

    deck_template = netlist.read_text()
    parameter_line = f".param RVAL={r_ohms:.12g} CVAL={c_farads:.12g}"
    deck, replacements = re.subn(
        r"(?im)^\.param\s+RVAL=.*$", parameter_line, deck_template
    )
    if replacements != 1:
        raise RuntimeError("Expected exactly one .param RVAL=... line in the netlist")

    with tempfile.TemporaryDirectory() as temp_dir:
        run_netlist = Path(temp_dir) / netlist.name
        run_netlist.write_text(deck)

        result = subprocess.run(
            [ngspice, "-b", str(run_netlist)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        error_lines = [
            f"{stream_name}: {line}"
            for stream_name, output in (
                ("stdout", result.stdout),
                ("stderr", result.stderr),
            )
            for line in output.splitlines()
            if "ERROR" in line.upper() and not _is_unmeasurable_cutoff(line)
        ]
        if error_lines:
            raise RuntimeError(
                f"ngspice reported an error for R={r_ohms}, C={c_farads}:\n"
                + "\n".join(error_lines)
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"ngspice failed for R={r_ohms}, C={c_farads} "
                f"(exit status {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )

        data_path = Path(temp_dir) / "ac_response.dat"
        if not data_path.is_file():
            raise RuntimeError("ngspice did not create ac_response.dat")
        data = np.loadtxt(data_path, skiprows=1, ndmin=2)

    if data.shape[1] < 3:
        raise RuntimeError(
            f"Expected at least 3 columns in ac_response.dat, got {data.shape[1]}"
        )

    match = re.search(
        r"^\s*fc\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        result.stdout,
        re.MULTILINE | re.IGNORECASE,
    )
    fc_measured = float(match.group(1)) if match else None

    return {
        "frequency_hz": data[:, 0],
        "gain_db": data[:, 1],
        "phase_deg": np.degrees(data[:, 2]),
        "fc_measured_hz": fc_measured,
        "fc_theoretical_hz": 1.0 / (2.0 * math.pi * r_ohms * c_farads),
    }
