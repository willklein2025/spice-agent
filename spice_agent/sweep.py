import argparse
import itertools
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from spice_agent.simulate import simulate_rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep RC low-pass filter component values using ngspice."
    )
    parser.add_argument(
        "-r",
        "--resistors",
        type=float,
        nargs="+",
        required=True,
        help="One or more resistor values in ohms (e.g., -r 1000 4700 10000)",
    )
    parser.add_argument(
        "-c",
        "--capacitors",
        type=float,
        nargs="+",
        required=True,
        help="One or more capacitor values in farads (e.g., -c 1e-6 10e-6)",
    )
    args = parser.parse_args()

    if shutil.which("ngspice") is None:
        raise RuntimeError("ngspice isn't installed or isn't on PATH")

    fig, (magnitude_ax, phase_ax) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True
    )
    combinations = list(itertools.product(args.resistors, args.capacitors))
    print(f"Starting sweep of {len(combinations)} component combination(s)...\n")

    for r_ohms, c_farads in combinations:
        if (
            r_ohms <= 0
            or c_farads <= 0
            or not math.isfinite(r_ohms)
            or not math.isfinite(c_farads)
        ):
            print(f"Skipping invalid pair: R={r_ohms}, C={c_farads}")
            continue

        try:
            response = simulate_rc(r_ohms, c_farads)
        except RuntimeError as error:
            print(f"Simulation failed for R={r_ohms}, C={c_farads}: {error}")
            continue

        fc_measured = response["fc_measured_hz"]
        fc_theoretical = response["fc_theoretical_hz"]
        run_label = (
            f"R={r_ohms:g}Ω, C={c_farads:g}F "
            f"(fc={fc_theoretical:.1f}Hz)"
        )
        print(
            f"Processed: R={r_ohms:g} Ω, C={c_farads:g} F | "
            f"Theoretical fc: {fc_theoretical:.2f} Hz | "
            f"Simulated fc: {fc_measured}"
        )

        line, = magnitude_ax.semilogx(
            response["frequency_hz"], response["gain_db"], label=run_label
        )
        color = line.get_color()
        phase_ax.semilogx(
            response["frequency_hz"], response["phase_deg"], color=color
        )

        if fc_measured:
            magnitude_ax.plot(fc_measured, -3.0, marker="o", color=color)
            phase_ax.plot(fc_measured, -45.0, marker="o", color=color)

    magnitude_ax.set_ylabel("Magnitude (dB)")
    magnitude_ax.set_title("RC Low-Pass Filter Parameter Sweep")
    magnitude_ax.grid(True, which="both")
    magnitude_ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0), borderaxespad=0)

    phase_ax.set_xlabel("Frequency (Hz)")
    phase_ax.set_ylabel("Phase (degrees)")
    phase_ax.grid(True, which="both")

    fig.tight_layout()
    output_path = Path("rc_sweep.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
