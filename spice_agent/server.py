import numpy as np

if __package__:
    from .simulate import simulate_rc
    from .standard_values import describe_options, find_options
else:
    from simulate import simulate_rc
    from standard_values import describe_options, find_options

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as error:
    if error.name != "mcp.server.fastmcp":
        raise
    from mcp.server.mcpserver import MCPServer as FastMCP

server = FastMCP(
    name="spice-agent",
    instructions="Tools for simulating simple electronic circuits.",
)


@server.tool(
    name="simulate_rc_lowpass",
    description=(
        "Simulate an RC low-pass filter and return its cutoff frequency, "
        "plus gain at up to three frequencies around the cutoff. "
        "Provide resistance in ohms and capacitance in farads."
    ),
)
def simulate_rc_lowpass(r_ohms: float, c_farads: float) -> str:
    """Return a compact summary of an RC low-pass filter simulation."""
    response = simulate_rc(r_ohms, c_farads)
    cutoff_hz = response["fc_theoretical_hz"]
    measured_cutoff_hz = response["fc_measured_hz"]
    frequencies = response["frequency_hz"]
    gains_db = response["gain_db"]

    summary = [
        f"Theoretical cutoff: {cutoff_hz:.6g} Hz.",
        (
            f"Measured cutoff: {measured_cutoff_hz:.6g} Hz."
            if measured_cutoff_hz is not None
            else "Measured cutoff: unavailable."
        ),
    ]

    sample_frequencies = (0.1 * cutoff_hz, cutoff_hz, 10 * cutoff_hz)
    samples = [
        (
            frequency,
            float(np.interp(np.log10(frequency), np.log10(frequencies), gains_db)),
        )
        for frequency in sample_frequencies
        if frequencies[0] <= frequency <= frequencies[-1]
    ]
    if samples:
        gain_summary = "; ".join(
            f"{frequency:.6g} Hz: {gain_db:.3f} dB"
            for frequency, gain_db in samples
        )
        summary.append(f"Gain samples: {gain_summary}.")
    else:
        summary.append("No gain sample frequencies fell within the simulated sweep.")

    return " ".join(summary)


@server.tool(
    name="find_standard_values",
    description=(
        "Find RC low-pass designs built from standard component values. Give the "
        "target cutoff in Hz and, optionally, a resistor (ohms) or capacitor (farads) "
        "the design must keep. Returns up to three options ranked by fewest parts, "
        "then smallest error, each with its calculated (not simulated) cutoff, or "
        "says that no standard-value design meets the tolerance. Series: 'E12', "
        "'E24' or 'E96'. Two capacitors in parallel are allowed unless "
        "allow_parallel_capacitors is false. Call this before choosing parts, then "
        "confirm the chosen design with simulate_rc_lowpass."
    ),
)
def find_standard_values(
    target_cutoff_hz: float,
    fixed_r_ohms: float | None = None,
    fixed_c_farads: float | None = None,
    resistor_series: str = "E96",
    capacitor_series: str = "E12",
    allow_parallel_capacitors: bool = True,
    tolerance_percent: float = 2.0,
) -> str:
    """Return a compact list of standard-value design options."""
    tolerance_fraction = tolerance_percent / 100
    try:
        options, meets_tolerance = find_options(
            target_cutoff_hz,
            fixed_r_ohms=fixed_r_ohms,
            fixed_c_farads=fixed_c_farads,
            resistor_series=resistor_series,
            capacitor_series=capacitor_series,
            allow_parallel_capacitors=allow_parallel_capacitors,
            tolerance_fraction=tolerance_fraction,
        )
    except ValueError as error:
        return f"Invalid request: {error}"   # let the agent correct itself
    return describe_options(
        target_cutoff_hz,
        options,
        meets_tolerance,
        tolerance_fraction,
        resistor_series,
        capacitor_series,
        fixed_r=fixed_r_ohms is not None,
        fixed_c=fixed_c_farads is not None,
    )


if __name__ == "__main__":
    server.run()
