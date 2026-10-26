import numpy as np

if __package__:
    from .simulate import simulate_rc
else:
    from simulate import simulate_rc

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


if __name__ == "__main__":
    server.run()
