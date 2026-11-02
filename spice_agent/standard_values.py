"""Search standard component values (E12, E24, E96) for an RC low-pass filter.

The agent calls this through the find_standard_values MCP tool instead of hunting
for part combinations itself. Results are calculated with fc = 1 / (2*pi*R*C);
the agent still confirms its chosen design with a SPICE simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Standard mantissas for one decade (IEC 60063).
E_SERIES: dict[str, tuple[float, ...]] = {
    "E12": (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2),
    "E24": (
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
        3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
    ),
    "E96": (
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
        1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
        1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
        2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
        3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
        4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
        5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
        7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
    ),
}

# Practical ranges for discrete parts.
RESISTOR_RANGE_OHMS = (10.0, 10e6)
CAPACITOR_RANGE_FARADS = (1e-12, 10e-6)


@dataclass(frozen=True)
class Option:
    r_ohms: float
    capacitors_farads: tuple[float, ...]   # one part, or two in parallel
    cutoff_hz: float
    error_fraction: float                  # signed: (cutoff / target) - 1

    @property
    def c_farads(self) -> float:
        return sum(self.capacitors_farads)

    @property
    def part_count(self) -> int:
        return 1 + len(self.capacitors_farads)


def standard_values(series: str, low: float, high: float) -> list[float]:
    """Every value of an E series between low and high, inclusive."""
    try:
        mantissas = E_SERIES[series.upper()]
    except KeyError:
        raise ValueError(f"Unknown series {series!r}; use one of {', '.join(E_SERIES)}")
    values = []
    for exponent in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1):
        for mantissa in mantissas:
            value = float(f"{mantissa}e{exponent}")   # exact decimal, e.g. 4.7e-9
            if low * (1 - 1e-9) <= value <= high * (1 + 1e-9):
                values.append(value)
    return sorted(set(values))


def is_standard_value(value: float, series: str) -> bool:
    """True if value is (to 0.1%) a member of the given E series."""
    if value <= 0 or not math.isfinite(value):
        return False
    mantissa = value / 10.0 ** math.floor(math.log10(value))
    return any(
        math.isclose(mantissa, candidate, rel_tol=1e-3)
        for candidate in E_SERIES[series.upper()] + (10.0,)
    )


def cutoff_hz(r_ohms: float, c_farads: float) -> float:
    return 1.0 / (2.0 * math.pi * r_ohms * c_farads)


def find_options(
    target_cutoff_hz: float,
    fixed_r_ohms: float | None = None,
    fixed_c_farads: float | None = None,
    resistor_series: str = "E96",
    capacitor_series: str = "E12",
    allow_parallel_capacitors: bool = True,
    tolerance_fraction: float = 0.02,
    max_results: int = 3,
) -> tuple[list[Option], bool]:
    """Return the best standard-value designs and whether any meets the tolerance.

    Ranking: designs that meet the tolerance come first, fewest parts first, then
    smallest error. If none meets it, the closest designs are returned instead.
    """
    if not (math.isfinite(target_cutoff_hz) and target_cutoff_hz > 0):
        raise ValueError("target_cutoff_hz must be a positive number")
    if fixed_r_ohms is not None and fixed_c_farads is not None:
        raise ValueError("Fix at most one of R and C; with both fixed there is nothing to choose")
    for name, value in (("fixed_r_ohms", fixed_r_ohms), ("fixed_c_farads", fixed_c_farads)):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive number")

    resistors = (
        [fixed_r_ohms]
        if fixed_r_ohms is not None
        else standard_values(resistor_series, *RESISTOR_RANGE_OHMS)
    )
    if fixed_c_farads is not None:
        capacitor_sets = [(fixed_c_farads,)]
    else:
        singles = standard_values(capacitor_series, *CAPACITOR_RANGE_FARADS)
        capacitor_sets = [(c,) for c in singles]
        if allow_parallel_capacitors:
            capacitor_sets += [
                (big, small)
                for i, big in enumerate(singles)
                for small in singles[:i + 1]
                if small >= big / 1000           # skip pairs where one part is negligible
            ]

    options = []
    for r_ohms in resistors:
        for caps in capacitor_sets:
            fc = cutoff_hz(r_ohms, sum(caps))
            options.append(Option(r_ohms, caps, fc, fc / target_cutoff_hz - 1))

    meeting = [o for o in options if abs(o.error_fraction) <= tolerance_fraction]
    if meeting:
        # Fewest parts first; among equals, smallest error, then a practical R.
        meeting.sort(key=lambda o: (o.part_count, _rounded_error(o), _impracticality(o)))
        return _distinct(meeting, max_results), True
    options.sort(key=lambda o: (_rounded_error(o), o.part_count, _impracticality(o)))
    return _distinct(options, max_results), False


PRACTICAL_R_OHMS = 10e3   # RC filters usually use roughly 1 kohm to 100 kohm


def _rounded_error(option: Option) -> float:
    return round(abs(option.error_fraction), 5)


def _impracticality(option: Option) -> float:
    """How far R is from the practical middle (decades away from 10 kohm)."""
    return abs(math.log10(option.r_ohms / PRACTICAL_R_OHMS))


def _distinct(ranked: list[Option], count: int) -> list[Option]:
    """Keep the first of options that are the same design scaled by a decade
    (e.g. 4.42 kohm/12 nF and 44.2 kohm/1.2 nF give the same cutoff)."""
    chosen: list[Option] = []
    seen: set[float] = set()
    for option in ranked:
        signature = round(option.cutoff_hz, 6 - int(math.floor(math.log10(option.cutoff_hz))))
        if signature in seen:
            continue
        seen.add(signature)
        chosen.append(option)
        if len(chosen) == count:
            break
    return chosen


def format_value(value: float, unit: str) -> str:
    """Engineering notation, e.g. 4.7e-09 F -> '4.7 nF', 10200 ohm -> '10.2 kohm'."""
    prefixes = [(1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")]
    for scale, prefix in prefixes:
        if abs(value) >= scale * 0.9995:
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.4g} {unit}"


def describe_options(
    target_cutoff_hz: float,
    options: list[Option],
    meets_tolerance: bool,
    tolerance_fraction: float,
    resistor_series: str,
    capacitor_series: str,
    fixed_r: bool,
    fixed_c: bool,
) -> str:
    """Compact text summary for the agent."""
    lines = [
        f"Target cutoff {target_cutoff_hz:.6g} Hz, tolerance {tolerance_fraction:.1%}. "
        f"Resistors {'fixed' if fixed_r else resistor_series}, "
        f"capacitors {'fixed' if fixed_c else capacitor_series}."
    ]
    if not meets_tolerance:
        lines.append(
            "NO standard-value design meets the tolerance with these constraints. "
            "Closest options:"
        )
    else:
        lines.append("Options that meet the tolerance (fewest parts first):")
    for number, option in enumerate(options, start=1):
        if len(option.capacitors_farads) == 1:
            c_text = format_value(option.c_farads, "F")
        else:
            parts = " + ".join(format_value(c, "F") for c in option.capacitors_farads)
            c_text = f"{parts} in parallel = {format_value(option.c_farads, 'F')}"
        lines.append(
            f"{number}. R = {format_value(option.r_ohms, 'ohm')} ({option.r_ohms:.6g} ohms), "
            f"C = {c_text} ({option.c_farads:.6g} F): calculated cutoff "
            f"{option.cutoff_hz:.6g} Hz ({option.error_fraction:+.2%})."
        )
    lines.append(
        "These cutoffs are calculated, not simulated. Simulate the chosen design "
        "with simulate_rc_lowpass using the exact r_ohms and c_farads shown in parentheses."
    )
    return "\n".join(lines)
