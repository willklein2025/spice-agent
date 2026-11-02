import math
import unittest

from spice_agent.server import find_standard_values
from spice_agent.standard_values import (
    E_SERIES,
    find_options,
    is_standard_value,
    standard_values,
)


class StandardValueTableTests(unittest.TestCase):
    def test_series_have_the_standard_number_of_values(self) -> None:
        self.assertEqual({name: len(values) for name, values in E_SERIES.items()},
                         {"E12": 12, "E24": 24, "E96": 96})

    def test_values_are_exact_decimals(self) -> None:
        capacitors = standard_values("E12", 1e-9, 1e-8)
        self.assertIn(4.7e-9, capacitors)          # not 4.7000000000000004e-09
        self.assertEqual(capacitors[0], 1e-9)
        self.assertEqual(capacitors[-1], 1e-8)

    def test_recognizes_standard_values(self) -> None:
        self.assertTrue(is_standard_value(7870, "E96"))
        self.assertFalse(is_standard_value(7960, "E96"))     # claimed "E96" in a real run
        self.assertTrue(is_standard_value(560e-12, "E12"))
        self.assertFalse(is_standard_value(580e-12, "E12"))  # suggested in a real run
        self.assertFalse(is_standard_value(620e-12, "E12"))
        self.assertTrue(is_standard_value(620e-12, "E24"))

    def test_rejects_unknown_series(self) -> None:
        with self.assertRaises(ValueError):
            standard_values("E7", 1, 10)


class FindOptionsTests(unittest.TestCase):
    """Answers worked out by hand earlier in the project."""

    def test_fixed_capacitor_picks_e96_resistor(self) -> None:
        options, ok = find_options(3300, fixed_c_farads=4.7e-9)
        self.assertTrue(ok)
        self.assertEqual(options[0].r_ohms, 10200)
        self.assertAlmostEqual(options[0].error_fraction, 0.0060, places=4)

    def test_fixed_resistor_uses_parallel_capacitors_when_needed(self) -> None:
        options, ok = find_options(150, fixed_r_ohms=47000)
        self.assertTrue(ok)
        self.assertEqual(sorted(options[0].capacitors_farads), [560e-12, 22e-9])
        self.assertTrue(math.isclose(options[0].c_farads, 22.56e-9))

    def test_wordy_fixed_resistor_case(self) -> None:
        options, ok = find_options(7000, fixed_r_ohms=2200)
        self.assertTrue(ok)
        self.assertEqual(sorted(options[0].capacitors_farads), [330e-12, 10e-9])

    def test_single_part_preferred_when_it_meets_tolerance(self) -> None:
        options, ok = find_options(2e6, fixed_c_farads=1e-11)
        self.assertTrue(ok)
        self.assertEqual(options[0].r_ohms, 7870)
        self.assertEqual(options[0].part_count, 2)

    def test_e12_only_is_infeasible_and_reports_closest(self) -> None:
        options, ok = find_options(
            1200, resistor_series="E12", capacitor_series="E12",
            allow_parallel_capacitors=False,
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(abs(options[0].error_fraction), 0.0305, places=4)
        self.assertEqual((options[0].r_ohms, options[0].c_farads), (3900, 33e-9))

    def test_unconstrained_design_uses_practical_resistor(self) -> None:
        options, ok = find_options(3000)
        self.assertTrue(ok)
        for option in options:
            self.assertTrue(1e3 <= option.r_ohms <= 100e3, option)

    def test_no_scaled_duplicates(self) -> None:
        options, _ = find_options(3000)
        cutoffs = [round(o.cutoff_hz, 3) for o in options]
        self.assertEqual(len(cutoffs), len(set(cutoffs)))

    def test_rejects_both_fixed_and_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            find_options(1000, fixed_r_ohms=1000, fixed_c_farads=1e-7)
        with self.assertRaises(ValueError):
            find_options(-5)


class FindStandardValuesToolTests(unittest.TestCase):
    def test_tool_lists_options_with_exact_values_to_simulate(self) -> None:
        text = find_standard_values(7000, fixed_r_ohms=2200)
        self.assertIn("10 nF + 330 pF in parallel", text)
        self.assertIn("(1.033e-08 F)", text)
        self.assertIn("calculated, not simulated", text)

    def test_tool_says_when_nothing_meets_tolerance(self) -> None:
        text = find_standard_values(
            1200, resistor_series="E12", capacitor_series="E12",
            allow_parallel_capacitors=False,
        )
        self.assertIn("NO standard-value design meets the tolerance", text)

    def test_tool_returns_message_instead_of_crashing(self) -> None:
        text = find_standard_values(1000, fixed_r_ohms=1000, fixed_c_farads=1e-7)
        self.assertTrue(text.startswith("Invalid request"))


if __name__ == "__main__":
    unittest.main()
