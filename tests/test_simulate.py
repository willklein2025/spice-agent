import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spice_agent.simulate import simulate_rc


class SimulateRCTests(unittest.TestCase):
    @patch("spice_agent.simulate.shutil.which", return_value="/usr/bin/ngspice")
    @patch("spice_agent.simulate.subprocess.run")
    def test_returns_parsed_ac_response(self, run, which) -> None:
        def write_response(command, *, cwd, capture_output, text, timeout):
            netlist_path = Path(command[-1])
            deck = netlist_path.read_text()
            self.assertIn(".param RVAL=1000 CVAL=1e-06", deck)
            self.assertTrue(capture_output)
            self.assertTrue(text)
            self.assertEqual(timeout, 30)
            np.savetxt(
                Path(cwd) / "ac_response.dat",
                [[10.0, -0.1, -0.2], [100.0, -3.0, -0.8]],
                header="frequency vdb vp",
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="fc = 1591.55\n", stderr=""
            )

        run.side_effect = write_response

        response = simulate_rc(1000, 1e-6)

        which.assert_called_once_with("ngspice")
        np.testing.assert_array_equal(response["frequency_hz"], [10.0, 100.0])
        np.testing.assert_array_equal(response["gain_db"], [-0.1, -3.0])
        np.testing.assert_allclose(response["phase_deg"], np.degrees([-0.2, -0.8]))
        self.assertEqual(response["fc_measured_hz"], 1591.55)
        self.assertAlmostEqual(response["fc_theoretical_hz"], 1 / (2 * np.pi * 1e-3))

    def test_rejects_nonpositive_or_nonfinite_values(self) -> None:
        for resistance, capacitance in (
            (0, 1e-6),
            (-1, 1e-6),
            (float("inf"), 1e-6),
            (1000, 0),
            (1000, float("nan")),
        ):
            with self.subTest(resistance=resistance, capacitance=capacitance):
                with self.assertRaises(ValueError):
                    simulate_rc(resistance, capacitance)

    @patch("spice_agent.simulate.shutil.which", return_value=None)
    def test_reports_missing_ngspice(self, which) -> None:
        with self.assertRaisesRegex(RuntimeError, "ngspice"):
            simulate_rc(2200, 1e-7)
        which.assert_called_once_with("ngspice")

    @unittest.skipUnless(shutil.which("ngspice"), "ngspice is not installed")
    def test_measured_cutoff_is_within_one_percent_of_theoretical(self) -> None:
        component_pairs = (
            (1000, 1e-6),
            (4700, 1e-7),
            (10000, 1e-8),
        )
        for resistance, capacitance in component_pairs:
            with self.subTest(resistance=resistance, capacitance=capacitance):
                response = simulate_rc(resistance, capacitance)
                measured_cutoff = response["fc_measured_hz"]
                if measured_cutoff is None:
                    self.fail("ngspice did not report the measured cutoff")
                self.assertLessEqual(
                    abs(measured_cutoff / response["fc_theoretical_hz"] - 1),
                    0.01,
                )

    @unittest.skipUnless(shutil.which("ngspice"), "ngspice is not installed")
    def test_gain_is_about_minus_twenty_db_at_ten_times_cutoff(self) -> None:
        response = simulate_rc(1000, 1e-6)
        target_frequency = 10 * response["fc_theoretical_hz"]
        gain_db = np.interp(
            np.log10(target_frequency),
            np.log10(response["frequency_hz"]),
            response["gain_db"],
        )
        self.assertAlmostEqual(gain_db, -20, delta=0.5)


if __name__ == "__main__":
    unittest.main()
