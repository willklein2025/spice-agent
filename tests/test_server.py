import unittest
from unittest.mock import patch

import numpy as np

from spice_agent.server import simulate_rc_lowpass


class SimulateRCLowpassToolTests(unittest.TestCase):
    @patch("spice_agent.server.simulate_rc")
    def test_returns_cutoff_and_at_most_three_gain_samples(self, simulate_rc) -> None:
        simulate_rc.return_value = {
            "frequency_hz": np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0]),
            "gain_db": np.array([-0.01, -0.1, -3.0, -20.0, -40.0]),
            "phase_deg": np.zeros(5),
            "fc_measured_hz": 1000.0,
            "fc_theoretical_hz": 1000.0,
        }

        summary = simulate_rc_lowpass(15915.5, 1e-8)

        simulate_rc.assert_called_once_with(15915.5, 1e-8)
        self.assertTrue(summary.startswith("VERIFIED BY SIMULATION: measured cutoff 1000 Hz"))
        self.assertIn("(formula predicts 1000 Hz)", summary)
        self.assertIn("100 Hz: -0.100 dB", summary)
        self.assertIn("1000 Hz: -3.000 dB", summary)
        self.assertIn("10000 Hz: -20.000 dB", summary)
        self.assertNotIn("100000 Hz", summary)

    @patch("spice_agent.server.simulate_rc")
    def test_reports_when_cutoff_gain_samples_are_outside_the_sweep(
        self, simulate_rc
    ) -> None:
        simulate_rc.return_value = {
            "frequency_hz": np.array([10.0, 100.0]),
            "gain_db": np.array([-0.1, -3.0]),
            "phase_deg": np.zeros(2),
            "fc_measured_hz": None,
            "fc_theoretical_hz": 10000.0,
        }

        summary = simulate_rc_lowpass(1.0, 1.0)

        self.assertTrue(summary.startswith("NOT VERIFIED"))
        self.assertIn("simulated range (10 Hz to 100 Hz)", summary)
        self.assertIn("Calculated (formula) cutoff: 10000 Hz. This value was NOT simulated", summary)
        self.assertNotIn("VERIFIED BY SIMULATION", summary)
        self.assertIn(
            "No gain sample frequencies fell within the simulated sweep.", summary
        )


@unittest.skipUnless(__import__("shutil").which("ngspice"), "ngspice is not installed")
class SimulateToolWithNgspiceTests(unittest.TestCase):
    def test_cutoff_below_sweep_is_reported_as_not_verified(self) -> None:
        summary = simulate_rc_lowpass(5360, 1e-5)        # about 3 Hz, below the 10 Hz sweep start
        self.assertTrue(summary.startswith("NOT VERIFIED"))
        self.assertIn("Calculated (formula) cutoff: 2.96931 Hz", summary)

    def test_normal_design_is_verified(self) -> None:
        summary = simulate_rc_lowpass(6340, 1e-8)
        self.assertTrue(summary.startswith("VERIFIED BY SIMULATION: measured cutoff 2510.33 Hz"))


if __name__ == "__main__":
    unittest.main()
