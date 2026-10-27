import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent import FilterDesign, run_agent
from evaluate import (
    append_result,
    count_tool_calls,
    evaluate,
    evaluate_case,
    final_values_in_tool_call,
    fixed_components_unchanged,
    load_cases,
)


class FakeMessage:
    type = "ai"

    def __init__(self, tool_calls) -> None:
        self.tool_calls = tool_calls


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = FilterDesign(
            r_ohms=6366.2,
            c_farads=1e-8,
            simulated_cutoff_hz=2500,
            notes="Chosen to meet the target.",
        )
        self.messages = [
            FakeMessage(
                [
                    {
                        "name": "simulate_rc_lowpass",
                        "args": {"r_ohms": 6366.2, "c_farads": 1e-8},
                    }
                ]
            )
        ]
        self.case = {
            "id": "test-case",
            "request": "Design a 2.5 kHz filter using a fixed 10 nF capacitor.",
            "target_cutoff_hz": 2500,
            "fixed_components": {"c_farads": 1e-8},
        }

    def test_grades_cutoff_tool_values_and_fixed_component(self) -> None:
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(self.design, self.messages))),
            patch(
                "evaluate.simulate_rc",
                return_value={
                    "fc_measured_hz": 2500.5,
                    "fc_theoretical_hz": 2500,
                },
            ),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))

        self.assertTrue(result["passed"])
        self.assertEqual(result["tool_call_count"], 1)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["model"], "test-model")

    def test_final_values_must_appear_in_one_tool_call(self) -> None:
        self.assertEqual(count_tool_calls(self.messages), 1)
        self.assertTrue(final_values_in_tool_call(self.design, self.messages))

        wrong_call = [
            FakeMessage(
                [
                    {
                        "name": "simulate_rc_lowpass",
                        "args": {"r_ohms": 6366.2, "c_farads": 2e-8},
                    }
                ]
            )
        ]
        self.assertFalse(final_values_in_tool_call(self.design, wrong_call))

    def test_checks_fixed_values(self) -> None:
        self.assertTrue(fixed_components_unchanged(self.case, self.design))
        changed_design = self.design.model_copy(update={"c_farads": 2e-8})
        self.assertFalse(fixed_components_unchanged(self.case, changed_design))

    def test_records_errors_and_continues_case_processing(self) -> None:
        with patch(
            "evaluate.run_agent",
            new=AsyncMock(side_effect=RuntimeError("model unavailable")),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertFalse(result["passed"])
        self.assertIn("model unavailable", result["error"])
        self.assertEqual(result["tool_call_count"], 0)

    def test_appends_and_skips_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            results_path = root / "results.jsonl"
            cases_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            recorded_result = {
                "case_id": self.case["id"],
                "passed": True,
                "tool_call_count": 1,
                "seconds": 0.5,
            }
            append_result(recorded_result, results_path)
            self.assertEqual(load_cases(cases_path), [self.case])
            with patch("evaluate.run_agent", new=AsyncMock()) as run_agent:
                results = asyncio.run(
                    evaluate(
                        model="test-model",
                        cases_path=cases_path,
                        results_path=results_path,
                    )
                )

        self.assertEqual(results, [recorded_result])
        run_agent.assert_not_awaited()

    def test_summarizes_previously_recorded_cases_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            results_path = root / "results.jsonl"
            cases_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            recorded_result = {
                "case_id": self.case["id"],
                "model": "previous-model",
                "passed": True,
                "tool_call_count": 2,
                "seconds": 1.5,
            }
            append_result(recorded_result, results_path)

            results = asyncio.run(
                evaluate(cases_path=cases_path, results_path=results_path)
            )

        self.assertEqual(results, [recorded_result])

    def test_run_agent_returns_structured_design_and_messages(self) -> None:
        messages = [FakeMessage([])]
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={
                "structured_response": self.design,
                "messages": messages,
            }
        )
        client = MagicMock()
        client.get_tools = AsyncMock(return_value=[])

        with (
            patch("agent.MultiServerMCPClient", return_value=client),
            patch("agent.create_agent", return_value=agent) as create_agent,
            patch("agent.ModelFallbackMiddleware"),
        ):
            design, returned_messages = asyncio.run(
                run_agent("Design a filter.", "test-model")
            )

        self.assertEqual(design, self.design)
        self.assertEqual(returned_messages, messages)
        self.assertEqual(create_agent.call_args.kwargs["model"], "test-model")
        agent.ainvoke.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
