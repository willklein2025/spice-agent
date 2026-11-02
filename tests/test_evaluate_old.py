import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent import DEFAULT_MODEL, FilterDesign, run_agent
from evaluate import (
    append_result,
    count_tool_calls,
    evaluate,
    evaluate_case,
    final_values_in_tool_call,
    fixed_components_unchanged,
    load_cases,
)
from langchain_google_genai.chat_models import GoogleRateLimitError


class FakeMessage:
    type = "ai"

    def __init__(self, tool_calls, model_name=None) -> None:
        self.tool_calls = tool_calls
        self.response_metadata = (
            {"model_name": model_name} if model_name is not None else {}
        )


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
                    },
                    {"name": "FilterDesign", "args": {}},
                ],
                model_name="gemini-3.7-flash",
            ),
            FakeMessage([], model_name="gemini-3.5-flash-lite"),
            FakeMessage([], model_name="gemini-3.7-flash"),
        ]
        self.case = {
            "id": "test-case",
            "request": "Design a 2.5 kHz filter using a fixed 10 nF capacitor.",
            "target_cutoff_hz": 2500,
            "fixed_components": {"c_farads": 1e-8},
        }

    def test_grades_cutoff_tool_values_and_fixed_component(self) -> None:
        with (
            patch(
                "evaluate.run_agent",
                new=AsyncMock(return_value=(self.design, self.messages)),
            ) as run_agent_mock,
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
        self.assertEqual(
            result["model_names"],
            ["gemini-3.5-flash-lite", "gemini-3.7-flash"],
        )
        run_agent_mock.assert_awaited_once_with(
            self.case["request"], "test-model", fallback_model=None
        )

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

    def test_missing_measured_cutoff_fails_without_using_theoretical_value(self) -> None:
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(self.design, self.messages))),
            patch(
                "evaluate.simulate_rc",
                return_value={
                    "fc_measured_hz": None,
                    "fc_theoretical_hz": 2500,
                },
            ),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))

        self.assertFalse(result["checks"]["cutoff_within_2_percent"])
        self.assertFalse(result["passed"])
        self.assertIsNone(result["simulated_cutoff_hz"])
        self.assertIsNone(result["cutoff_error_fraction"])
        self.assertIn("measured cutoff", result["cutoff_measurement_error"])

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

    def test_rate_limit_is_raised_so_the_case_can_be_retried(self) -> None:
        with patch(
            "evaluate.run_agent",
            new=AsyncMock(side_effect=GoogleRateLimitError("daily limit")),
        ):
            with self.assertRaisesRegex(GoogleRateLimitError, "daily limit"):
                asyncio.run(evaluate_case(self.case, "test-model"))

    def test_appends_and_skips_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            results_path = root / "results.jsonl"
            cases_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            recorded_result = {
                "case_id": self.case["id"],
                "model": "test-model",
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
                "model": DEFAULT_MODEL,
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
            patch("agent.ModelFallbackMiddleware") as fallback_middleware,
            patch("agent.ModelRetryMiddleware") as retry_middleware,
        ):
            design, returned_messages = asyncio.run(
                run_agent("Design a filter.", "test-model")
            )

        self.assertEqual(design, self.design)
        self.assertEqual(returned_messages, messages)
        self.assertEqual(create_agent.call_args.kwargs["model"], "test-model")
        fallback_middleware.assert_not_called()
        self.assertEqual(
            create_agent.call_args.kwargs["middleware"], [retry_middleware.return_value]
        )
        agent.ainvoke.assert_awaited_once()

    def test_run_agent_uses_fallback_only_when_configured(self) -> None:
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={
                "structured_response": self.design,
                "messages": self.messages,
            }
        )
        client = MagicMock()
        client.get_tools = AsyncMock(return_value=[])

        with (
            patch("agent.MultiServerMCPClient", return_value=client),
            patch("agent.create_agent", return_value=agent) as create_agent,
            patch("agent.ModelFallbackMiddleware") as fallback_middleware,
            patch("agent.ModelRetryMiddleware") as retry_middleware,
        ):
            asyncio.run(
                run_agent(
                    "Design a filter.",
                    "primary-model",
                    fallback_model="fallback-model",
                )
            )

        fallback_middleware.assert_called_once_with("fallback-model")
        middleware = create_agent.call_args.kwargs["middleware"]
        self.assertEqual(
            middleware,
            [retry_middleware.return_value, fallback_middleware.return_value],
        )


if __name__ == "__main__":
    unittest.main()
