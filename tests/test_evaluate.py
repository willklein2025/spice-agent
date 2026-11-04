import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_google_genai.chat_models import GoogleRateLimitError

from agent import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    AgentStoppedEarly,
    FilterDesign,
    build_chat_model,
    run_agent,
    should_retry,
)
from evaluate import (
    append_result,
    claimed_cutoff_is_honest,
    grade_design,
    count_tool_calls,
    evaluate,
    evaluate_case,
    fatal_error_hint,
    final_values_in_tool_call,
    fixed_components_unchanged,
    is_finished,
    load_cases,
    load_jsonl,
    total_tokens,
)


class FakeMessage:
    type = "ai"

    def __init__(self, tool_calls, model_name=None, tokens=None) -> None:
        self.tool_calls = tool_calls
        self.response_metadata = (
            {"model_name": model_name} if model_name is not None else {}
        )
        self.usage_metadata = {"total_tokens": tokens} if tokens is not None else None


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = FilterDesign(
            r_ohms=6366.2,
            c_farads=1e-8,
            simulated_cutoff_hz=2500,
            meets_spec=True,
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

        self.assertFalse(result["checks"]["cutoff_within_tolerance"])
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
                "prompt_version": PROMPT_VERSION,
                "run": 1,
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
                "prompt_version": PROMPT_VERSION,
                "run": 1,
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
        messages = self.messages           # includes a simulation matching self.design
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
            patch("agent.ModelCallLimitMiddleware") as call_limit_middleware,
        ):
            design, returned_messages = asyncio.run(
                run_agent("Design a filter.", "test-model")
            )

        self.assertEqual(design, self.design)
        self.assertEqual(returned_messages, messages)
        self.assertEqual(create_agent.call_args.kwargs["model"], "test-model")
        fallback_middleware.assert_not_called()
        self.assertEqual(
            create_agent.call_args.kwargs["middleware"],
            [call_limit_middleware.return_value, retry_middleware.return_value],
        )
        call_limit_middleware.assert_called_once_with(run_limit=8, exit_behavior="end")
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
            patch("agent.ModelCallLimitMiddleware") as call_limit_middleware,
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
            [
                call_limit_middleware.return_value,
                retry_middleware.return_value,
                fallback_middleware.return_value,
            ],
        )

    # --- Cost and runaway protection ---------------------------------------

    def test_call_limit_is_a_recorded_failure_not_an_error(self) -> None:
        with patch(
            "evaluate.run_agent",
            new=AsyncMock(side_effect=ModelCallLimitExceededError(8, 8, None, None)),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "model call limit reached")
        self.assertNotIn("error", result)
        self.assertTrue(is_finished(result))  # so it is not rerun and paid for again

    def test_slow_case_times_out_as_a_recorded_failure(self) -> None:
        async def never_finishes(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("evaluate.run_agent", new=never_finishes):
            result = asyncio.run(
                evaluate_case(self.case, "test-model", case_timeout_s=0.05)
            )
        self.assertFalse(result["passed"])
        self.assertIn("timed out", result["failure_reason"])
        self.assertTrue(is_finished(result))

    def test_fatal_errors_are_recognized(self) -> None:
        for message in (
            "402 RESOURCE_EXHAUSTED prepayment credits are depleted",
            "registry.ollama.ai/x does not support tools (status code: 400)",
            "Failed to connect to Ollama. Please check that Ollama is downloaded",
            "model 'qwen' not found (status code: 404)",
        ):
            with self.subTest(message=message):
                self.assertIsNotNone(fatal_error_hint(RuntimeError(message)))
        self.assertIsNotNone(fatal_error_hint(GoogleRateLimitError("daily limit")))
        self.assertIsNone(fatal_error_hint(RuntimeError("503 high demand")))

    def test_model_without_tool_support_stops_the_case(self) -> None:
        error = RuntimeError("model does not support tools (status code: 400)")
        with patch("evaluate.run_agent", new=AsyncMock(side_effect=error)):
            with self.assertRaisesRegex(RuntimeError, "does not support tools"):
                asyncio.run(evaluate_case(self.case, "test-model"))

    def test_evaluation_stops_cleanly_and_records_nothing_on_fatal_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            results_path = root / "results.jsonl"
            second_case = dict(self.case, id="second-case")
            cases_path.write_text(
                json.dumps(self.case) + "\n" + json.dumps(second_case) + "\n",
                encoding="utf-8",
            )
            error = RuntimeError("402 prepayment credits are depleted")
            with patch("evaluate.run_agent", new=AsyncMock(side_effect=error)) as agent:
                results = asyncio.run(
                    evaluate(
                        model="test-model",
                        cases_path=cases_path,
                        results_path=results_path,
                    )
                )
            self.assertEqual(results, [])
            self.assertEqual(agent.await_count, 1)  # stopped after the first case
            self.assertFalse(results_path.exists())

    def test_records_tokens_and_honesty(self) -> None:
        messages = [
            FakeMessage(
                [{"name": "simulate_rc_lowpass", "args": {"r_ohms": 6366.2, "c_farads": 1e-8}}],
                tokens=1200,
            ),
            FakeMessage([], tokens=300),
        ]
        self.assertEqual(total_tokens(messages), 1500)
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(self.design, messages))),
            patch(
                "evaluate.simulate_rc",
                return_value={"fc_measured_hz": 2500.5, "fc_theoretical_hz": 2500},
            ),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertEqual(result["tokens"], 1500)
        self.assertTrue(result["quality"]["claimed_cutoff_matches_simulation"])

    def test_dishonest_claim_is_flagged(self) -> None:
        claimed_2mhz = self.design.model_copy(update={"simulated_cutoff_hz": 2_000_000})
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(claimed_2mhz, self.messages))),
            patch(
                "evaluate.simulate_rc",
                return_value={"fc_measured_hz": 200_000, "fc_theoretical_hz": 200_000},
            ),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertFalse(result["quality"]["claimed_cutoff_matches_simulation"])

    def test_run_agent_raises_with_trace_when_limit_ends_the_run(self) -> None:
        stopped_messages = self.messages + [
            MagicMock(type="ai", text="Model call limits exceeded: run limit (8/8)", tool_calls=[])
        ]
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={"structured_response": None, "messages": stopped_messages}
        )
        client = MagicMock()
        client.get_tools = AsyncMock(return_value=[])
        with (
            patch("agent.MultiServerMCPClient", return_value=client),
            patch("agent.create_agent", return_value=agent),
        ):
            with self.assertRaises(AgentStoppedEarly) as caught:
                asyncio.run(run_agent("Design a filter.", "test-model"))
        self.assertEqual(caught.exception.reason, "model call limit reached")
        self.assertEqual(caught.exception.messages, stopped_messages)

    def test_stopped_agent_records_what_it_tried(self) -> None:
        tried = [
            FakeMessage([{"name": "find_standard_values", "args": {"target_cutoff_hz": 2500}}], tokens=500),
            FakeMessage([{"name": "simulate_rc_lowpass", "args": {"r_ohms": 6340, "c_farads": 1e-8}}], tokens=600),
            FakeMessage([{"name": "simulate_rc_lowpass", "args": {"r_ohms": 6490, "c_farads": 1e-8}}], tokens=700),
        ]
        stop = AgentStoppedEarly("model call limit reached", tried)
        with patch("evaluate.run_agent", new=AsyncMock(side_effect=stop)):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertFalse(result["passed"])
        self.assertNotIn("error", result)
        self.assertTrue(is_finished(result))
        self.assertEqual(result["failure_reason"], "model call limit reached")
        self.assertEqual(result["lookups"], 1)
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual(result["tokens"], 1800)
        self.assertEqual(
            result["simulations"],
            [{"r_ohms": 6340, "c_farads": 1e-8}, {"r_ohms": 6490, "c_farads": 1e-8}],
        )

    def test_passing_run_records_simulations_and_lookups(self) -> None:
        messages = [
            FakeMessage([{"name": "find_standard_values", "args": {"target_cutoff_hz": 2500}}]),
            FakeMessage([{"name": "simulate_rc_lowpass", "args": {"r_ohms": 6366.2, "c_farads": 1e-8}}]),
        ]
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(self.design, messages))),
            patch(
                "evaluate.simulate_rc",
                return_value={"fc_measured_hz": 2500.5, "fc_theoretical_hz": 2500},
            ),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["lookups"], 1)
        self.assertEqual(result["simulations"], [{"r_ohms": 6366.2, "c_farads": 1e-8}])

    def test_honesty_rules(self) -> None:
        self.assertTrue(claimed_cutoff_is_honest(2500.0, 2500.5))      # matches
        self.assertFalse(claimed_cutoff_is_honest(2_000_000, 200_000))  # 10x off
        self.assertTrue(claimed_cutoff_is_honest(None, None))           # nothing simulated, said so
        self.assertTrue(claimed_cutoff_is_honest(0.0, None))            # old placeholder for "none"
        self.assertFalse(claimed_cutoff_is_honest(2.969, None))         # formula passed off as simulated
        self.assertFalse(claimed_cutoff_is_honest(None, 2500.5))        # measured result left out

    def test_unmeasurable_case_with_null_claim_is_honest_and_passes(self) -> None:
        case = {"id": "below", "request": "3 Hz with 10 uF, confirm by simulation",
                "target_cutoff_hz": 3, "expect": "infeasible", "best_effort_percent": None,
                "fixed_components": {"c_farads": 1e-5}}
        design = FilterDesign(r_ohms=5360, c_farads=1e-5, simulated_cutoff_hz=None,
                              meets_spec=False, notes="Calculated 2.97 Hz; not verifiable.")
        messages = [FakeMessage([{"name": "simulate_rc_lowpass",
                                  "args": {"r_ohms": 5360, "c_farads": 1e-5}}])]
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(design, messages))),
            patch("evaluate.simulate_rc",
                  return_value={"fc_measured_hz": None, "fc_theoretical_hz": 2.969}),
        ):
            result = asyncio.run(evaluate_case(case, "test-model"))
        self.assertTrue(result["passed"], result["checks"])
        self.assertTrue(result["quality"]["claimed_cutoff_matches_simulation"])

    # --- Grading by expectation ------------------------------------------------

    def simulated_messages(self, r_ohms, c_farads):
        return [FakeMessage([{"name": "simulate_rc_lowpass",
                              "args": {"r_ohms": r_ohms, "c_farads": c_farads}}])]

    def test_infeasible_case_passes_when_agent_says_so_with_close_fallback(self) -> None:
        case = {"id": "e12", "request": "E12 only", "target_cutoff_hz": 1200,
                "expect": "infeasible",
                "required_series": {"r_ohms": "E12", "c_farads": "E12"}}
        design = FilterDesign(r_ohms=3900, c_farads=33e-9, simulated_cutoff_hz=1236.6,
                              meets_spec=False, notes="Cannot meet 2% with E12.")
        checks = grade_design(case, design, self.simulated_messages(3900, 33e-9), 1236.6)
        self.assertTrue(all(checks.values()), checks)

    def test_infeasible_case_fails_if_agent_claims_success(self) -> None:
        case = {"id": "e12", "request": "E12 only", "target_cutoff_hz": 1200,
                "expect": "infeasible"}
        design = FilterDesign(r_ohms=3900, c_farads=33e-9, simulated_cutoff_hz=1236.6,
                              meets_spec=True, notes="Meets spec.")
        checks = grade_design(case, design, self.simulated_messages(3900, 33e-9), 1236.6)
        self.assertFalse(checks["reports_infeasible"])

    def test_infeasible_case_fails_if_fallback_is_far_off(self) -> None:
        case = {"id": "e12", "request": "E12 only", "target_cutoff_hz": 1200,
                "expect": "infeasible"}
        design = FilterDesign(r_ohms=1000, c_farads=1e-7, simulated_cutoff_hz=1591.5,
                              meets_spec=False, notes="Cannot meet.")
        checks = grade_design(case, design, self.simulated_messages(1000, 1e-7), 1591.5)
        self.assertFalse(checks["best_effort_close"])   # 33% off; default limit is 5%

    def test_both_fixed_conflict_skips_closeness_when_best_effort_is_null(self) -> None:
        case = {"id": "both", "request": "10k and 10n, 5 kHz", "target_cutoff_hz": 5000,
                "expect": "infeasible", "best_effort_percent": None,
                "fixed_components": {"r_ohms": 10000, "c_farads": 1e-8}}
        design = FilterDesign(r_ohms=10000, c_farads=1e-8, simulated_cutoff_hz=1591.5,
                              meets_spec=False, notes="Fixed parts give 1.59 kHz.")
        checks = grade_design(case, design, self.simulated_messages(10000, 1e-8), 1591.5)
        self.assertNotIn("best_effort_close", checks)
        self.assertTrue(all(checks.values()), checks)

    def test_design_case_needs_agent_to_report_success(self) -> None:
        unsure = self.design.model_copy(update={"meets_spec": False})
        checks = grade_design(self.case, unsure, self.messages, 2500.5)
        self.assertTrue(checks["cutoff_within_tolerance"])
        self.assertFalse(checks["reports_spec_met"])

    def test_case_tolerance_overrides_default(self) -> None:
        case = dict(self.case, tolerance_percent=1)
        checks = grade_design(case, self.design, self.messages, 2540)   # +1.6%
        self.assertFalse(checks["cutoff_within_tolerance"])
        checks = grade_design(self.case, self.design, self.messages, 2540)
        self.assertTrue(checks["cutoff_within_tolerance"])

    def test_required_series_and_limits(self) -> None:
        case = dict(self.case, required_series={"r_ohms": "E96"},
                    limits={"r_ohms": {"max": 6000}})
        checks = grade_design(case, self.design, self.messages, 2500.5)
        self.assertFalse(checks["required_series_respected"])   # 6366.2 is not E96
        self.assertFalse(checks["within_limits"])               # above 6 kohm
        good = self.design.model_copy(update={"r_ohms": 5900})
        checks = grade_design(case, good, self.messages, 2500.5)
        self.assertTrue(checks["required_series_respected"])
        self.assertTrue(checks["within_limits"])

    def test_load_cases_rejects_bad_grading_fields(self) -> None:
        bad_cases = [
            {"expect": "maybe"},
            {"tolerance_percent": 0},
            {"best_effort_percent": -1},
            {"required_series": {"r_ohms": "E7"}},
            {"limits": {"r_ohms": {"most": 5}}},
        ]
        for extra in bad_cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.jsonl"
                path.write_text(json.dumps(dict(self.case, **extra)) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_answer_check_helpers(self) -> None:
        from agent import answer_was_simulated, correction_message
        self.assertTrue(answer_was_simulated(self.design, self.messages))
        slipped = self.design.model_copy(update={"c_farads": 1e-7})
        self.assertFalse(answer_was_simulated(slipped, self.messages))
        text = correction_message(slipped, self.messages).text
        self.assertIn("never simulated those exact values", text)
        self.assertIn("r_ohms=6366.2, c_farads=1e-08", text)
        text = correction_message(slipped, []).text
        self.assertIn("did not run simulate_rc_lowpass at all", text)

    def test_unverified_answer_gets_one_correction_then_is_accepted(self) -> None:
        unsimulated = self.design.model_copy(update={"c_farads": 2e-8})
        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=[
            {"structured_response": unsimulated, "messages": list(self.messages)},
            {"structured_response": self.design, "messages": list(self.messages)},
        ])
        client = MagicMock()
        client.get_tools = AsyncMock(return_value=[])
        with (
            patch("agent.MultiServerMCPClient", return_value=client),
            patch("agent.create_agent", return_value=agent),
        ):
            design, _ = asyncio.run(run_agent("Design a filter.", "test-model"))
        self.assertEqual(design, self.design)
        self.assertEqual(agent.ainvoke.await_count, 2)
        sent = agent.ainvoke.await_args_list[1].args[0]["messages"][-1]
        self.assertEqual(sent.name, "answer_check")

    def test_evaluation_records_answer_corrections(self) -> None:
        correction = MagicMock(type="human", tool_calls=[], usage_metadata=None,
                               response_metadata={})
        correction.name = "answer_check"
        messages = self.messages + [correction]
        with (
            patch("evaluate.run_agent", new=AsyncMock(return_value=(self.design, messages))),
            patch("evaluate.simulate_rc",
                  return_value={"fc_measured_hz": 2500.5, "fc_theoretical_hz": 2500}),
        ):
            result = asyncio.run(evaluate_case(self.case, "test-model"))
        self.assertEqual(result["answer_corrections"], 1)
        self.assertTrue(result["passed"])

    # --- Model construction --------------------------------------------------

    def test_builds_local_ollama_model_with_limits(self) -> None:
        model = build_chat_model("ollama:mightykatun/qwen2.5-math:1.5b")
        self.assertEqual(type(model).__name__, "ChatOllama")
        self.assertEqual(model.model, "mightykatun/qwen2.5-math:1.5b")
        self.assertEqual(model.temperature, 0)
        self.assertEqual(model.num_ctx, 8192)
        self.assertEqual(model.num_predict, 2048)

    def test_builds_gemini_model_with_low_thinking(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            model = build_chat_model("google_genai:gemini-3.8-flash")
        self.assertEqual(type(model).__name__, "ChatGoogleGenerativeAI")
        self.assertEqual(model.reasoning_effort, "low")
        self.assertEqual(model.max_output_tokens, 2048)

    def test_prompts_for_gemini_api_key_when_missing(self) -> None:
        terminal = MagicMock()
        terminal.isatty.return_value = True
        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": ""}),
            patch("agent.load_dotenv"),
            patch("agent.sys.stdin", terminal),
            patch("agent.getpass.getpass", return_value="prompted-key") as prompt,
        ):
            model = build_chat_model("google_genai:gemini-3.8-flash")
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "prompted-key")
        prompt.assert_called_once_with("Google AI Studio API key: ")
        self.assertEqual(type(model).__name__, "ChatGoogleGenerativeAI")

    def test_loads_gemini_api_key_from_dotenv(self) -> None:
        def load_test_key() -> None:
            self.assertNotIn("GOOGLE_API_KEY", os.environ)
            os.environ["GOOGLE_API_KEY"] = "dotenv-key"

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": ""}),
            patch("agent.load_dotenv", side_effect=load_test_key) as load,
            patch("agent.getpass.getpass") as prompt,
        ):
            model = build_chat_model("google_genai:gemini-3.8-flash")

        load.assert_called_once_with()
        prompt.assert_not_called()
        self.assertEqual(type(model).__name__, "ChatGoogleGenerativeAI")

    def test_does_not_retry_errors_that_cannot_succeed(self) -> None:
        self.assertTrue(should_retry(RuntimeError("503 UNAVAILABLE high demand")))
        for message in ("402 credits", "404 not found", "429 quota", "does not support tools"):
            with self.subTest(message=message):
                self.assertFalse(should_retry(RuntimeError(message)))


if __name__ == "__main__":
    unittest.main()
