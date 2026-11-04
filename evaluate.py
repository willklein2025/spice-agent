"""Run resumable evaluations of the RC filter design agent.

Usage:
    python evaluate.py [--model MODEL] [--limit 6] [--runs 1] [--cases PATH] [--results PATH]
                       [--case-timeout 300]
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from agent import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    AgentStoppedEarly,
    FilterDesign,
    answer_corrections,
    run_agent,
)
from spice_agent.simulate import simulate_rc
from spice_agent.standard_values import E_SERIES, is_standard_value

try:
    from langchain_google_genai.chat_models import GoogleRateLimitError
except ImportError:  # Gemini package not installed: nothing to catch
    class GoogleRateLimitError(Exception):
        pass


EVALS_DIR = Path(__file__).resolve().parent / "evals"
CASES_PATH = EVALS_DIR / "cases.jsonl"
RESULTS_PATH = EVALS_DIR / "results.jsonl"
MAX_CASES_PER_RUN = 6
TOOL_NAME = "simulate_rc_lowpass"
LOOKUP_TOOL_NAME = "find_standard_values"
FIXED_COMPONENT_NAMES = {"r_ohms", "c_farads"}
CASE_TIMEOUT_S = 300          # wall-clock limit for one case; a runaway becomes a FAIL
HONESTY_TOLERANCE = 0.005     # claimed cutoff must match the re-simulation within 0.5%
DEFAULT_TOLERANCE_PERCENT = 2.0
DEFAULT_BEST_EFFORT_PERCENT = 5.0   # infeasible cases: how close the fallback design must be
EXPECTATIONS = {"design", "infeasible"}

# Problems no amount of retrying will fix during this run. The evaluation stops,
# nothing is recorded for the case, and it runs again next time.
FATAL_ERROR_HINTS = {
    "401": "The API key was rejected. Check GOOGLE_API_KEY in your .env file.",
    "API key not valid": "The API key was rejected. Check GOOGLE_API_KEY in your .env file.",
    "No Google AI Studio API key": "No API key found. Put GOOGLE_API_KEY in your .env file.",
    "402": "API credits are used up. Top up, then rerun to continue.",
    "429": "Daily quota reached. Rerun after it resets to continue.",
    "does not support tools": (
        "This model cannot call tools in Ollama, so it cannot run the simulator. "
        "Try a tool-capable model, for example: ollama pull qwen3:4b"
    ),
    "Failed to connect to Ollama": "Ollama is not running. Start it with: ollama serve",
    "Connection refused": "Could not reach the model server. Is Ollama running?",
    "not found": "Model not found. Check the name, or pull it with: ollama pull <model>",
}


def fatal_error_hint(error: BaseException) -> str | None:
    """Return a short explanation if this error should stop the whole evaluation."""
    if isinstance(error, GoogleRateLimitError):
        return FATAL_ERROR_HINTS["429"]
    message = str(error)
    for marker, hint in FATAL_ERROR_HINTS.items():
        if marker in message:
            return hint
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            records.append(record)
    return records


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = load_jsonl(path)
    seen_ids: set[str] = set()
    for line_number, case in enumerate(cases, start=1):
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Case on line {line_number} must have a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id {case_id!r} in {path}")
        seen_ids.add(case_id)

        request = case.get("request")
        target = case.get("target_cutoff_hz")
        fixed_components = case.get("fixed_components", {})
        if not isinstance(request, str) or not request:
            raise ValueError(f"Case {case_id!r} must have a non-empty request")
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(target)
            or target <= 0
        ):
            raise ValueError(
                f"Case {case_id!r} must have a positive finite target cutoff"
            )
        if not isinstance(fixed_components, dict):
            raise ValueError(f"Case {case_id!r} fixed_components must be an object")
        unknown_components = set(fixed_components) - FIXED_COMPONENT_NAMES
        if unknown_components:
            raise ValueError(
                f"Case {case_id!r} has unknown fixed components: "
                f"{', '.join(sorted(unknown_components))}"
            )
        for name, value in fixed_components.items():
            if not _is_positive_number(value):
                raise ValueError(
                    f"Case {case_id!r} fixed component {name} must be positive "
                    "and finite"
                )
        _validate_grading_fields(case_id, case)
    return cases


def _is_positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _validate_grading_fields(case_id: str, case: dict[str, Any]) -> None:
    """Check the optional fields that change how a case is graded."""
    expect = case.get("expect", "design")
    if expect not in EXPECTATIONS:
        raise ValueError(f"Case {case_id!r} expect must be one of {sorted(EXPECTATIONS)}")
    if not _is_positive_number(case.get("tolerance_percent", DEFAULT_TOLERANCE_PERCENT)):
        raise ValueError(f"Case {case_id!r} tolerance_percent must be positive")
    best_effort = case.get("best_effort_percent", DEFAULT_BEST_EFFORT_PERCENT)
    if best_effort is not None and not _is_positive_number(best_effort):
        raise ValueError(f"Case {case_id!r} best_effort_percent must be positive or null")
    for part, series in case.get("required_series", {}).items():
        if part not in FIXED_COMPONENT_NAMES or series not in E_SERIES:
            raise ValueError(
                f"Case {case_id!r} required_series must map r_ohms/c_farads to "
                f"one of {', '.join(E_SERIES)}"
            )
    for part, bounds in case.get("limits", {}).items():
        if (
            part not in FIXED_COMPONENT_NAMES
            or not isinstance(bounds, dict)
            or set(bounds) - {"min", "max"}
            or not all(_is_positive_number(v) for v in bounds.values())
        ):
            raise ValueError(
                f"Case {case_id!r} limits must look like "
                '{"r_ohms": {"max": 10000}} with positive numbers'
            )


def result_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    """Identify one experiment result: which model, which prompt, which case, which run."""
    return (
        record.get("model"),
        record.get("prompt_version", "v1"),  # older records had no version: treat as v1
        record.get("case_id"),
        record.get("run", 1),                # older records were a single run: run 1
    )


def is_finished(record: dict[str, Any]) -> bool:
    """A real PASS or FAIL is finished; an infrastructure error should be rerun."""
    return "error" not in record


def count_tool_calls(messages: list[Any]) -> int:
    """Count simulator calls only; a structured-output call such as FilterDesign is not one."""
    return sum(
        1
        for message in messages
        if getattr(message, "type", None) == "ai"
        for call in message.tool_calls
        if call.get("name") == TOOL_NAME
    )


def tool_calls_named(messages: list[Any], name: str) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if getattr(message, "type", None) == "ai"
        for call in message.tool_calls
        if call.get("name") == name
    ]


def simulations_run(messages: list[Any]) -> list[dict[str, Any]]:
    """The R and C values the agent simulated, in order."""
    return [
        {key: call.get("args", {}).get(key) for key in ("r_ohms", "c_farads")}
        for call in tool_calls_named(messages, TOOL_NAME)
    ]


def lookup_count(messages: list[Any]) -> int:
    return len(tool_calls_named(messages, LOOKUP_TOOL_NAME))


def describe_trace(record: dict[str, Any], messages: list[Any]) -> None:
    """Add what the agent did (tools, tokens, models) to a result record."""
    record["tool_call_count"] = count_tool_calls(messages)
    record["lookups"] = lookup_count(messages)
    record["answer_corrections"] = answer_corrections(messages)
    record["simulations"] = simulations_run(messages)
    record["tokens"] = total_tokens(messages)
    record["model_names"] = model_names_used(messages)


def model_names_used(messages: list[Any]) -> list[str]:
    """Which models actually answered (shows when a fallback or alias was used)."""
    names = {
        (getattr(message, "response_metadata", None) or {}).get("model_name")
        for message in messages
        if getattr(message, "type", None) == "ai"
    }
    return sorted(name for name in names if name)


def final_values_in_tool_call(design: FilterDesign, messages: list[Any]) -> bool:
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        for call in message.tool_calls:
            if call.get("name") != TOOL_NAME:
                continue
            arguments = call.get("args", {})
            if all(
                isinstance(arguments.get(name), (int, float))
                and not isinstance(arguments.get(name), bool)
                and math.isclose(
                    arguments[name],
                    getattr(design, name),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
                for name in ("r_ohms", "c_farads")
            ):
                return True
    return False


def fixed_components_unchanged(
    case: dict[str, Any], design: FilterDesign
) -> bool:
    return all(
        math.isclose(
            getattr(design, name),
            value,
            rel_tol=1e-9,
            abs_tol=0.0,
        )
        for name, value in case.get("fixed_components", {}).items()
    )

def claimed_cutoff_is_honest(
    claimed_hz: float | None, measured_hz: float | None
) -> bool:
    """Does the agent's "simulated cutoff" agree with our own re-simulation?

    Honest:   both missing (nothing was simulated, and the agent said so), or
              both present and within HONESTY_TOLERANCE.
    Dishonest: a value claimed when nothing could be measured, or a measured
              cutoff the agent left out or misreported.
    A claim of 0 is treated as "not simulated" (older answers used 0 as a placeholder).
    """
    if claimed_hz is not None and claimed_hz <= 0:
        claimed_hz = None
    if claimed_hz is None or measured_hz is None:
        return claimed_hz is None and measured_hz is None
    return abs(claimed_hz / measured_hz - 1) <= HONESTY_TOLERANCE


def grade_design(
    case: dict[str, Any],
    design: FilterDesign,
    messages: list[Any],
    measured_cutoff: float | None,
) -> dict[str, bool]:
    """Pass/fail checks for one answer, chosen by what the case expects.

    expect "design":     the cutoff must be within tolerance and the agent must say so.
    expect "infeasible": the agent must say the spec cannot be met, and (unless
                         best_effort_percent is null) its fallback must be close.
    """
    tolerance = case.get("tolerance_percent", DEFAULT_TOLERANCE_PERCENT) / 100
    error = (
        None
        if measured_cutoff is None
        else abs(measured_cutoff / case["target_cutoff_hz"] - 1)
    )
    checks = {
        "final_values_in_tool_call": final_values_in_tool_call(design, messages),
        "fixed_components_unchanged": fixed_components_unchanged(case, design),
    }
    if case.get("expect", "design") == "design":
        checks["cutoff_within_tolerance"] = error is not None and error <= tolerance
        checks["reports_spec_met"] = design.meets_spec is True
    else:
        checks["reports_infeasible"] = design.meets_spec is False
        best_effort = case.get("best_effort_percent", DEFAULT_BEST_EFFORT_PERCENT)
        if best_effort is not None:
            checks["best_effort_close"] = error is not None and error <= best_effort / 100
    if case.get("required_series"):
        checks["required_series_respected"] = all(
            is_standard_value(getattr(design, part), series)
            for part, series in case["required_series"].items()
        )
    if case.get("limits"):
        checks["within_limits"] = all(
            bounds.get("min", 0) * (1 - 1e-9)
            <= getattr(design, part)
            <= bounds.get("max", math.inf) * (1 + 1e-9)
            for part, bounds in case["limits"].items()
        )
    return checks


def total_tokens(messages: list[Any]) -> int:
    return sum(
        (getattr(message, "usage_metadata", None) or {}).get("total_tokens", 0)
        for message in messages
        if getattr(message, "type", None) == "ai"
    )


async def evaluate_case(
    case: dict[str, Any],
    model: str,
    run: int = 1,
    case_timeout_s: float = CASE_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one case and grade it.

    A case that hits the model-call limit or the time limit is a real FAIL (recorded,
    not rerun). Fatal errors (quota, credits, Ollama down, no tool support) are raised
    so the evaluation stops without recording the case.
    """
    started_at = time.perf_counter()
    record: dict[str, Any] = {
        "case_id": case["id"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "run": run,
        "expect": case.get("expect", "design"),
        "passed": False,
        "checks": {},
        "tool_call_count": 0,
        "lookups": 0,
        "tokens": 0,
        "seconds": 0.0,
    }
    try:
        try:
            design, messages = await asyncio.wait_for(
                run_agent(case["request"], model, fallback_model=None),
                timeout=case_timeout_s,
            )
        except AgentStoppedEarly as stop:
            # The agent ran away but we kept its conversation: record what it tried.
            record["failure_reason"] = stop.reason
            describe_trace(record, stop.messages)
            return record
        except (ModelCallLimitExceededError, asyncio.TimeoutError) as error:
            # Stopped without a trace: still a failure, so it is not rerun (and paid for) again.
            record["failure_reason"] = (
                "model call limit reached"
                if isinstance(error, ModelCallLimitExceededError)
                else f"timed out after {case_timeout_s:g} s"
            )
            return record

        describe_trace(record, messages)

        # Grade by re-simulating the agent's final values ourselves.
        response = simulate_rc(design.r_ohms, design.c_farads)
        measured_cutoff = response["fc_measured_hz"]
        if measured_cutoff is None:
            # Do not substitute the formula: a missing measurement is worth seeing.
            cutoff_error = None
            record["cutoff_measurement_error"] = (
                "ngspice did not report a measured cutoff for the final design"
            )
        else:
            cutoff_error = abs(measured_cutoff / case["target_cutoff_hz"] - 1)

        record.update(
            {
                "design": design.model_dump(),
                "simulated_cutoff_hz": measured_cutoff,
                "cutoff_error_fraction": cutoff_error,
                "checks": grade_design(case, design, messages, measured_cutoff),
                # Recorded for reporting, not part of pass/fail.
                "quality": {
                    "claimed_cutoff_matches_simulation": claimed_cutoff_is_honest(
                        design.simulated_cutoff_hz, measured_cutoff
                    ),
                },
            }
        )
        record["passed"] = bool(record["checks"]) and all(record["checks"].values())
    except Exception as error:
        if fatal_error_hint(error) is not None:
            raise
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["seconds"] = round(time.perf_counter() - started_at, 3)
    return record


def append_result(record: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, allow_nan=False) + "\n")
        file.flush()


def summarize(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No cases to evaluate.")
        return
    passed = sum(bool(result["passed"]) for result in results)
    average_tool_calls = (
        sum(result["tool_call_count"] for result in results) / len(results)
    )
    average_seconds = sum(result["seconds"] for result in results) / len(results)
    average_tokens = sum(result.get("tokens", 0) for result in results) / len(results)
    runaways = sum("failure_reason" in result for result in results)
    corrected = sum(bool(result.get("answer_corrections")) for result in results)
    print(
        f"Pass rate: {passed}/{len(results)} ({passed / len(results):.1%}) | "
        f"Average tool calls: {average_tool_calls:.2f} | "
        f"Average tokens: {average_tokens:,.0f} | "
        f"Average time: {average_seconds:.2f} s"
    )
    if runaways:
        print(f"  Stopped by call or time limit: {runaways}")
    if corrected:
        print(f"  Answers corrected by the answer check: {corrected}")


async def evaluate(
    model: str = DEFAULT_MODEL,
    limit: int = MAX_CASES_PER_RUN,
    runs: int = 1,
    cases_path: Path = CASES_PATH,
    results_path: Path = RESULTS_PATH,
    case_timeout_s: float = CASE_TIMEOUT_S,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")

    cases = load_cases(cases_path)
    recorded_results = load_jsonl(results_path) if results_path.exists() else []
    results_by_key = {
        result_key(record): record
        for record in recorded_results
        if is_finished(record)
    }

    # Every (case, run) this model + prompt still needs, run 1 of every case first.
    pending = [
        (case, run)
        for run in range(1, runs + 1)
        for case in cases
        if (model, PROMPT_VERSION, case["id"], run) not in results_by_key
    ][:limit]

    for case, run in pending:
        try:
            result = await evaluate_case(case, model, run, case_timeout_s)
        except Exception as error:
            hint = fatal_error_hint(error)
            if hint is None:
                raise
            print(f"\nStopping: {hint}\n  ({type(error).__name__}: {str(error)[:200]})")
            break
        append_result(result, results_path)
        if is_finished(result):
            results_by_key[result_key(result)] = result
        print(
            f"{case['id']} (run {run}): {'PASS' if result['passed'] else 'FAIL'} "
            f"({result.get('lookups', 0)} lookups, {result['tool_call_count']} simulations, "
            f"{result.get('tokens', 0):,} tokens, "
            f"{result['seconds']:.2f} s)"
        )
        if result.get("answer_corrections"):
            print(f"  Answer check asked for a correction ({result['answer_corrections']}x)")
        failed_checks = [name for name, ok in result.get("checks", {}).items() if not ok]
        if failed_checks and "failure_reason" not in result and "error" not in result:
            print(f"  Failed checks: {', '.join(failed_checks)}")
        if "failure_reason" in result:
            print(f"  Stopped: {result['failure_reason']}")
            for simulation in result.get("simulations", []):
                print(f"    simulated R={simulation['r_ohms']}, C={simulation['c_farads']}")
        if "error" in result:
            print(f"  Error: {result['error']}")

    if not pending:
        print("All evaluation cases are already recorded.")

    # Summarize only this model + prompt, across all runs.
    results = [
        results_by_key[(model, PROMPT_VERSION, case["id"], run)]
        for run in range(1, runs + 1)
        for case in cases
        if (model, PROMPT_VERSION, case["id"], run) in results_by_key
    ]
    print(f"\nModel: {model} | Prompt: {PROMPT_VERSION} | Runs per case: {runs}")
    summarize(results)
    report_inconsistent_cases(results)
    return results


def report_inconsistent_cases(results: list[dict[str, Any]]) -> None:
    """List cases whose runs did not all agree (some passed, some failed)."""
    outcomes: dict[str, list[bool]] = {}
    for result in results:
        outcomes.setdefault(result["case_id"], []).append(bool(result["passed"]))
    for case_id, passes in outcomes.items():
        if len(passes) > 1 and len(set(passes)) > 1:
            print(f"  Inconsistent: {case_id} passed {sum(passes)} of {len(passes)} runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RC filter design agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LangChain model name")
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_CASES_PER_RUN,
        help="Maximum number of unfinished cases to run (default: 6)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run each case (default: 1)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="JSONL file of test cases (default: evals/cases.jsonl)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="JSONL file to read and append results (default: evals/results.jsonl)",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=CASE_TIMEOUT_S,
        help=f"Seconds allowed per case before it counts as a FAIL (default: {CASE_TIMEOUT_S})",
    )
    args = parser.parse_args()
    asyncio.run(
        evaluate(
            model=args.model,
            limit=args.limit,
            runs=args.runs,
            cases_path=args.cases,
            results_path=args.results,
            case_timeout_s=args.case_timeout,
        )
    )


if __name__ == "__main__":
    main()