"""Run resumable evaluations of the RC filter design agent.

Usage:
    python evaluate.py [--model MODEL] [--limit 6]
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from agent import DEFAULT_MODEL, FilterDesign, run_agent
from spice_agent.simulate import simulate_rc


EVALS_DIR = Path(__file__).resolve().parent / "evals"
CASES_PATH = EVALS_DIR / "cases.jsonl"
RESULTS_PATH = EVALS_DIR / "results.jsonl"
MAX_CASES_PER_RUN = 6
TOOL_NAME = "simulate_rc_lowpass"
FIXED_COMPONENT_NAMES = {"r_ohms", "c_farads"}


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
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Case {case_id!r} fixed component {name} must be positive "
                    "and finite"
                )
    return cases


def count_tool_calls(messages: list[Any]) -> int:
    return sum(
        len(message.tool_calls)
        for message in messages
        if getattr(message, "type", None) == "ai"
    )


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


async def evaluate_case(case: dict[str, Any], model: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    record: dict[str, Any] = {
        "case_id": case["id"],
        "model": model,
        "passed": False,
        "checks": {
            "cutoff_within_2_percent": False,
            "final_values_in_tool_call": False,
            "fixed_components_unchanged": False,
        },
        "tool_call_count": 0,
        "seconds": 0.0,
    }
    try:
        design, messages = await run_agent(case["request"], model)
        tool_call_count = count_tool_calls(messages)
        record["tool_call_count"] = tool_call_count
        final_values_in_call = final_values_in_tool_call(design, messages)
        fixed_unchanged = fixed_components_unchanged(case, design)

        response = simulate_rc(design.r_ohms, design.c_farads)
        measured_cutoff = response["fc_measured_hz"]
        simulated_cutoff = (
            measured_cutoff
            if measured_cutoff is not None
            else response["fc_theoretical_hz"]
        )
        cutoff_error = abs(simulated_cutoff / case["target_cutoff_hz"] - 1)

        record.update(
            {
                "design": design.model_dump(),
                "simulated_cutoff_hz": simulated_cutoff,
                "cutoff_error_fraction": cutoff_error,
                "checks": {
                    "cutoff_within_2_percent": cutoff_error <= 0.02,
                    "final_values_in_tool_call": final_values_in_call,
                    "fixed_components_unchanged": fixed_unchanged,
                },
            }
        )
        record["passed"] = all(record["checks"].values())
    except Exception as error:
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
    print(
        f"Pass rate: {passed}/{len(results)} ({passed / len(results):.1%}) | "
        f"Average tool calls: {average_tool_calls:.2f} | "
        f"Average time: {average_seconds:.2f} s"
    )


async def evaluate(
    model: str = DEFAULT_MODEL,
    limit: int = MAX_CASES_PER_RUN,
    cases_path: Path = CASES_PATH,
    results_path: Path = RESULTS_PATH,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    cases = load_cases(cases_path)
    recorded_results = load_jsonl(results_path) if results_path.exists() else []
    results_by_case_id = {
        record["case_id"]: record
        for record in recorded_results
        if isinstance(record.get("case_id"), str)
    }
    finished_ids = set(results_by_case_id)
    pending_cases = [case for case in cases if case["id"] not in finished_ids][:limit]
    for case in pending_cases:
        result = await evaluate_case(case, model)
        append_result(result, results_path)
        results_by_case_id[case["id"]] = result
        print(
            f"{case['id']}: {'PASS' if result['passed'] else 'FAIL'} "
            f"({result['tool_call_count']} tool calls, {result['seconds']:.2f} s)"
        )
        if "error" in result:
            print(f"  Error: {result['error']}")

    if not pending_cases:
        print("All evaluation cases are already recorded.")
    results = [
        results_by_case_id[case["id"]]
        for case in cases
        if case["id"] in results_by_case_id
    ]
    summarize(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RC filter design agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LangChain model name")
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_CASES_PER_RUN,
        help="Maximum number of unfinished cases to run (default: 6)",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(model=args.model, limit=args.limit))


if __name__ == "__main__":
    main()
