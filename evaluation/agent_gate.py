"""Score a trace against frozen expectations and return a release-gate exit code."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def fraction(checks):
    return sum(checks) / len(checks) if checks else None


def matches(actual, expected):
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def score(manifest, trace, manifest_sha256):
    cases = []
    routes, slots, tools, permissions, latencies, token_counts = [], [], [], [], [], []
    shape_ok = trace.get("fixture_sha256") == manifest_sha256
    expected_case_ids = [case["id"] for case in manifest["cases"]]
    repetitions = trace.get("repetitions", [])
    shape_ok = shape_ok and bool(repetitions)
    usage_complete = True

    for run_index, repetition in enumerate(repetitions, start=1):
        shape_ok = shape_ok and list(repetition) == expected_case_ids
        for case in manifest["cases"]:
            expected_ids = [turn["id"] for turn in case["turns"]]
            actual_turns = repetition.get(case["id"], {})
            shape_ok = shape_ok and list(actual_turns) == expected_ids
            checks = {}
            total_tokens = 0
            for turn in case["turns"]:
                actual = actual_turns.get(turn["id"], {})
                turn_checks = {"reply": bool(actual.get("reply")) and all(
                    text in actual["reply"] for text in turn.get("reply_contains", [])
                )}
                if "route" in turn:
                    turn_checks["route"] = actual.get("routes") == [turn["route"]]
                    routes.append(turn_checks["route"])
                if "slot" in turn:
                    turn_checks["slot"] = matches(actual.get("slot"), turn["slot"])
                    slots.extend(matches(actual.get("slot"), {field: value}) for field, value in turn["slot"].items())
                if "tools" in turn:
                    turn_checks["tool_trace"] = actual.get("tools") == turn["tools"]
                if "slot_tool" in turn:
                    expected = turn["slot_tool"]
                    call = actual.get("slot_tool") or {}
                    turn_checks["tool_trace"] = call.get("name") == expected["name"] and matches(call.get("args"), expected["args"])
                if "tool_observation" in turn:
                    turn_checks["tool_observation"] = any(
                        matches(item, turn["tool_observation"]) for item in actual.get("observations", [])
                    )
                if "tool_trace" in turn_checks:
                    tools.append(turn_checks["tool_trace"])
                if turn.get("permission"):
                    turn_checks["permission"] = actual.get("writes") == turn["writes"] and turn_checks.get("tool_observation", True)
                    permissions.append(turn_checks["permission"])
                calls = actual.get("model_calls")
                requires_model = "route" in turn or "slot_tool" in turn or bool(turn.get("tools"))
                if not isinstance(calls, list) or (requires_model and not calls):
                    usage_complete = False
                else:
                    duration = actual.get("wall_seconds")
                    if calls and type(duration) in (int, float) and math.isfinite(duration) and duration > 0:
                        latencies.append(duration)
                    elif calls:
                        usage_complete = False
                    for call in calls:
                        counts = [call.get("input_tokens"), call.get("output_tokens")]
                        if all(type(count) is int and count >= 0 for count in counts) and counts[0] > 0:
                            total_tokens += sum(counts)
                        else:
                            usage_complete = False
                checks[turn["id"]] = turn_checks
            token_counts.append(total_tokens)
            cases.append({"repetition": run_index, "id": case["id"], "checks": checks, "tokens": total_tokens,
                          "passed": all(value for turn_checks in checks.values() for value in turn_checks.values())})

    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
    actuals = {
        "task_success_min": fraction([case["passed"] for case in cases]), "route_accuracy_min": fraction(routes),
        "slot_accuracy_min": fraction(slots), "tool_trace_min": fraction(tools), "permission_min": fraction(permissions),
        "p95_seconds_max": p95, "tokens_per_task_max": max(token_counts, default=None),
    }
    metrics = {}
    for name, limit in manifest["limits"].items():
        value = actuals[name]
        passed = value is not None and (value >= limit if name.endswith("_min") else value <= limit)
        metrics[name] = {"actual": value, "limit": limit, "passed": passed}
    return {"passed": shape_ok and usage_complete and all(metric["passed"] for metric in metrics.values()),
            "fixture": manifest["version"], "fixture_sha256": manifest_sha256, "trace_shape_valid": shape_ok,
            "token_usage_complete": usage_complete, "counts": {"cases": len(cases), "routes": len(routes),
            "slots": len(slots), "tool_traces": len(tools), "permission_checks": len(permissions),
            "model_backed_turns": len(latencies)}, "metrics": metrics, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture.read_bytes()
    result = score(json.loads(fixture), json.loads(args.trace.read_text(encoding="utf-8")), hashlib.sha256(fixture).hexdigest())
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
