import hashlib
import json
import subprocess
import sys
from pathlib import Path

from evaluation.agent_gate import score


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evaluation/agent-smoke.v1.json"


def expected_trace():
    raw = FIXTURE.read_bytes()
    fixture = json.loads(raw)
    fingerprint = hashlib.sha256(raw).hexdigest()
    repetition = {}
    for case in fixture["cases"]:
        repetition[case["id"]] = {
            turn["id"]: {
                "reply": " ".join(turn.get("reply_contains", [])) or "正常返回",
                "wall_seconds": 1.0,
                "routes": [turn["route"]] if "route" in turn else [],
                "slot": turn.get("slot"), "slot_tool": turn.get("slot_tool"),
                "tools": turn.get("tools", []), "observations": [turn.get("tool_observation", {})],
                "writes": turn["writes"], "model_calls": [{"input_tokens": 10, "output_tokens": 1}],
            } for turn in case["turns"]
        }
    return fixture, {"fixture_sha256": fingerprint, "repetitions": [repetition]}, fingerprint


def test_gate_cli_blocks_permission_regression(tmp_path):
    _, trace, _ = expected_trace()
    stored, report = tmp_path / "trace.json", tmp_path / "report.json"
    command = [sys.executable, str(ROOT / "evaluation/agent_gate.py"), "--fixture", str(FIXTURE),
               "--trace", str(stored), "--report", str(report)]

    stored.write_text(json.dumps(trace), encoding="utf-8")
    assert subprocess.run(command, check=False, capture_output=True).returncode == 0
    trace["repetitions"][0]["supervisor_booking"]["visitor"]["writes"] = 1
    stored.write_text(json.dumps(trace), encoding="utf-8")
    assert subprocess.run(command, check=False, capture_output=True).returncode == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["metrics"]["permission_min"]["passed"] is False


def test_gate_blocks_missing_metrics_or_unfrozen_trace():
    fixture, trace, fingerprint = expected_trace()
    assert score(fixture, trace, fingerprint)["passed"]
    assert not score(fixture, trace, "wrong-hash")["passed"]
    call = trace["repetitions"][0]["slot_booking"]["slots"]["model_calls"][0]
    call["input_tokens"] = None
    result = score(fixture, trace, fingerprint)
    assert not result["token_usage_complete"] and not result["passed"]
