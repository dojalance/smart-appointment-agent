"""Collect live Agent traces in an isolated SQLite database with local model endpoints."""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from shutil import copytree
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import Cookies
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


def main():
    fixture_path = Path(__file__).with_name("agent-smoke.v1.json")
    fixture = fixture_path.read_bytes()
    manifest = json.loads(fixture)
    run = Path.cwd()
    work = run / "agent-eval"
    data = work / "data"
    data.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    copytree(project / "web/static", work / "web/static")
    os.chdir(work)

    from agents.task_classification.task_classifier import TaskClassifier
    from api.chat_handler import USER_COOKIE
    from app import app
    from services.appointment_service import AppointmentService
    from slot_extractor.inference.llama_server import LlamaServerBackend
    from slot_extractor.schemas.output import OutputValidationError, parse_model_json

    service = AppointmentService()
    busy = service.add_technician("测试技师甲", "女", "肩颈")
    service.add_technician("测试技师乙", "女", "肩颈")
    service.technician_repo.add_schedule(busy, datetime(2030, 1, 8, 14), datetime(2030, 1, 8, 15), "busy")

    original_chat = ChatOpenAI._agenerate
    original_stream = ChatOpenAI._astream
    original_slot = LlamaServerBackend.generate
    original_classify = TaskClassifier.classify_task
    calls, routes, decisions, observations = [], [], [], []

    def capture_chat(model, message, prompt):
        usage = message.usage_metadata or {}
        calls.append({"kind": "chat", "model": model.model_name,
                      "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                      "prompt": [entry.model_dump() for entry in prompt], "response": message.model_dump()})
        decisions.extend({"name": tool["name"], "args": tool["args"]} for tool in message.tool_calls)
        observations.extend(json.loads(entry.content) for entry in prompt if isinstance(entry, ToolMessage)
                            and entry.content.startswith("{") and entry.content not in {"{}"})

    async def chat(model, messages, *args, **kwargs):
        result = await original_chat(model, messages, *args, **kwargs)
        if not model.streaming:
            capture_chat(model, result.generations[0].message, messages)
        return result

    async def stream(model, messages, *args, **kwargs):
        combined = None
        kwargs["stream_usage"] = True
        async for chunk in original_stream(model, messages, *args, **kwargs):
            combined = chunk.message if combined is None else combined + chunk.message
            yield chunk
        if combined is not None:
            capture_chat(model, combined, messages)

    def slot(backend, messages, params=None):
        result = original_slot(backend, messages, params)
        try:
            parsed = parse_model_json(result.text)
        except OutputValidationError:
            parsed = None
        calls.append({"kind": "slot", "model": backend.model, "input_tokens": result.input_tokens,
                      "output_tokens": result.output_tokens, "prompt": [dict(message) for message in messages],
                      "response": result.text, "parsed": parsed})
        return result

    async def classify(classifier, text):
        result = await original_classify(classifier, text)
        routes.append(result)
        return result

    actual = {}
    with TestClient(app) as client:
        for scenario, text in (("after_sales", "请查手册，P200 打印机 E03 表示什么？"),
                               ("appointment", "我想预约按摩")):
            client.post("/chat/stream", json={"scenario": scenario, "session_id": "warmup", "message": text}).raise_for_status()

        with (patch.object(ChatOpenAI, "_agenerate", chat), patch.object(ChatOpenAI, "_astream", stream),
              patch.object(LlamaServerBackend, "generate", slot), patch.object(TaskClassifier, "classify_task", classify)):
            jars = {"owner": Cookies(), "visitor": Cookies()}
            for case in manifest["cases"]:
                scenario = case["scenario"]
                database = "after_sales.db" if scenario == "after_sales" else "smart_appointment.db"
                turns = actual[case["id"]] = {}
                for turn in case["turns"]:
                    calls.clear()
                    routes.clear()
                    decisions.clear()
                    observations.clear()
                    actor = turn.get("actor", "owner")
                    session = turn.get("session", case["id"])
                    client.cookies = jars[actor]
                    start = time.perf_counter()
                    response = client.post("/chat/stream", json={"scenario": scenario, "session_id": session,
                                                                 "message": turn["text"]})
                    duration = time.perf_counter() - start
                    response.raise_for_status()
                    jars[actor] = client.cookies
                    with closing(sqlite3.connect(data / database)) as db:
                        writes = db.execute("SELECT count(*) FROM appointment_writes").fetchone()[0]
                    user = app.state.chat_sessions.users[client.cookies[USER_COOKIE]]
                    agent = app.state.chat_sessions.agents.get((user, session))
                    output = next((call.get("parsed") for call in reversed(calls) if call["kind"] == "slot"), None)
                    slot_tool = {"name": output.get("tool_name"), "args": output.get("arguments")} if output and output.get("action") == "tool_call" else None
                    turns[turn["id"]] = {"reply": response.text, "wall_seconds": duration, "model_calls": calls.copy(),
                                         "routes": routes.copy(), "tools": decisions.copy(), "observations": observations.copy(),
                                         "slot": agent.appointment_agent.appointment_history.copy() if agent else None,
                                         "slot_tool": slot_tool, "writes": writes}
                    if turns[turn["id"]]["slot"] is not None:
                        relevant_slots = {"start_time", "duration", "technician_name"}
                        turns[turn["id"]]["slot"] = {key: value for key, value in turns[turn["id"]]["slot"].items()
                                                        if key in relevant_slots}

    trace = {"fixture_sha256": hashlib.sha256(fixture).hexdigest(), "repetitions": [actual]}
    (run / "agent-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {sum(len(turns) for turns in actual.values())} turns")


if __name__ == "__main__":
    main()
