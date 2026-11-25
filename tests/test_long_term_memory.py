from datetime import datetime, timedelta, timezone

import pytest

from agents.appointment.conversation_memory import AppointmentMemory
from agents.appointment.long_term_memory import AppointmentRecall, PreferenceMemory, rank_memories


NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)


def test_semantic_recency_and_importance_have_independent_weights():
    memories = [
        PreferenceMemory(1, "baseline", NOW - timedelta(days=30), 0.5),
        PreferenceMemory(2, "more similar", NOW - timedelta(days=30), 0.5),
        PreferenceMemory(3, "more recent", NOW, 0.5),
        PreferenceMemory(4, "more important", NOW - timedelta(days=30), 1.0),
    ]
    ranked = rank_memories(memories, [1.0, 0.0], [[0, 1], [1, 0], [0, 1], [0, 1]], NOW)
    scores = {item.memory.behavior_id: item.score for item in ranked}
    assert [item.memory.behavior_id for item in ranked] == [2, 3, 4, 1]
    assert scores[2] - scores[1] == pytest.approx(0.6)
    assert scores[3] - scores[1] == pytest.approx(0.3 * 0.5)
    assert scores[4] - scores[1] == pytest.approx(0.1 * 0.5)


def test_pick_newest_preference_and_exclude_historical_slots():
    data = {"project": "按摩", "preference": "力道轻", "duration": 90,
            "start_time": "2030-01-01 10:00", "technician_id": 100}
    records = [{"id": 2, "created_at": NOW, "action_data": data},
               {"id": 1, "created_at": NOW - timedelta(days=1), "action_data": data}]
    memories = AppointmentRecall._preferences(records)
    assert [(item.behavior_id, item.text) for item in memories] == [(2, "服务项目：按摩，偏好：力道轻")]


def test_recall_shrinks_at_token_limit_without_crowding_out_recent_context(monkeypatch):
    from langchain_core.messages import HumanMessage

    memory = AppointmentMemory("http://unused/v1", 1000)
    counts = iter([270, 530, 490, 450])
    monkeypatch.setattr(memory, "_count_tokens", lambda *_: next(counts))
    prompt, _, dropped = memory.prepare([HumanMessage(content="recent turn")], lambda history: history,
                                       recalled=["first", "second", "third"])
    assert "first" in str(prompt) and "second" not in str(prompt)
    assert prompt[-1]["content"] == "recent turn"
    assert dropped == 0
