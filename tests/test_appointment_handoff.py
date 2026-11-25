import json
from unittest.mock import AsyncMock, Mock

import pytest

from agents.appointment_agent import AppointmentAgent
from agents.appointment.input_parser import InputParser
from agents.task_classification_agent import TaskClassificationAgent
from config.constants import StateEnum


@pytest.mark.asyncio
async def test_conflicting_appointment_parse_clarifies_without_reclassifying(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    appointment = AppointmentAgent()
    router = TaskClassificationAgent(appointment, None)
    classify = AsyncMock(return_value="appointment")
    monkeypatch.setattr(router.task_classifier, "classify_task", classify)
    conflicting_parse = {
        "unrelated": True,
        "project": "按摩",
        "start_time": "2030-01-08 14:00",
        "duration": "60分钟",
        "gender": "女",
    }
    monkeypatch.setattr(
        appointment.input_parser, "parse_stream", lambda *_: [json.dumps(conflicting_parse)]
    )

    reply = "".join([token async for token in router.classify_task_stream("我想预约按摩")])

    classify.assert_awaited_once_with("我想预约按摩")
    assert "请再说一下" in reply
    assert router.state.value == StateEnum.APPOINTMENT
    assert all(value is None for value in appointment.appointment_history.values())


@pytest.mark.asyncio
async def test_unclear_recommendation_reply_keeps_waiting(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    appointment = AppointmentAgent()
    appointment.appointment_history["awaiting_confirmation"] = True
    appointment.appointment_history["recommended_technician"] = {"id": 2}
    monkeypatch.setattr(
        appointment.input_parser, "parse_stream", lambda *_: ['{"confirmation": null}']
    )
    save = Mock()
    monkeypatch.setattr(appointment.appointment_database, "save_appointment", save)

    reply = "".join([token async for token in appointment.run_stream("我再想想")])

    assert "请确认" in reply
    assert appointment.appointment_history["awaiting_confirmation"] is True
    save.assert_not_called()


@pytest.mark.parametrize(
    ("message", "duration"),
    [
        ("2030年1月8日下午2点，做60分钟，指定测试技师甲。", "60分钟"),
        ("预约1小时30分钟", "90分钟"),
        ("30分钟或者60分钟", None),
        ("60分钟后开始", None),
        ("1小时30分钟后开始", None),
    ],
)
def test_explicit_duration_ignores_start_time_and_ambiguous_options(message, duration):
    assert InputParser.extract_explicit_duration(message) == duration
