from unittest.mock import AsyncMock, Mock

import pytest

from agents.appointment.local_slot_extractor import LocalSlotExtractor, answer_confirmation
from agents.appointment_agent import AppointmentAgent


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("好，我同意预约你推荐的测试技师乙。", "同意"),
        ("好？", None),
        ("可以预约吗？", None),
        ("同意，但是想改到明天", None),
        ("我同意预约另一位技师", None),
        ("拒绝", "拒绝"),
    ],
)
def test_confirmation_is_specific_to_current_proposal(answer, expected):
    assert answer_confirmation(answer, "测试技师乙") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("available_at_confirmation", [True, False])
async def test_structured_model_proposes_before_write_and_rechecks(monkeypatch, available_at_confirmation):
    monkeypatch.setenv("MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    slots = {
        "project": "按摩", "start_time": "2030-01-08 14:00",
        "duration": "60分钟", "technician_name": "测试技师乙", "gender": "女",
        # The routing layer must ignore authorization in slots.
        "confirmation": "同意",
    }
    mock_extractor = Mock(extract=AsyncMock(return_value=slots))
    monkeypatch.setattr(LocalSlotExtractor, "from_environment", lambda **_: mock_extractor)
    agent = AppointmentAgent()
    agent.appointment_processor.llm = None
    monkeypatch.setattr(
        agent.technician_finder, "find_technician_with_thought",
        lambda *_: {"id": 2, "name": "测试技师乙", "gender": "女"},
    )
    monkeypatch.setattr(agent.technician_finder, "resolve_mentioned_name", lambda *_: "测试技师乙")
    service = Mock()
    service.is_technician_available.return_value = available_at_confirmation
    agent.appointment_database._appointment_service = service
    save = Mock(return_value=True)
    monkeypatch.setattr(agent.appointment_database, "save_appointment", save)

    async def reply(message):
        return "".join([part async for part in agent.run_stream(message)])

    assert "请回复「同意」" in await reply("希望预约测试技师乙")
    assert "请确认" in await reply("可以预约吗？")
    save.assert_not_called()
    mock_extractor.extract.assert_awaited_once()
    result = await reply("同意")
    service.is_technician_available.assert_called_once()
    if available_at_confirmation:
        assert "预约成功" in result
        save.assert_called_once()
    else:
        assert "时段已不可用" in result
        save.assert_not_called()
