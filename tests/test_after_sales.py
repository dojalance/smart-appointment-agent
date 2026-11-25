from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agents import after_sales


def call(name, call_id, args=None):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


@pytest.fixture
def supervisor(monkeypatch):
    services = Mock()
    services.knowledge.search = AsyncMock(return_value=[{"id": 7, "category": "光网络", "content": "演示 R100 手册"}])
    services.appointments.get_all_technicians.return_value = [{"id": 2, "name": "工程师林", "strength": "光网络"}]
    model = Mock(ainvoke=AsyncMock())
    model.bind_tools.return_value = model
    monkeypatch.setattr(after_sales, "create_chat_model", lambda **_: model)
    return after_sales.AfterSalesSupervisor(services, "owner", "session")


@pytest.mark.asyncio
async def test_tool_observation_provides_skill_but_only_confirmation_writes(supervisor):
    messages = []
    responses = iter([call("consult_product", "consult"), call("prepare_service_appointment", "prepare"),
                      AIMessage(content="等待确认")])

    async def decide(prompt):
        messages.append(prompt.copy())
        return next(responses)

    supervisor.model.ainvoke.side_effect = decide
    text = await supervisor.run("R100 红灯，请安排上门维修，2099-01-08 16:00，60分钟。")
    assert "工程师林" in text and "同意" in text
    assert isinstance(messages[1][-1], ToolMessage) and "光网络" in messages[1][-1].content
    assert isinstance(messages[2][-1], ToolMessage) and "pending_confirmation" in messages[2][-1].content
    assert "同意" in await supervisor.run("同意吗？")
    supervisor.services.appointments.save_appointment.assert_not_called()
    assert "预约成功" in await supervisor.run("同意")
    assert "没有待确认" in await supervisor.run("同意")
    supervisor.services.appointments.save_appointment.assert_called_once()


@pytest.mark.asyncio
async def test_rejected_arguments_and_repeat_do_not_query_history(supervisor):
    supervisor.model.ainvoke.side_effect = [call("analyze_customer_history", "1", {"user_id": "stranger"}),
                                            call("analyze_customer_history", "2")]
    await supervisor.run("查询历史")
    supervisor.services.behavior_agent.get_user_analysis.assert_not_called()
    assert supervisor.model.ainvoke.await_count == 2
