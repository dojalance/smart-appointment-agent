import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.appointment.conversation_memory import AppointmentMemory


def build_prompt(history):
    return [{"role": "system", "content": "rules"}, *history, {"role": "user", "content": "current"}]


@pytest.mark.parametrize("candidate_tokens,compacted", [(599, False), (600, True), (601, True)])
def test_window_and_summary_boundary(monkeypatch, candidate_tokens, compacted):
    memory = AppointmentMemory("http://unused/v1", 1000)
    counts = iter([candidate_tokens, 150])
    monkeypatch.setattr(memory, "_count_tokens", lambda *_: next(counts))
    turns = []
    for turn in range(11):
        turns.extend([HumanMessage(content=f"note {turn}。details {turn}"), AIMessage(content="reply")])
    # A turn can include two assistant messages. It still counts as one turn.
    turns.append(AIMessage(content="follow-up"))
    messages, summary, dropped = memory.prepare(turns, build_prompt)
    user_messages = [item["content"] for item in messages if item["role"] == "user"]
    assert user_messages[-11:] == [f"note {turn}。details {turn}" for turn in range(1, 11)] + ["current"]
    assert "details 0" not in str(messages)
    assert summary == (["note 0"] if compacted else [])
    assert dropped == (2 if compacted else 0)


def test_oversized_last_turn_leaves_output_space(monkeypatch):
    memory = AppointmentMemory("http://unused/v1", 1000)
    counts = iter([700, 200])
    monkeypatch.setattr(memory, "_count_tokens", lambda *_: next(counts))
    history = [HumanMessage(content="context。" + "long" * 1000), AIMessage(content="reply")]
    messages, summary, dropped = memory.prepare(history, build_prompt)
    assert "long" not in str(messages)
    assert summary == ["context"]
    assert dropped == 2


def test_rejects_oversized_current_message(monkeypatch):
    memory = AppointmentMemory("http://unused/v1", 1000)
    monkeypatch.setattr(memory, "_count_tokens", lambda *_: 700)
    with pytest.raises(ValueError, match="context budget"):
        memory.prepare([], build_prompt)
