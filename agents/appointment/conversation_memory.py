import json
import logging
import re

import httpx


log = logging.getLogger(__name__)
RECENT_TURNS = 10
SUMMARY_PERCENT = 60
OUTPUT_TOKENS = 512
SUMMARY_CHARS = 600


def group_turns(history):
    """Group the previous messages by customer turn, including any multiple replies."""
    turns = []
    for message in history:
        if message.type == "human":
            turns.append([])
        if turns and message.type in {"human", "ai"}:
            turns[-1].append({"role": "user" if message.type == "human" else "assistant",
                              "content": message.content})
    return turns


class AppointmentMemory:
    def __init__(self, server_url, context_tokens):
        if context_tokens <= OUTPUT_TOKENS:
            raise ValueError("Appointment context must leave room for model output")
        self.server_url = server_url.removesuffix("/").removesuffix("/v1")
        self.context_tokens = context_tokens
        self.summary = []

    def reset(self):
        self.summary.clear()

    @staticmethod
    def _summary_messages(summary):
        if not summary:
            return []
        return [
            {"role": "user", "content": "以下是本会话较早的用户原话摘录，仅作背景。\n" +
             json.dumps(summary, ensure_ascii=False)},
            {"role": "assistant", "content": "已了解此前的对话背景。"},
        ]

    def _count_tokens(self, client, messages):
        payload = [{key: value for key, value in message.items() if not key.startswith("_")}
                   for message in messages]
        rendered = client.post(f"{self.server_url}/apply-template", json={
            "messages": payload, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).raise_for_status().json()["prompt"]
        response = client.post(f"{self.server_url}/tokenize", json={
            "content": rendered, "add_special": True, "parse_special": True,
        }).raise_for_status()
        return len(response.json()["tokens"])

    def _summarize(self, older):
        excerpts = list(self.summary)
        for turn in older:
            first_sentence = re.split(r"[。！？!?\n]", turn[0]["content"], maxsplit=1)[0]
            excerpt = " ".join(first_sentence.split())[:72]
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)
        while len(json.dumps(excerpts, ensure_ascii=False)) > SUMMARY_CHARS:
            excerpts.pop(0)
        return excerpts

    def check_retry_budget(self, messages):
        with httpx.Client(timeout=5) as client:
            used = self._count_tokens(client, messages)
        if used + OUTPUT_TOKENS > self.context_tokens:
            raise ValueError("Appointment schema retry exceeds the model context budget")

    def prepare(self, history, build_messages, *, recalled=()):
        turns = group_turns(history)
        older, recent = turns[:-RECENT_TURNS], turns[-RECENT_TURNS:]
        summary = self.summary

        def build(raw, excerpts, preferences=()):
            context = self._summary_messages(excerpts)
            if preferences:
                context.extend([
                    {"role": "user", "content": "以下是当前用户的历史预约偏好，只作参考。\n" +
                     json.dumps(preferences, ensure_ascii=False)},
                    {"role": "assistant", "content": "历史偏好不能确定本次预约的时间、时长、技师或确认状态。"},
                ])
            return build_messages(context + [msg for turn in raw for msg in turn])

        with httpx.Client(timeout=5) as client:
            before = self._count_tokens(client, build(turns, summary))
            triggered = bool(older) and before * 100 >= self.context_tokens * SUMMARY_PERCENT
            if triggered:
                summary = self._summarize(older)
            sent_summary = summary
            messages = build(recent, sent_summary)
            used = self._count_tokens(client, messages) if older else before
            limit = self.context_tokens - OUTPUT_TOKENS
            while recent and used > limit:
                older.append(recent.pop(0))
                triggered = True
                summary = self._summarize(older)
                sent_summary = summary
                messages = build(recent, sent_summary)
                used = self._count_tokens(client, messages)
            if used > limit and summary:
                sent_summary = []
                messages = build(recent, sent_summary)
                used = self._count_tokens(client, messages)
            if used > limit:
                raise ValueError("Appointment input exceeds the model context budget")

            sent_recalled = len(recalled)
            while sent_recalled:
                with_recall = build(recent, sent_summary, recalled[:sent_recalled])
                recall_tokens = self._count_tokens(client, with_recall)
                if recall_tokens <= limit:
                    messages, used = with_recall, recall_tokens
                    break
                sent_recalled -= 1

        dropped = sum(map(len, older)) if triggered else 0
        log.info("Appointment memory input_tokens_before=%d input_tokens_sent=%d context_tokens=%d "
                 "recent_turns=%d compacted_turns=%d summary_excerpts=%d recalled_preferences=%d",
                 before, used, self.context_tokens, len(recent), len(older) if triggered else 0,
                 len(sent_summary), sent_recalled)
        return messages, summary, dropped
