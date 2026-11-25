"""Use the separately evaluated local model only for appointment slots."""

import asyncio
import logging
import os
import re

from config.time_config import time_config
from .conversation_memory import AppointmentMemory, OUTPUT_TOKENS
from .long_term_memory import AppointmentRecall


log = logging.getLogger(__name__)
_TERMS = {"female": "女", "male": "男"}


def answer_confirmation(text, technician_name):
    """Recognize an explicit reply to the current server-side proposal."""
    normalized = re.sub(r"[\s，。,.！!]+", "", text)
    if normalized in {"不", "不行", "不同意", "拒绝", "先不了", "不约", "不要预约"}:
        return "拒绝"
    if normalized in {"是", "好", "好的", "同意", "确认", "可以", "确定", "行"}:
        return "同意"
    name = re.escape(technician_name)
    if re.fullmatch(
        rf"(?:好(?:的)?)?(?:我)?(?:同意|确认|确定)(?:预约(?:你推荐的|您推荐的)?{name}(?:技师)?|预约)",
        normalized,
    ):
        return "同意"
    return None


class LocalSlotExtractor:
    def __init__(self, base_url, model, context_tokens=8192, *, user_id="default_user"):
        from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig

        self.backend = LlamaServerBackend(LlamaServerConfig(
            model=model, base_url=base_url, timeout_s=20, max_tokens=OUTPUT_TOKENS,
        ))
        self.memory = AppointmentMemory(base_url, context_tokens)
        self.recall = AppointmentRecall(user_id)

    @classmethod
    def from_environment(cls, *, user_id="default_user"):
        base_url = os.getenv("APPOINTMENT_SLOT_BASE_URL")
        model = os.getenv("APPOINTMENT_SLOT_MODEL")
        if bool(base_url) != bool(model):
            raise ValueError("Set both APPOINTMENT_SLOT_BASE_URL and APPOINTMENT_SLOT_MODEL")
        if base_url:
            return cls(base_url, model, int(os.getenv("APPOINTMENT_SLOT_CONTEXT_TOKENS", "8192")), user_id=user_id)
        return None

    async def extract(self, user_input, history, appointment):
        data, summary, dropped = await asyncio.to_thread(self._extract, user_input, tuple(history), appointment.copy())
        self.memory.summary = summary
        del history[:dropped]
        return data

    def _extract(self, user_input, history, appointment):
        from slot_extractor.prompts.template import PromptBuilder
        from slot_extractor.schemas.output import (
            OutputValidationError, parse_model_json, validate_final_output, validate_tool_call_output,
        )
        from slot_extractor.schemas.sample import Sample

        duration = appointment["duration"]
        prior = {
            "start_time": appointment["start_time"],
            "duration_minutes": int(duration.removesuffix("分钟")) if duration and duration.removesuffix("分钟").isdigit() else None,
            "preferences": [appointment["preference"]] if appointment["preference"] else [],
            "technician_name": appointment["technician_name"],
            "gender_preference": {"女": "female", "男": "male"}.get(appointment["gender"]),
            "technician_gender": None,
            "technician_status": "not_checked",
        }
        current_time = time_config.current_datetime_str()

        def build_messages(turns):
            sample = Sample("live", "final", "multi_turn" if turns else "single_turn", {
                "history": turns, "user_input": user_input,
                "current_time": current_time,
                "current_state": prior if any(value for key, value in prior.items() if key != "technician_status") else None,
                "available_tools": ["find_technicians"],
            }, {}, [], [])
            return PromptBuilder().build_messages(sample)

        recalled = [match.memory.text for match in self.recall.retrieve(user_input)]
        messages, summary, dropped = self.memory.prepare(history, build_messages, recalled=recalled)
        for attempt in (1, 2):
            generation = self.backend.generate(messages)
            try:
                output = parse_model_json(generation.text)
                if output.get("action") == "tool_call":
                    validate_tool_call_output(output)
                    if output["tool_name"] != "find_technicians":
                        raise OutputValidationError("Unexpected appointment tool")
                    facts = output["arguments"]
                else:
                    validate_final_output(output)
                    facts = output
            except OutputValidationError:
                if attempt == 2:
                    log.warning("Local slot extractor returned invalid JSON or schema after retry")
                    return None, summary, dropped
                log.warning("Local slot extractor returned invalid JSON or schema; retrying once")
                messages[0]["content"] += (
                    "\n再次检查字段合同。没有具体时间或时长的预约意图，只输出完整的 final JSON。"
                    "未给出的字段填 null；missing_info 顺序为 start_time、duration_minutes。"
                )
                self.memory.check_retry_budget(messages)
            else:
                break
        log.info("Local slot extractor actual_prompt_tokens=%s", generation.input_tokens)
        log.info("Local slot extractor model=%s action=%s latency_ms=%.1f",
                 self.backend.model, output["action"], generation.total_ms)
        if output.get("unrelated") is True:
            return {"unrelated": True}, summary, dropped
        # Even a model-emitted tool call provides slots only. The application owns all side effects.
        return {
            "unrelated": False,
            "project": "按摩",
            "gender": _TERMS.get(facts["gender_preference"]),
            "start_time": facts["start_time"],
            "duration": f"{facts['duration_minutes']}分钟" if facts["duration_minutes"] else None,
            "technician_name": facts["technician_name"],
            "preference": "、".join(p for p in facts["preferences"] if p not in {"按摩", "无"}) or None,
        }, summary, dropped
