import logging
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from config.time_config import time_config
from db.repositories.appointment_repository import OwnedAppointment, WriteResult, local_wall_time


log = logging.getLogger(__name__)
_CHANGE = re.compile(
    r"^(?:请\s*)?(?:(?:帮我|我想|把)\s*)?"
    r"(?:(?:我(?:的)?)?(?:最近(?:一笔|一次)?的?)?预约\s*(?:改到|改为)|改(?:约|期)(?:到)?)"
    r"(?=\s|\d|[。！!]|$)"
)
_TIME = re.compile(r"\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*[。！!]?")
_ACCEPT = {"同意", "确认", "确认改约", "同意改约"}
_DECLINE = {"拒绝", "取消", "取消改约", "不同意"}
_CANCEL = re.compile(r"(?:请|帮我|我想)?\s*(?:取消|撤销)(?:我(?:的)?)?(?:最近(?:一笔|一次)?的?)?预约")
_CONFIRM_CANCEL = "确认取消预约"


@dataclass(frozen=True)
class ChangeProposal:
    request_id: str
    appointment: OwnedAppointment
    new_start: datetime | None = None


class AppointmentChanges:
    """Keep the proposal in the initiating conversation until the user confirms it."""

    def __init__(self, repository, user_id, session_id):
        self.repository = repository
        self.user_id = user_id
        self.session_id = session_id
        self.pending = None
        self.last_applied = None

    def handle(self, message):
        message = message.strip()
        answer = message.rstrip("。！!").strip()
        cancel = _CANCEL.fullmatch(answer)
        command = _CHANGE.match(message)
        if command or cancel:
            self.pending = self.last_applied = None
            now = time_config.now()
            new_start = None
            if command and not cancel:
                match = _TIME.fullmatch(message[command.end():])
                if not match:
                    return "请提供完整新时间，例如「把我的预约改到 2030-01-08 16:00」。"
                parsed = time_config.parse_datetime(match[1])
                if parsed is None or parsed <= now:
                    return "请输入有效的将来日期和时间。"
                new_start = local_wall_time(parsed)
            try:
                appointment = self.repository.latest_upcoming(self.user_id, now)
            except Exception:
                log.exception("Appointment lookup failed")
                return "暂时无法读取预约，请稍后重试。"
            if appointment is None:
                if cancel:
                    return "当前浏览器身份没有可取消的预约。"
                return "当前浏览器身份没有可改期的预约。"
            if new_start == appointment.start:
                return "新时间与原预约相同，请选择其他时间。"
            self.pending = ChangeProposal(str(uuid4()), appointment, new_start)
            if cancel:
                return (f"拟取消{appointment.technician_name}在 {appointment.start:%Y-%m-%d %H:%M} 的预约。"
                        "请回复「确认取消预约」或「拒绝」。")
            new_end = new_start + (appointment.end - appointment.start)
            return (f"请确认将{appointment.technician_name}的预约从{appointment.start:%Y-%m-%d %H:%M}"
                    f"改到{new_start:%Y-%m-%d %H:%M}至{new_end:%H:%M}。"
                    "请回复「同意」或「拒绝」。")

        proposal = self.pending if self.pending is not None else self.last_applied
        cancelling = proposal is not None and proposal.new_start is None
        accepted = answer == _CONFIRM_CANCEL if cancelling else answer in _ACCEPT
        if self.pending is not None:
            if answer in _DECLINE:
                self.pending = None
                return "已撤回变更申请，原预约保持有效。"
            if not accepted:
                return "请回复「确认取消预约」或「拒绝」。" if cancelling else "请明确回复「同意」或「拒绝」本次改约。"
        elif proposal is None or not accepted:
            self.last_applied = None
            if answer == _CONFIRM_CANCEL:
                return "当前会话没有待确认的取消申请。"
            return "当前会话没有待确认的改约。" if answer in {"确认改约", "同意改约"} else None

        try:
            args = (proposal.request_id, self.user_id, self.session_id, proposal.appointment.schedule_id,
                    proposal.appointment.revision)
            now = time_config.now()
            result = (self.repository.cancel(*args, now) if cancelling else
                      self.repository.reschedule(*args, proposal.new_start, now))
        except Exception:
            log.exception("Appointment reschedule transaction failed")
            return "暂时无法提交变更，请稍后重新确认。"
        self.pending = None
        if result in {WriteResult.APPLIED, WriteResult.REPLAYED}:
            self.last_applied = proposal
            if cancelling:
                return "预约已取消。" if result is WriteResult.APPLIED else "这次取消已处理。"
            return "改约成功。" if result is WriteResult.APPLIED else "这次改约已处理。"
        self.last_applied = None
        if result is WriteResult.CONFLICT:
            return "新时段已被占用，请选择其他时间。原预约保持有效。"
        return "预约状态已变化，请重新发起变更。"
