from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.dialects.sqlite import insert

from db.models import FollowUpDelivery
from db.repositories.appointment_repository import WriteResult


class FollowUpInbox:
    def __init__(self, session_manager):
        self.session_manager = session_manager

    def deliver(self, request_id, user_id, session_id, content):
        with self.session_manager.session_scope() as session:
            result = session.execute(insert(FollowUpDelivery).values(
                request_id=request_id, user_id=user_id, session_id=session_id, content=content,
            ).on_conflict_do_nothing(index_elements=[FollowUpDelivery.request_id]))
            if result.rowcount == 1:
                return WriteResult.APPLIED
            existing = session.get(FollowUpDelivery, request_id)
            if (existing.user_id, existing.session_id, existing.content) == (user_id, session_id, content):
                return WriteResult.REPLAYED
            return WriteResult.INVALID

    def latest(self, user_id):
        with self.session_manager.session_scope() as session:
            return session.query(FollowUpDelivery.content).filter_by(user_id=user_id).order_by(
                FollowUpDelivery.created_at.desc(), FollowUpDelivery.request_id.desc(),
            ).limit(1).scalar()


@dataclass(frozen=True)
class FollowUpPreview:
    request_id: str
    content: str


class FollowUpConfirmation:
    def __init__(self, inbox, behavior_agent, user_id, session_id):
        self.inbox = inbox
        self.behavior_agent = behavior_agent
        self.user_id = user_id
        self.session_id = session_id
        self.pending = None
        self.last_delivered = None

    def handle(self, user_input):
        answer = user_input.strip().rstrip("。！!").strip()
        if answer in {"发送回访消息", "给我发送回访消息"}:
            self.last_delivered = None
            analysis = self.behavior_agent.get_user_analysis(self.user_id) or {}
            count = analysis.get("total_appointments") or 0
            if not count:
                self.pending = None
                return "当前浏览器身份没有可回访的历史服务。"
            self.pending = FollowUpPreview(
                str(uuid4()), f"您好，您有 {count} 条历史服务预约。请问近期设备使用是否正常？如需协助，请在售后会话留言。"
            )
            return f"拟投递到您的站内收件箱：{self.pending.content}\n请回复「确认发送回访」或「拒绝」。"
        if self.pending is None and answer == "查看我的回访消息":
            return self.inbox.latest(self.user_id) or "站内收件箱暂无回访消息。"
        if answer != "确认发送回访":
            if self.pending is None:
                self.last_delivered = None
                return None
            if answer in {"拒绝", "不同意", "取消发送"}:
                self.pending = None
                return "已撤回回访发送申请。"
            return "请回复「确认发送回访」或「拒绝」。"

        proposed = self.pending or self.last_delivered
        if proposed is None:
            return "当前会话没有待确认的回访发送申请。"
        result = self.inbox.deliver(proposed.request_id, self.user_id, self.session_id, proposed.content)
        self.pending = None
        if result in {WriteResult.APPLIED, WriteResult.REPLAYED}:
            self.last_delivered = proposed
            return "回访已投递到您的站内收件箱。" if result is WriteResult.APPLIED else "该回访已投递。"
        self.last_delivered = None
        return "回访状态已变化，请重新发起。"
