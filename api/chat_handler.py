import asyncio
import logging
from secrets import token_urlsafe
from uuid import uuid4

from fastapi import Request, Response

from agents.appointment_agent import AppointmentAgent
from agents.after_sales import AfterSalesServices, AfterSalesSupervisor
from agents.consultant_agent import ConsultantAgent
from agents.task_classification_agent import TaskClassificationAgent


USER_COOKIE = "appointment_user"
log = logging.getLogger(__name__)


class ChatSessions:
    def __init__(self):
        self.users: dict[str, str] = {}
        self.agents: dict[tuple[str, str], TaskClassificationAgent] = {}
        self.after_sales_agents: dict[tuple[str, str], AfterSalesSupervisor] = {}
        self.after_sales_services = None
        # The scheduler is shared, so keep booking checks and writes serialized.
        self.lock = asyncio.Lock()

    def user_id(self, request: Request) -> str | None:
        return self.users.get(request.cookies.get(USER_COOKIE, ""))

    def identify(self, request: Request, response: Response) -> str:
        user_id = self.user_id(request)
        if user_id is None:
            token = token_urlsafe(32)
            user_id = str(uuid4())
            self.users[token] = user_id
            response.set_cookie(USER_COOKIE, token, httponly=True, samesite="strict",
                                secure=request.url.scheme == "https")
        return user_id

    async def stream(self, user_id: str, session_name: str, message: str, *, scenario="appointment"):
        async with self.lock:
            key = (user_id, session_name)
            if scenario == "after_sales":
                try:
                    if self.after_sales_services is None:
                        services = AfterSalesServices()
                        await services.initialize()
                        self.after_sales_services = services
                    supervisor = self.after_sales_agents.get(key)
                    if supervisor is None:
                        supervisor = AfterSalesSupervisor(self.after_sales_services, user_id, str(uuid4()))
                        self.after_sales_agents[key] = supervisor
                    reply = await supervisor.run(message)
                except Exception:
                    log.exception("After-sales chat request failed")
                    reply = "售后服务暂时不可用，请稍后重试。"
                yield f"[REPLY][售后主管]{reply}"
                return
            agent = self.agents.get(key)
            if agent is None:
                session_id = str(uuid4())
                agent = TaskClassificationAgent(
                    AppointmentAgent(session_id=session_id, user_id=user_id),
                    ConsultantAgent(session_id=session_id, user_id=user_id),
                )
                self.agents[key] = agent
            change_reply = agent.appointment_agent.changes.handle(message)
            if change_reply is not None:
                agent.appointment_agent.reset()
                agent.reset_conversation()
                yield f"[REPLY][预约机器人]{change_reply}"
                return
            async for token in agent.classify_task_stream(message):
                yield token
