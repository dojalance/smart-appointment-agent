from dotenv import load_dotenv
import logging
import uuid
from functools import cached_property
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from config.model_provider import create_chat_model
from .appointment.local_slot_extractor import LocalSlotExtractor, answer_confirmation
from .appointment.reschedule import AppointmentChanges
from .appointment import (
    InputParser, 
    TechnicianFinder, 
    AppointmentProcessor, 
    MessageBuilder, 
    AppointmentDatabase
)

load_dotenv()
log = logging.getLogger(__name__)


class AppointmentAgent:
    """
    预约机器人主控制器
    
    职责：
    1. 初始化各个组件
    2. 管理会话状态
    3. 协调整个预约流程
    """
    
    def __init__(self, session_id=None, unrelated_callback=None, *, user_id="default_user"):
        # 基础设置
        self.session_id = session_id or str(uuid.uuid4())
        self.unrelated_callback = unrelated_callback
        self.state = None
        
        # 初始化LLM
        self.llm = self._initialize_llm()
        
        # 初始化组件
        self.input_parser = InputParser(self.llm)
        self.slot_extractor = LocalSlotExtractor.from_environment(user_id=user_id)
        self.technician_finder = TechnicianFinder()
        self.message_builder = MessageBuilder()
        self.appointment_database = AppointmentDatabase(user_id=user_id)
        self.appointment_processor = AppointmentProcessor(
            self.input_parser, 
            self.technician_finder,
            self.message_builder, 
            self.appointment_database,
            self.llm,
            require_confirmation=self.slot_extractor is not None,
        )
        
        # 会话管理
        self.chats_by_session_id = {}
        self.chat_history = self._get_chat_history(self.session_id)
        
        # 预约状态
        self.reset()

    def _initialize_llm(self):
        """初始化通用聊天模型"""
        return create_chat_model(temperature=0)

    @cached_property
    def changes(self):
        database = self.appointment_database
        return AppointmentChanges(database.appointment_service.appointments, database.user_id, self.session_id)

    def _get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """获取或创建会话历史记录"""
        chat_history = self.chats_by_session_id.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chats_by_session_id[session_id] = chat_history
        return chat_history
    
    def reset(self):
        """重置预约历史和状态"""
        self.appointment_history = {
            "gender": None,
            "start_time": None,
            "duration": None,
            "project": None,
            "preference": None,
            "technician": None,
            "technician_name": None
        }
        self.finished = False
        self.chat_history.clear()
        if self.slot_extractor is not None:
            self.slot_extractor.memory.reset()

    def set_shared_state(self, shared_state):
        """设置共享状态"""
        self.state = shared_state

    async def run_stream(self, user_input=None, *, classified_as_appointment=False):
        """
        流式处理用户预约请求的主函数

        这是整个预约流程的入口点，协调各个组件完成预约
        """
        if user_input is None:
            user_input = input("用户：")
        
        # 1. 解析用户输入（内部 JSON，不向用户流式输出，避免英文字段名暴露在聊天界面）
        try:
            # 2. 解析AI返回的数据
            if self.slot_extractor is None:
                ai_content = "".join(self.input_parser.parse_stream(user_input, self.chat_history))
                data = self.input_parser.parse_data(ai_content)
            elif self.appointment_history.get("awaiting_confirmation"):
                proposed = self.appointment_history["recommended_technician"]
                data = {"confirmation": answer_confirmation(user_input, proposed["name"])}
                self.chat_history.add_message(HumanMessage(content=user_input))
            else:
                try:
                    data = await self.slot_extractor.extract(
                        user_input, self.chat_history.messages, self.appointment_history
                    )
                except Exception as error:
                    log.warning("Local slot extraction failed: %s", type(error).__name__)
                    if classified_as_appointment:
                        yield "[REPLY][预约机器人]请告诉我您要预约的日期时间，以及服务时长。"
                    else:
                        yield "[REPLY][预约机器人]这次没能识别预约信息，请重试。"
                    return
                if data is None:
                    reply = "请告诉我您要预约的日期时间，以及服务时长。"
                    self.chat_history.add_messages([HumanMessage(content=user_input), AIMessage(content=reply)])
                    yield f"[REPLY][预约机器人]{reply}"
                    return
            
            # 3. 处理与预约无关的请求
            # 如果正在等待用户确认推荐技师，不要转交给归类机器人
            if data.get("unrelated", False) and not self.appointment_history.get('awaiting_confirmation'):
                if classified_as_appointment:
                    # 分类和槽位提取冲突时，先澄清，避免再次分发同一句话或使用有冲突的槽位。
                    reply = "请再说一下您要预约的服务、日期时间和时长。"
                    if self.slot_extractor:
                        self.chat_history.add_messages([HumanMessage(content=user_input), AIMessage(content=reply)])
                    yield f"[REPLY][预约机器人]{reply}"
                    return
                # 注意：这里不清空预约历史，保留用户已输入的信息
                # 只设置状态为CLASSIFY，让系统转交给其他机器人处理
                if self.state:
                    from config.constants import StateEnum
                    self.state.value = StateEnum.CLASSIFY
                
                async for token in self.appointment_processor.handle_unrelated_request(
                    user_input, self.unrelated_callback, self.state
                ):
                    yield token
                return

            if self.slot_extractor and not self.appointment_history.get("awaiting_confirmation"):
                self.chat_history.add_message(HumanMessage(content=user_input))
            
            technician_name = data.get("technician_name")
            duration = self.input_parser.extract_explicit_duration(user_input)
            if duration and not self.appointment_history.get("awaiting_confirmation"):
                data["duration"] = duration
            if (
                isinstance(technician_name, str)
                and technician_name.strip() not in {"", "未知"}
                and technician_name.strip() in user_input
                and not self.appointment_history.get("awaiting_confirmation")
            ):
                data["technician_name"] = self.technician_finder.resolve_mentioned_name(
                    technician_name.strip(), user_input
                )
            self.finished = self.appointment_processor.update_history_from_data(self.appointment_history, data)

            # 4. 处理预约完成的情况
            if self.finished:
                recommendation_pending = False
                async for token in self.appointment_processor.handle_complete_appointment(
                    self.appointment_history, self.session_id
                ):
                    # 检查是否有推荐等待确认
                    if token == "[SIGNAL]recommendation_pending":
                        recommendation_pending = True
                        # 将 finished 设为 False，让预约流程继续
                        self.finished = False
                        continue
                    reply_prefix = "[REPLY][预约机器人]"
                    if self.appointment_history.get("awaiting_confirmation") and token.startswith(reply_prefix):
                        self.chat_history.add_message(
                            AIMessage(content=token.removeprefix(reply_prefix))
                        )
                    yield token
                
                # 只有在真正完成预约时才重置状态
                if not recommendation_pending and not self.appointment_history.get('awaiting_confirmation'):
                    self._reset_state_after_appointment()
                return
            
            # 5. 处理信息不完整的情况
            async for token in self.appointment_processor.handle_incomplete_info(data, self.appointment_history):
                if self.slot_extractor and token.startswith("[REPLY][预约机器人]"):
                    self.chat_history.add_message(AIMessage(content=token.removeprefix("[REPLY][预约机器人]")))
                yield token
                
        except Exception as e:
            yield self.message_builder.create_parse_error_message()

    def _reset_state_after_appointment(self):
        """预约完成后重置状态"""
        self.reset()
        if self.state:
            from config.constants import StateEnum
            self.state.value = StateEnum.CLASSIFY
