"""
用户行为分析API - 简化版本
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/user-behavior", tags=["用户行为分析"])
router_underscore = APIRouter(prefix="/api/user_behavior", tags=["用户行为分析"])


class UserAnalysisResponse(BaseModel):
    """用户分析响应"""
    favorite_technician_id: Optional[int] = None
    favorite_technician_name: Optional[str] = None
    favorite_service: Optional[str] = None
    favorite_duration: Optional[int] = None
    total_appointments: int = 0
    days_since_last_appointment: Optional[int] = None
    should_send_reminder: bool = False


async def get_user_analysis(user_id: str = "default_user") -> UserAnalysisResponse:
    """获取用户行为分析数据"""
    try:
        from agents.user_behavior_agent import UserBehaviorAgent
        
        agent = UserBehaviorAgent()
        analysis = agent.get_user_analysis(user_id)
        
        if not analysis:
            return UserAnalysisResponse()
        
        # 获取技师姓名
        technician_name = None
        if analysis.get('favorite_technician_id'):
            from db import TechnicianDBRouter
            db = TechnicianDBRouter()
            tech_info = db.get_technician_by_id(analysis['favorite_technician_id'])
            if tech_info:
                technician_name = tech_info.get('name')
        
        return UserAnalysisResponse(
            favorite_technician_id=analysis.get('favorite_technician_id'),
            favorite_technician_name=technician_name,
            favorite_service=analysis.get('favorite_service'),
            favorite_duration=analysis.get('favorite_duration'),
            total_appointments=analysis.get('total_appointments', 0),
            days_since_last_appointment=analysis.get('days_since_last_appointment'),
            should_send_reminder=analysis.get('should_send_reminder', False)
        )
    except Exception as e:
        # 记录异常日志并返回空数据，而不是硬编码的假数据
        import logging
        logging.error(f"获取用户分析数据失败: {e}")
        return UserAnalysisResponse()


@router.get("/analysis", response_model=UserAnalysisResponse)
@router.get("/dashboard_data", response_model=UserAnalysisResponse)
@router_underscore.get("/dashboard_data", response_model=UserAnalysisResponse)
async def get_current_user_analysis(request: Request):
    """获取当前浏览器的行为分析数据"""
    user_id = request.app.state.chat_sessions.user_id(request)
    return await get_user_analysis(user_id or "default_user")


class ReminderRequest(BaseModel):
    """发送提醒请求"""
    user_id: str = "default_user"


class ReminderResponse(BaseModel):
    """提醒消息响应"""
    message: str
    technician_available_times: Optional[list] = None


@router.post("/send-reminder", response_model=ReminderResponse, summary="发送回访提醒")
async def send_reminder(reminder: ReminderRequest, request: Request):
    """生成并返回回访提醒消息"""
    user_id = request.app.state.chat_sessions.user_id(request) or "default_user"
    if reminder.user_id not in ("default_user", user_id):
        raise HTTPException(status_code=403, detail="无权查询该用户的提醒")
    try:
        from agents.user_behavior_agent import UserBehaviorAgent
        
        agent = UserBehaviorAgent()
        result = await agent.get_reminder_with_schedule(user_id)
        
        return ReminderResponse(
            message=result["message"],
            technician_available_times=result["technician_available_times"]
        )
        
    except Exception as e:
        import logging
        logging.error(f"生成回访提醒失败: {e}")
        return ReminderResponse(
            message="尊敬的Tom，您好！系统暂时无法查询技师时间，请稍后再试或直接联系我们预约。",
            technician_available_times=[]
        )
