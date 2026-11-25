"""
预约数据库操作器

负责处理预约相关的数据库操作
注意：现在通过Services层访问数据库，符合分层架构
"""

from typing import Dict, Any
from datetime import datetime
from config.time_config import time_config
from config.constants import busy_periods_dict


class AppointmentDatabase:
    """预约数据库操作器"""
    
    def __init__(self, *, user_id="default_user"):
        # 延迟导入Services避免循环依赖
        self._appointment_service = None
        self.user_id = user_id
    
    @property
    def appointment_service(self):
        """懒加载预约服务"""
        if self._appointment_service is None:
            from services.appointment_service import AppointmentService
            self._appointment_service = AppointmentService()
        return self._appointment_service
    
    def save_appointment(self, technician_id: str, start_time: datetime, 
                        end_time: datetime, appointment_history: Dict[str, Any], 
                        session_id: str) -> bool:
        """保存预约信息到数据库"""
        return self.appointment_service.save_appointment(
            technician_id, start_time, end_time, appointment_history, session_id, user_id=self.user_id
        )
    
    def update_memory_schedule(self, technician_id: str, start_time: datetime, end_time: datetime):
        """更新内存中的技师忙碌时间段"""
        busy_period = {
            "start": time_config.format_datetime(start_time, "%H:%M"),
            "end": time_config.format_datetime(end_time, "%H:%M")
        }
        busy_periods_dict.setdefault(technician_id, []).append(busy_period)
