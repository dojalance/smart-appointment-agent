import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from sqlalchemy import text

from config.time_config import time_config
from ..models import AppointmentWrite, Technician, TechnicianSchedule, UserBehavior
from .technician_repository import TechnicianRepository


class WriteResult(str, Enum):
    APPLIED = "applied"
    REPLAYED = "replayed"
    CONFLICT = "conflict"
    STALE = "stale"
    INVALID = "invalid"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class OwnedAppointment:
    schedule_id: int
    revision: int
    technician_name: str
    start: datetime
    end: datetime


def local_wall_time(value):
    if value.tzinfo is not None:
        return value.astimezone(time_config.BEIJING_TZ).replace(tzinfo=None)
    return value


class AppointmentRepository:
    def __init__(self, session_manager):
        self.session_manager = session_manager

    @staticmethod
    def _prior_result(session, request_id, user_id, operation, payload):
        prior = session.query(AppointmentWrite).filter_by(request_id=request_id).one_or_none()
        if prior is None:
            return None
        if (prior.user_id, prior.operation, prior.payload["input"]) == (user_id, operation, payload):
            return WriteResult.REPLAYED
        return WriteResult.INVALID

    @staticmethod
    def _current_revision(session, schedule_id):
        return session.query(AppointmentWrite.id).filter_by(schedule_id=schedule_id).order_by(
            AppointmentWrite.id.desc()
        ).limit(1).scalar()

    def create(self, request_id, user_id, session_id, technician_id, start, end, behavior):
        if start is None or end is None:
            return WriteResult.INVALID
        start, end = local_wall_time(start), local_wall_time(end)
        if end <= start:
            return WriteResult.INVALID
        payload = {"session_id": session_id, "technician_id": technician_id,
                   "start": start.isoformat(), "end": end.isoformat(), "behavior": behavior}
        with self.session_manager.session_scope() as session:
            # Serialize SQLite's conflict check, schedule write and receipt in one transaction.
            session.execute(text("BEGIN IMMEDIATE"))
            prior = self._prior_result(session, request_id, user_id, "create", payload)
            if prior is not None:
                return prior
            if session.get(Technician, technician_id) is None:
                return WriteResult.NOT_FOUND
            if TechnicianRepository.has_conflict(session, technician_id, start, end):
                return WriteResult.CONFLICT
            schedule = TechnicianSchedule(technician_id=technician_id, start_time=start, end_time=end,
                                          status="busy", appointment_id=int(time.time() * 1000))
            session.add(schedule)
            session.flush()
            session.add_all([
                AppointmentWrite(request_id=request_id, user_id=user_id, schedule_id=schedule.id,
                                 operation="create", payload={"input": payload}),
                UserBehavior(user_id=user_id, session_id=session_id, technician_id=technician_id,
                             action_type="appointment", action_data=behavior),
            ])
            return WriteResult.APPLIED

    def latest_upcoming(self, user_id, now):
        with self.session_manager.session_scope() as session:
            session.execute(text("BEGIN"))
            record = session.query(TechnicianSchedule, Technician.name).join(
                AppointmentWrite, AppointmentWrite.schedule_id == TechnicianSchedule.id
            ).join(Technician, TechnicianSchedule.technician_id == Technician.id).filter(
                AppointmentWrite.user_id == user_id, AppointmentWrite.operation == "create",
                TechnicianSchedule.status == "busy", TechnicianSchedule.start_time > local_wall_time(now),
            ).order_by(AppointmentWrite.id.desc()).first()
            if record is None:
                return None
            schedule, name = record
            return OwnedAppointment(schedule.id, self._current_revision(session, schedule.id), name,
                                    schedule.start_time, schedule.end_time)

    def reschedule(self, request_id, user_id, session_id, schedule_id, revision, new_start, now):
        new_start = local_wall_time(new_start)
        return self._change(request_id, user_id, session_id, schedule_id, revision, now, new_start=new_start)

    def cancel(self, request_id, user_id, session_id, schedule_id, revision, now):
        return self._change(request_id, user_id, session_id, schedule_id, revision, now)

    def _change(self, request_id, user_id, session_id, schedule_id, revision, now, *, new_start=None):
        operation = "cancel" if new_start is None else "reschedule"
        payload = {"session_id": session_id, "schedule_id": schedule_id,
                   "revision": revision}
        if new_start is not None:
            payload["new_start"] = new_start.isoformat()
        with self.session_manager.session_scope() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            prior = self._prior_result(session, request_id, user_id, operation, payload)
            if prior is not None:
                return prior
            owner = session.query(AppointmentWrite.id).filter_by(
                schedule_id=schedule_id, user_id=user_id, operation="create"
            ).first()
            schedule = session.get(TechnicianSchedule, schedule_id) if owner else None
            if schedule is None or schedule.status != "busy" or schedule.start_time <= local_wall_time(now):
                return WriteResult.NOT_FOUND
            if self._current_revision(session, schedule_id) != revision:
                return WriteResult.STALE
            audit = {"input": payload, "old_start": schedule.start_time.isoformat(),
                     "old_end": schedule.end_time.isoformat()}
            if new_start is None:
                schedule.status = "cancelled"
            else:
                if new_start <= local_wall_time(now) or new_start == schedule.start_time:
                    return WriteResult.INVALID
                new_end = new_start + (schedule.end_time - schedule.start_time)
                if TechnicianRepository.has_conflict(session, schedule.technician_id, new_start, new_end,
                                                     exclude_schedule_id=schedule_id):
                    return WriteResult.CONFLICT
                audit["new_end"] = new_end.isoformat()
                schedule.start_time, schedule.end_time = new_start, new_end
            session.add(AppointmentWrite(request_id=request_id, user_id=user_id, schedule_id=schedule_id,
                                         operation=operation, payload=audit))
            return WriteResult.APPLIED
