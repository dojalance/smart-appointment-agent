from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Barrier

import pytest
from sqlalchemy import event

from agents.appointment.follow_up import FollowUpInbox
from db.models import AppointmentWrite, FollowUpDelivery, TechnicianSchedule, UserBehavior
from db.repositories.appointment_repository import WriteResult
from services.appointment_service import AppointmentService


NOW = datetime(2030, 1, 1)
START = datetime(2030, 1, 8, 14)
END = datetime(2030, 1, 8, 15)
MOVED = datetime(2030, 1, 8, 16)


@pytest.fixture
def booking(tmp_path):
    url = f"sqlite:///{(tmp_path / 'booking.sqlite').as_posix()}"
    service = AppointmentService(url)
    technician = service.add_technician("测试技师甲", "女", "肩颈")
    return url, service, technician


def create(service, technician, request="create-1", user="owner"):
    return service.appointments.create(request, user, "chat", technician, START, END, {"project": "按摩"})


def test_same_create_key_concurrently_writes_one_schedule_and_behavior(booking):
    url, service, technician = booking
    other = AppointmentService(url)
    barrier = Barrier(2)

    def worker(client):
        barrier.wait(timeout=5)
        return create(client, technician)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(worker, [service, other])) == {WriteResult.APPLIED, WriteResult.REPLAYED}

    with service.db_router.session_manager.session_scope() as session:
        assert session.query(TechnicianSchedule).count() == 1
        assert session.query(UserBehavior).count() == 1
        assert session.query(AppointmentWrite).count() == 1
    assert create(other, technician, user="visitor") is WriteResult.INVALID
    assert create(other, technician, request="create-2") is WriteResult.CONFLICT


def test_same_move_key_concurrently_writes_once_and_rejects_stale_version(booking):
    url, service, technician = booking
    assert create(service, technician) is WriteResult.APPLIED
    old = service.appointments.latest_upcoming("owner", NOW)
    other = AppointmentService(url)
    barrier = Barrier(2)

    def move(client, request="move-1", user="owner", new_start=MOVED):
        return client.appointments.reschedule(request, user, "chat", old.schedule_id, old.revision, new_start, NOW)

    def worker(client):
        barrier.wait(timeout=5)
        return move(client)

    assert move(other, request="visitor-move", user="visitor") is WriteResult.NOT_FOUND
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(worker, [service, other])) == {WriteResult.APPLIED, WriteResult.REPLAYED}
    assert move(AppointmentService(url)) is WriteResult.REPLAYED
    assert move(other, new_start=datetime(2030, 1, 8, 17)) is WriteResult.INVALID
    assert move(other, request="stale") is WriteResult.STALE
    current = other.appointments.latest_upcoming("owner", NOW)
    assert (current.schedule_id, current.start, current.end - current.start) == (old.schedule_id, MOVED, END - START)
    with service.db_router.session_manager.session_scope() as session:
        assert session.query(AppointmentWrite).filter_by(operation="reschedule").count() == 1


def test_legacy_schedule_without_ownership_still_blocks_but_cannot_be_rescheduled(booking):
    _, service, technician = booking
    legacy = service.technician_repo.add_schedule(technician, START, END, "busy", appointment_id=123)
    assert service.appointments.latest_upcoming("owner", NOW) is None
    assert service.appointments.reschedule("move", "owner", "chat", legacy, 0, MOVED, NOW) is WriteResult.NOT_FOUND
    assert create(service, technician) is WriteResult.CONFLICT


def test_audit_failure_rolls_back_schedule_and_behavior(booking):
    _, service, technician = booking
    engine = service.db_router.session_manager.engine

    def fail_audit(_connection, _cursor, statement, *_):
        if statement.startswith("INSERT INTO appointment_writes"):
            raise RuntimeError("audit insert failed")

    event.listen(engine, "before_cursor_execute", fail_audit)
    try:
        with pytest.raises(RuntimeError, match="audit insert failed"):
            create(service, technician)
    finally:
        event.remove(engine, "before_cursor_execute", fail_audit)
    with service.db_router.session_manager.session_scope() as session:
        assert session.query(TechnicianSchedule).count() == 0
        assert session.query(UserBehavior).count() == 0
    assert create(service, technician) is WriteResult.APPLIED


def test_cancel_checks_owner_and_revision_and_releases_time(booking):
    url, service, technician = booking
    assert create(service, technician) is WriteResult.APPLIED
    old = service.appointments.latest_upcoming("owner", NOW)
    repo = service.appointments
    assert repo.cancel("visitor", "visitor", "chat", old.schedule_id, old.revision, NOW) is WriteResult.NOT_FOUND
    assert repo.cancel("stale", "owner", "chat", old.schedule_id, old.revision - 1, NOW) is WriteResult.STALE
    assert repo.cancel("cancel", "owner", "chat", old.schedule_id, old.revision, NOW) is WriteResult.APPLIED
    assert AppointmentService(url).appointments.cancel(
        "cancel", "owner", "chat", old.schedule_id, old.revision, NOW
    ) is WriteResult.REPLAYED
    assert service.is_technician_available(technician, START, END)
    with service.db_router.session_manager.session_scope() as session:
        assert session.query(AppointmentWrite).filter_by(operation="cancel").count() == 1


def test_inbox_delivery_is_atomic_idempotent_and_owner_bound(booking):
    url, service, _ = booking
    first = FollowUpInbox(service.db_router.session_manager)
    second = FollowUpInbox(AppointmentService(url).db_router.session_manager)
    barrier = Barrier(2)

    def send(inbox):
        barrier.wait(timeout=5)
        return inbox.deliver("follow-up", "owner", "chat", "预览内容")

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(send, [first, second])) == {WriteResult.APPLIED, WriteResult.REPLAYED}
    assert first.deliver("follow-up", "visitor", "chat", "预览内容") is WriteResult.INVALID
    assert first.latest("visitor") is None and second.latest("owner") == "预览内容"
    with service.db_router.session_manager.session_scope() as session:
        assert session.query(FollowUpDelivery).count() == 1
