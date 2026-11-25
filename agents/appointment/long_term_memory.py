import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property

import numpy as np

from config.model_provider import create_embedding_model


log = logging.getLogger(__name__)
TOP_K = 5
WEIGHTS = (0.6, 0.3, 0.1)
HALF_LIFE_DAYS = 30
MAX_CANDIDATES = 100
RECALL_REQUEST = re.compile(r"(?:按(?:照)?|参考).{0,8}(?:上次|以前|之前|平时|往常)|(?:预约|服务)偏好")


@dataclass(frozen=True)
class PreferenceMemory:
    behavior_id: int
    text: str
    recorded_at: datetime
    importance: float


@dataclass(frozen=True)
class RankedMemory:
    memory: PreferenceMemory
    semantic: float
    recency: float
    score: float


def rank_memories(memories, query_vector, memory_vectors, now):
    if not memories:
        return []
    query = np.asarray(query_vector, dtype=float)
    vectors = np.asarray(memory_vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    similarity = np.divide(vectors @ query, norms, out=np.zeros(len(memories)), where=norms > 0)

    ranked = []
    for memory, cosine in zip(memories, np.clip(similarity, 0, 1), strict=True):
        age_days = max((now - memory.recorded_at).total_seconds() / 86400, 0)
        recency = 2 ** (-age_days / HALF_LIFE_DAYS)
        score = WEIGHTS[0] * cosine + WEIGHTS[1] * recency + WEIGHTS[2] * memory.importance
        ranked.append(RankedMemory(memory, float(cosine), recency, float(score)))
    return sorted(ranked, key=lambda item: (item.score, item.memory.behavior_id), reverse=True)


class AppointmentRecall:
    def __init__(self, user_id):
        self.user_id = user_id

    @cached_property
    def behavior_repository(self):
        from services.user_behavior_service import UserBehaviorService

        return UserBehaviorService().user_behavior_repo

    @cached_property
    def embeddings(self):
        return create_embedding_model()

    def retrieve(self, query, *, now=None):
        if not RECALL_REQUEST.search(query):
            return []
        try:
            behaviors = self.behavior_repository.get_recent_behaviors(
                self.user_id, limit=MAX_CANDIDATES, action_type="appointment"
            )
            memories = self._preferences(behaviors)
            if not memories:
                return []
            query_vector = self.embeddings.embed_query(query[:300])
            memory_vectors = self.embeddings.embed_documents([memory.text for memory in memories])
            return rank_memories(memories, query_vector, memory_vectors, now or datetime.now(timezone.utc))[:TOP_K]
        except Exception as exc:
            log.warning("Appointment preference retrieval unavailable: %s", type(exc).__name__)
            return []

    @staticmethod
    def _preferences(behaviors):
        memories = {}
        for behavior in behaviors:
            data = behavior["action_data"] or {}
            if not isinstance(data, dict):
                continue
            project = str(data.get("project") or "按摩").strip()[:32]
            if project == "massage":
                project = "按摩"
            preference = " ".join(str(data.get("preference") or "").split())[:72]
            preference = "" if preference in {"无", "未知"} else preference
            text = f"服务项目：{project}，偏好：{preference}" if preference else f"服务项目：{project}"
            recorded_at = behavior["created_at"]
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            memories.setdefault(text, PreferenceMemory(
                behavior["id"], text, recorded_at, 1.0 if preference else 0.5
            ))
        return list(memories.values())
