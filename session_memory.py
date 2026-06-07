"""
memory/session_memory.py
------------------------
In-process session memory with TTL expiration.
Manages per-ticket conversation state for active sessions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from models import SupportTicket, AgentTier
from settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    ticket: SupportTicket
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    current_agent: AgentTier = AgentTier.INTAKE
    resolution_attempts: int = 0
    is_escalated: bool = False

    def touch(self):
        self.last_active = time.time()

    def age_seconds(self) -> float:
        return time.time() - self.last_active


class SessionMemory:
    """
    Manages in-memory state for active support sessions.

    - Stores the live SupportTicket per session_id (usually ticket_id)
    - Enforces max_turns rolling window on conversation history
    - TTL-based expiration for abandoned sessions
    - Thread-safe enough for single-process use
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._sessions: dict[str, ActiveSession] = {}

    def create_session(self, ticket: SupportTicket) -> ActiveSession:
        session = ActiveSession(ticket=ticket)
        self._sessions[ticket.ticket_id] = session
        logger.debug("Created session for ticket %s", ticket.ticket_id)
        return session

    def get_session(self, ticket_id: str) -> Optional[ActiveSession]:
        session = self._sessions.get(ticket_id)
        if session is None:
            return None
        if session.age_seconds() > self.settings.session_ttl_seconds:
            logger.info("Session %s expired, removing.", ticket_id)
            del self._sessions[ticket_id]
            return None
        session.touch()
        return session

    def update_session(self, session: ActiveSession) -> None:
        self._sessions[session.ticket.ticket_id] = session
        session.touch()

    def close_session(self, ticket_id: str) -> Optional[SupportTicket]:
        session = self._sessions.pop(ticket_id, None)
        return session.ticket if session else None

    def get_lc_history(self, ticket_id: str) -> list:
        """Return LangChain message list for a session's recent history."""
        session = self.get_session(ticket_id)
        if not session:
            return []
        max_msgs = self.settings.max_memory_turns * 2
        recent = session.ticket.messages[-max_msgs:]
        return [m.to_lc_message() for m in recent]

    def evict_expired(self) -> int:
        """Remove all expired sessions. Returns count evicted."""
        expired = [
            tid for tid, s in self._sessions.items()
            if s.age_seconds() > self.settings.session_ttl_seconds
        ]
        for tid in expired:
            del self._sessions[tid]
        if expired:
            logger.info("Evicted %d expired sessions.", len(expired))
        return len(expired)

    def active_count(self) -> int:
        return len(self._sessions)

    def all_active_ids(self) -> list[str]:
        return list(self._sessions.keys())
