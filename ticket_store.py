"""
memory/ticket_store.py
----------------------
SQLite-backed persistent store for support tickets.
Handles CRUD, search, and analytics queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

from ..models import (
    SupportTicket, Message, SupportCategory, TicketStatus,
    TicketPriority, AgentTier, EscalationReason,
)
from ..settings import Settings

logger = logging.getLogger(__name__)


class TicketStore:
    """
    Persistent SQLite store for SupportTicket objects.

    Features:
    - Full ticket lifecycle CRUD
    - Message thread storage
    - Search by status, category, priority, date range
    - Analytics aggregates (resolution rate, avg handle time, CSAT)
    - Auto-migration on first run
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.db_path = self.settings.ticket_db_path
        self._init_db()
        logger.info("TicketStore initialised at %s", self.db_path)

    # ── DB Setup ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id         TEXT PRIMARY KEY,
                    customer_id       TEXT,
                    customer_email    TEXT,
                    subject           TEXT,
                    category          TEXT,
                    priority          TEXT,
                    status            TEXT,
                    created_at        TEXT,
                    updated_at        TEXT,
                    resolved_at       TEXT,
                    current_agent     TEXT,
                    assigned_human    TEXT,
                    escalation_reason TEXT,
                    escalation_notes  TEXT,
                    resolution_attempts INTEGER DEFAULT 0,
                    kb_sources_used   TEXT,
                    confidence_score  REAL DEFAULT 1.0,
                    resolution_summary TEXT,
                    was_auto_resolved  INTEGER DEFAULT 0,
                    feedback_score    INTEGER,
                    tags              TEXT,
                    custom_fields     TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id   TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    agent_tier  TEXT,
                    timestamp   TEXT,
                    metadata    TEXT,
                    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tickets_status   ON tickets(status);
                CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
                CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
                CREATE INDEX IF NOT EXISTS idx_tickets_created  ON tickets(created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_ticket  ON messages(ticket_id);
            """)

    # ── Serialisation Helpers ──────────────────────────────────────────

    def _ticket_to_row(self, t: SupportTicket) -> dict:
        return {
            "ticket_id": t.ticket_id,
            "customer_id": t.customer_id,
            "customer_email": t.customer_email,
            "subject": t.subject,
            "category": t.category.value,
            "priority": t.priority.value,
            "status": t.status.value,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
            "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
            "current_agent": t.current_agent.value,
            "assigned_human": t.assigned_human,
            "escalation_reason": t.escalation_reason.value if t.escalation_reason else None,
            "escalation_notes": t.escalation_notes,
            "resolution_attempts": t.resolution_attempts,
            "kb_sources_used": json.dumps(t.kb_sources_used),
            "confidence_score": t.confidence_score,
            "resolution_summary": t.resolution_summary,
            "was_auto_resolved": int(t.was_auto_resolved),
            "feedback_score": t.feedback_score,
            "tags": json.dumps(t.tags),
            "custom_fields": json.dumps(t.custom_fields),
        }

    def _row_to_ticket(self, row: sqlite3.Row, messages: list[Message]) -> SupportTicket:
        t = SupportTicket(
            ticket_id=row["ticket_id"],
            customer_id=row["customer_id"] or "",
            customer_email=row["customer_email"] or "",
            subject=row["subject"] or "",
            category=SupportCategory(row["category"]),
            priority=TicketPriority(row["priority"]),
            status=TicketStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
            current_agent=AgentTier(row["current_agent"]),
            assigned_human=row["assigned_human"],
            escalation_reason=EscalationReason(row["escalation_reason"]) if row["escalation_reason"] else None,
            escalation_notes=row["escalation_notes"] or "",
            resolution_attempts=row["resolution_attempts"],
            kb_sources_used=json.loads(row["kb_sources_used"] or "[]"),
            confidence_score=row["confidence_score"],
            resolution_summary=row["resolution_summary"] or "",
            was_auto_resolved=bool(row["was_auto_resolved"]),
            feedback_score=row["feedback_score"],
            tags=json.loads(row["tags"] or "[]"),
            custom_fields=json.loads(row["custom_fields"] or "{}"),
            messages=messages,
        )
        return t

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            role=row["role"],
            content=row["content"],
            agent_tier=AgentTier(row["agent_tier"]) if row["agent_tier"] else None,
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # ── CRUD ───────────────────────────────────────────────────────────

    def save(self, ticket: SupportTicket) -> None:
        """Insert or update a ticket and all its messages."""
        row = self._ticket_to_row(ticket)
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO tickets VALUES (
                    :ticket_id, :customer_id, :customer_email, :subject,
                    :category, :priority, :status, :created_at, :updated_at,
                    :resolved_at, :current_agent, :assigned_human,
                    :escalation_reason, :escalation_notes, :resolution_attempts,
                    :kb_sources_used, :confidence_score, :resolution_summary,
                    :was_auto_resolved, :feedback_score, :tags, :custom_fields
                )
                ON CONFLICT(ticket_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    customer_email=excluded.customer_email,
                    subject=excluded.subject,
                    category=excluded.category,
                    priority=excluded.priority,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    resolved_at=excluded.resolved_at,
                    current_agent=excluded.current_agent,
                    assigned_human=excluded.assigned_human,
                    escalation_reason=excluded.escalation_reason,
                    escalation_notes=excluded.escalation_notes,
                    resolution_attempts=excluded.resolution_attempts,
                    kb_sources_used=excluded.kb_sources_used,
                    confidence_score=excluded.confidence_score,
                    resolution_summary=excluded.resolution_summary,
                    was_auto_resolved=excluded.was_auto_resolved,
                    feedback_score=excluded.feedback_score,
                    tags=excluded.tags,
                    custom_fields=excluded.custom_fields
            """, row)

            # Re-sync messages (delete & reinsert for simplicity)
            conn.execute("DELETE FROM messages WHERE ticket_id = ?", (ticket.ticket_id,))
            for msg in ticket.messages:
                conn.execute("""
                    INSERT INTO messages (ticket_id, role, content, agent_tier, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ticket.ticket_id,
                    msg.role,
                    msg.content,
                    msg.agent_tier.value if msg.agent_tier else None,
                    msg.timestamp.isoformat(),
                    json.dumps(msg.metadata),
                ))

    def get(self, ticket_id: str) -> Optional[SupportTicket]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not row:
                return None
            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE ticket_id = ? ORDER BY id", (ticket_id,)
            ).fetchall()
            messages = [self._row_to_message(r) for r in msg_rows]
            return self._row_to_ticket(row, messages)

    def update_status(self, ticket_id: str, status: TicketStatus) -> None:
        now = datetime.utcnow().isoformat()
        resolved_at = now if status == TicketStatus.RESOLVED else None
        with self._conn() as conn:
            conn.execute(
                "UPDATE tickets SET status=?, updated_at=?, resolved_at=? WHERE ticket_id=?",
                (status.value, now, resolved_at, ticket_id),
            )

    def update_feedback(self, ticket_id: str, score: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tickets SET feedback_score=? WHERE ticket_id=?",
                (max(1, min(5, score)), ticket_id),
            )

    # ── Search ─────────────────────────────────────────────────────────

    def list_tickets(
        self,
        status: Optional[TicketStatus] = None,
        category: Optional[SupportCategory] = None,
        priority: Optional[TicketPriority] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[SupportTicket]:
        clauses, params = [], []
        if status:
            clauses.append("status = ?"); params.append(status.value)
        if category:
            clauses.append("category = ?"); params.append(category.value)
        if priority:
            clauses.append("priority = ?"); params.append(priority.value)
        if since:
            clauses.append("created_at >= ?"); params.append(since.isoformat())

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM tickets {where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
            tickets = []
            for row in rows:
                msg_rows = conn.execute(
                    "SELECT * FROM messages WHERE ticket_id = ? ORDER BY id",
                    (row["ticket_id"],),
                ).fetchall()
                messages = [self._row_to_message(r) for r in msg_rows]
                tickets.append(self._row_to_ticket(row, messages))
        return tickets

    def get_recently_resolved(self, days: int = 30, limit: int = 100) -> list[SupportTicket]:
        since = datetime.utcnow() - timedelta(days=days)
        return self.list_tickets(status=TicketStatus.RESOLVED, since=since, limit=limit)

    # ── Analytics ──────────────────────────────────────────────────────

    def get_analytics(self, days: int = 30) -> dict:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE created_at >= ? AND status = 'resolved'",
                (since,),
            ).fetchone()[0]
            auto_resolved = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE created_at >= ? AND was_auto_resolved = 1",
                (since,),
            ).fetchone()[0]
            escalated = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE created_at >= ? AND status = 'escalated'",
                (since,),
            ).fetchone()[0]
            avg_csat = conn.execute(
                "SELECT AVG(feedback_score) FROM tickets WHERE created_at >= ? AND feedback_score IS NOT NULL",
                (since,),
            ).fetchone()[0]
            by_category = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM tickets WHERE created_at >= ? GROUP BY category",
                (since,),
            ).fetchall()
            by_priority = conn.execute(
                "SELECT priority, COUNT(*) as cnt FROM tickets WHERE created_at >= ? GROUP BY priority",
                (since,),
            ).fetchall()

        return {
            "period_days": days,
            "total_tickets": total,
            "resolved": resolved,
            "auto_resolved": auto_resolved,
            "escalated": escalated,
            "resolution_rate": round(resolved / max(total, 1), 3),
            "automation_rate": round(auto_resolved / max(total, 1), 3),
            "escalation_rate": round(escalated / max(total, 1), 3),
            "avg_csat": round(avg_csat, 2) if avg_csat else None,
            "by_category": {r["category"]: r["cnt"] for r in by_category},
            "by_priority": {r["priority"]: r["cnt"] for r in by_priority},
        }
