"""
supervisor.py
-------------
Supervisor — the master orchestrator of the multi-agent system.

The Supervisor is the single public entry point. It:
  1. Creates / resumes sessions and tickets
  2. Routes to the correct agent at each turn
  3. Manages escalation transitions between tiers
  4. Persists ticket state after every turn
  5. Triggers post-resolution KB updates

Agent flow:
  new message
      ↓
  IntakeAgent (classify + route)
      ↓
  Tier1Agent (RAG auto-resolve)
      ↓ if confidence low / attempts exceeded
  Tier2Agent (domain specialist)
      ↓ if still unresolved
  EscalationAgent (human handoff)

  Post-resolve:
  KBManagerAgent (update KB from resolution)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .agents import (
    EscalationAgent, IntakeAgent, KBManagerAgent,
    Tier1Agent, Tier2Agent,
)
from .kb import KnowledgeBase
from .memory import SessionMemory, TicketStore
from .models import (
    AgentResponse, AgentTier, EscalationReason,
    SupportCategory, SupportTicket, TicketPriority, TicketStatus,
)
from .settings import Settings

logger = logging.getLogger(__name__)


class Supervisor:
    """
    Master orchestrator for the multi-agent customer support system.

    Usage:
        supervisor = Supervisor()

        # New conversation
        response, ticket = supervisor.handle("I was double-charged!", customer_email="user@example.com")

        # Follow-up
        response, ticket = supervisor.handle("It happened twice this month.", ticket_id=ticket.ticket_id)

        # Resolve
        supervisor.resolve(ticket.ticket_id, "Issued full refund for duplicate charge.")

        # Feedback
        supervisor.record_feedback(ticket.ticket_id, score=5)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

        # Shared services
        self.kb = KnowledgeBase(self.settings)
        self.ticket_store = TicketStore(self.settings)
        self.session_memory = SessionMemory(self.settings)

        # Agents
        self.intake_agent = IntakeAgent(self.settings)
        self.tier1_agent = Tier1Agent(self.kb, self.settings)
        self.tier2_agent = Tier2Agent(self.kb, self.settings)
        self.escalation_agent = EscalationAgent(self.settings)
        self.kb_manager = KBManagerAgent(self.kb, self.settings)

        logger.info(
            "Supervisor initialised. Company: %s | Model: %s",
            self.settings.company_name, self.settings.llm_model,
        )

    # ── Main Entry Point ───────────────────────────────────────────────

    def handle(
        self,
        message: str,
        ticket_id: Optional[str] = None,
        customer_email: str = "",
        customer_id: str = "",
        force_tier: Optional[AgentTier] = None,
    ) -> tuple[str, SupportTicket]:
        """
        Handle a single customer message.

        Args:
            message:        The customer's text message
            ticket_id:      Existing ticket ID for follow-ups (None = new ticket)
            customer_email: Customer's email (required for new tickets)
            customer_id:    Optional internal customer identifier
            force_tier:     Override routing and send to a specific agent tier

        Returns:
            (response_text, updated_ticket)
        """
        # 1. Load or create session + ticket
        ticket, is_new = self._get_or_create_ticket(
            message=message,
            ticket_id=ticket_id,
            customer_email=customer_email,
            customer_id=customer_id,
        )
        session = self.session_memory.get_session(ticket.ticket_id) or \
                  self.session_memory.create_session(ticket)

        # 2. Add user message to ticket
        ticket.add_message("user", message)

        # 3. Intake classification (only on new tickets or when re-routing)
        if is_new or ticket.current_agent == AgentTier.INTAKE:
            routing = self.intake_agent.classify(ticket)
            ticket.category = routing.category
            ticket.priority = routing.priority
            ticket.current_agent = routing.target_agent

        # 4. Check for explicit escalation request
        if self._user_wants_human(message):
            return self._do_escalation(
                ticket=ticket,
                reason=EscalationReason.USER_REQUESTED,
                notes="Customer explicitly requested a human agent.",
            )

        # 5. Route to the appropriate agent
        if force_tier:
            ticket.current_agent = force_tier

        agent_response = self._route(ticket, message)

        # 6. Handle escalation signals from agents
        if agent_response.should_escalate and agent_response.escalation_reason:
            # Tier-1 → Tier-2 (don't immediately go to human)
            if (ticket.current_agent == AgentTier.TIER1
                    and agent_response.escalation_reason != EscalationReason.CRITICAL_ISSUE):
                logger.info("Tier-1 escalating ticket %s to Tier-2.", ticket.ticket_id)
                ticket.current_agent = AgentTier.TIER2
                agent_response = self._route(ticket, message)

            # Tier-2 → human or critical direct to human
            if agent_response.should_escalate:
                return self._do_escalation(
                    ticket=ticket,
                    reason=agent_response.escalation_reason,
                    notes=agent_response.escalation_notes,
                )

        # 7. Update ticket with response metadata
        if agent_response.suggested_category:
            ticket.category = agent_response.suggested_category
        if agent_response.suggested_priority:
            ticket.priority = agent_response.suggested_priority
        ticket.confidence_score = agent_response.confidence
        ticket.kb_sources_used = list(set(
            ticket.kb_sources_used + agent_response.kb_sources
        ))

        # 8. Persist response
        response_text = agent_response.content
        ticket.add_message(
            role="assistant",
            content=response_text,
            agent_tier=agent_response.agent_tier,
            metadata={
                "confidence": agent_response.confidence,
                "kb_sources": agent_response.kb_sources,
            },
        )
        ticket.status = TicketStatus.IN_PROGRESS
        ticket.current_agent = agent_response.agent_tier
        self._save(ticket)

        return response_text, ticket

    # ── Resolution & Feedback ──────────────────────────────────────────

    def resolve(
        self,
        ticket_id: str,
        resolution_summary: str,
        was_auto: bool = False,
    ) -> Optional[SupportTicket]:
        """
        Mark a ticket as resolved and trigger KB update.

        Args:
            ticket_id:          The ticket to resolve
            resolution_summary: Human-readable summary of what was done
            was_auto:           True if resolved automatically

        Returns:
            Updated SupportTicket or None if not found
        """
        ticket = self._load_ticket(ticket_id)
        if not ticket:
            logger.warning("resolve() called for unknown ticket %s.", ticket_id)
            return None

        ticket.status = TicketStatus.RESOLVED
        ticket.resolution_summary = resolution_summary
        ticket.was_auto_resolved = was_auto
        ticket.resolved_at = datetime.utcnow()
        self._save(ticket)
        self.session_memory.close_session(ticket_id)

        logger.info("Ticket %s resolved. Auto: %s", ticket_id, was_auto)

        # Trigger KB update from resolution
        if self.settings.auto_kb_update:
            try:
                result = self.kb_manager.process_resolved_ticket(ticket)
                logger.info("KB update result for %s: %s", ticket_id, result)
            except Exception as exc:
                logger.warning("KB update failed for %s: %s", ticket_id, exc)

        return ticket

    def record_feedback(self, ticket_id: str, score: int) -> bool:
        """Record CSAT feedback (1–5 scale). Returns True if successful."""
        try:
            self.ticket_store.update_feedback(ticket_id, score)
            logger.info("Feedback recorded for ticket %s: %d/5.", ticket_id, score)
            return True
        except Exception as exc:
            logger.error("Feedback recording failed: %s", exc)
            return False

    def close_ticket(self, ticket_id: str) -> bool:
        """Close a resolved ticket."""
        try:
            self.ticket_store.update_status(ticket_id, TicketStatus.CLOSED)
            self.session_memory.close_session(ticket_id)
            return True
        except Exception as exc:
            logger.error("Close ticket failed: %s", exc)
            return False

    # ── KB Management ──────────────────────────────────────────────────

    def ingest_kb_file(self, path: str,
                       category: SupportCategory = SupportCategory.GENERAL,
                       tags: Optional[list[str]] = None) -> int:
        """Ingest a file into the knowledge base."""
        return self.kb.ingest_file(path, category=category, tags=tags)

    def ingest_kb_text(self, text: str, title: str,
                       category: SupportCategory = SupportCategory.GENERAL,
                       tags: Optional[list[str]] = None) -> int:
        """Ingest raw text as a KB article."""
        return self.kb.ingest_text(text, title=title, category=category, tags=tags)

    def run_kb_update_batch(self, days: int = 30) -> list[dict]:
        """Run KB update cycle on all tickets resolved in the last N days."""
        resolved = self.ticket_store.get_recently_resolved(days=days)
        return self.kb_manager.run_kb_update_cycle(resolved)

    # ── Analytics ──────────────────────────────────────────────────────

    def get_analytics(self, days: int = 30) -> dict:
        """Return support analytics for the last N days."""
        stats = self.ticket_store.get_analytics(days=days)
        stats["kb_stats"] = self.kb_manager.kb_stats()
        stats["active_sessions"] = self.session_memory.active_count()
        return stats

    def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        return self._load_ticket(ticket_id)

    # ── Private Helpers ────────────────────────────────────────────────

    def _get_or_create_ticket(
        self,
        message: str,
        ticket_id: Optional[str],
        customer_email: str,
        customer_id: str,
    ) -> tuple[SupportTicket, bool]:
        """Load an existing ticket or create a new one."""
        if ticket_id:
            session = self.session_memory.get_session(ticket_id)
            if session:
                return session.ticket, False
            ticket = self.ticket_store.get(ticket_id)
            if ticket:
                return ticket, False

        # New ticket
        ticket = SupportTicket(
            customer_email=customer_email,
            customer_id=customer_id,
            subject="",
            category=SupportCategory.UNKNOWN,
            priority=TicketPriority.MEDIUM,
            status=TicketStatus.OPEN,
            current_agent=AgentTier.INTAKE,
        )
        self.ticket_store.save(ticket)
        return ticket, True

    def _route(self, ticket: SupportTicket, message: str) -> AgentResponse:
        """Route to the correct agent based on ticket's current_agent."""
        history = self.session_memory.get_lc_history(ticket.ticket_id)

        if ticket.current_agent in (AgentTier.INTAKE, AgentTier.TIER1):
            return self.tier1_agent.respond(ticket, message, chat_history=history)

        elif ticket.current_agent == AgentTier.TIER2:
            return self.tier2_agent.respond(ticket, message, chat_history=history)

        else:
            # Fallback: Tier-1
            return self.tier1_agent.respond(ticket, message, chat_history=history)

    def _do_escalation(
        self,
        ticket: SupportTicket,
        reason: EscalationReason,
        notes: str = "",
    ) -> tuple[str, SupportTicket]:
        """Execute escalation to human agent."""
        agent_response = self.escalation_agent.escalate(ticket, reason, notes)
        ticket.status = TicketStatus.ESCALATED
        ticket.current_agent = AgentTier.ESCALATION
        ticket.add_message(
            role="assistant",
            content=agent_response.content,
            agent_tier=AgentTier.ESCALATION,
            metadata=agent_response.metadata,
        )
        self._save(ticket)
        return agent_response.content, ticket

    def _user_wants_human(self, message: str) -> bool:
        keywords = {
            "speak to a human", "talk to a person", "human agent",
            "real person", "live agent", "speak with someone",
            "talk to someone", "connect me to a person", "i want a human",
        }
        lower = message.lower()
        return any(kw in lower for kw in keywords)

    def _save(self, ticket: SupportTicket) -> None:
        self.ticket_store.save(ticket)
        session = self.session_memory.get_session(ticket.ticket_id)
        if session:
            session.ticket = ticket
            self.session_memory.update_session(session)

    def _load_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        session = self.session_memory.get_session(ticket_id)
        if session:
            return session.ticket
        return self.ticket_store.get(ticket_id)
