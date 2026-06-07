"""
agents/escalation_agent.py
--------------------------
Escalation Agent — manages human handoff.

Responsibilities:
  - Generate a detailed handoff summary for human agents
  - Format the escalation notification
  - Update ticket state for human agent queue
  - (Optionally) send alerts via email/Slack hooks
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from models import (
    AgentResponse, AgentTier, EscalationReason,
    SupportTicket, TicketStatus,
)
from agent_prompts import ESCALATION_PROMPT
from settings import Settings

logger = logging.getLogger(__name__)


class EscalationAgent:
    """
    Handles the final escalation step: preparing the handoff package
    for a human support agent.

    Output:
    - A structured handoff summary (written by the LLM)
    - Updated ticket status → PENDING_HUMAN
    - Console / log alert (extend with email/Slack as needed)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.llm = ChatOllama(
            model=self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.2,
        )
        logger.info("EscalationAgent initialised.")

    def escalate(
        self,
        ticket: SupportTicket,
        reason: EscalationReason,
        escalation_notes: str = "",
    ) -> AgentResponse:
        """
        Prepare and deliver the escalation handoff.

        Args:
            ticket:           The ticket being escalated
            reason:           Why this is being escalated
            escalation_notes: Extra context from the preceding agent

        Returns:
            AgentResponse with the handoff summary as content
        """
        ticket.status = TicketStatus.PENDING_HUMAN
        ticket.current_agent = AgentTier.ESCALATION
        ticket.escalation_reason = reason
        ticket.escalation_notes = escalation_notes
        ticket.updated_at = datetime.utcnow()

        # Build conversation summary for the LLM
        conversation_lines = []
        for msg in ticket.messages:
            tier = f" [{msg.agent_tier.value}]" if msg.agent_tier else ""
            conversation_lines.append(f"{msg.role.upper()}{tier}: {msg.content[:300]}")
        conversation_summary = "\n".join(conversation_lines[-20:])  # last 20 turns

        # Generate handoff summary
        try:
            chain = ESCALATION_PROMPT | self.llm | StrOutputParser()
            handoff_summary = chain.invoke({
                "company_name": self.settings.company_name,
                "ticket_id": ticket.ticket_id,
                "category": ticket.category.value,
                "priority": ticket.priority.value,
                "escalation_reason": reason.value,
                "conversation_summary": conversation_summary,
            })
        except Exception as exc:
            logger.error("EscalationAgent LLM call failed: %s", exc)
            handoff_summary = self._fallback_summary(ticket, reason, escalation_notes)

        # Customer-facing message
        customer_message = self._customer_message(ticket, reason)

        # Log the escalation (extend with real alerts here)
        self._alert(ticket, reason, handoff_summary)

        return AgentResponse(
            content=customer_message,
            agent_tier=AgentTier.ESCALATION,
            ticket_id=ticket.ticket_id,
            confidence=1.0,
            should_escalate=False,   # Already escalated, no further escalation needed
            metadata={
                "handoff_summary": handoff_summary,
                "escalation_reason": reason.value,
                "escalated_at": datetime.utcnow().isoformat(),
            },
        )

    def _customer_message(self, ticket: SupportTicket, reason: EscalationReason) -> str:
        reason_phrases = {
            EscalationReason.LOW_CONFIDENCE:
                "Your question requires personalised attention that our automated system can't provide.",
            EscalationReason.NEGATIVE_SENTIMENT:
                "We can see this situation has been frustrating, and we want to make it right.",
            EscalationReason.REPEATED_FAILURE:
                "Our automated system wasn't able to fully resolve your issue.",
            EscalationReason.CRITICAL_ISSUE:
                "Your issue has been flagged as high priority and requires immediate attention.",
            EscalationReason.USER_REQUESTED:
                "As requested, we're connecting you with a human agent.",
            EscalationReason.COMPLEX_BILLING:
                "Your billing matter requires manual review by our accounts team.",
            EscalationReason.POLICY_VIOLATION:
                "Your case involves a policy matter that requires human review.",
        }
        phrase = reason_phrases.get(reason, "Your issue requires specialist attention.")

        return (
            f"We're escalating your ticket ({ticket.ticket_id}) to our human support team. "
            f"{phrase} A specialist will contact you at **{ticket.customer_email or 'the email on your account'}** "
            f"within 1 business day. Your case reference is **{ticket.ticket_id}**.\n\n"
            f"If you have additional information to share, please reply to this conversation "
            f"and it will be included in your case file.\n\n"
            f"Thank you for your patience — we appreciate you as a {self.settings.company_name} customer."
        )

    def _fallback_summary(self, ticket: SupportTicket, reason: EscalationReason,
                          notes: str) -> str:
        return (
            f"ESCALATION HANDOFF — {ticket.ticket_id}\n"
            f"Category: {ticket.category.value} | Priority: {ticket.priority.value}\n"
            f"Reason: {reason.value}\n"
            f"Notes: {notes}\n"
            f"Customer: {ticket.customer_email}\n"
            f"Subject: {ticket.subject}\n"
            f"Attempts: {ticket.resolution_attempts}\n"
        )

    def _alert(self, ticket: SupportTicket, reason: EscalationReason,
               handoff_summary: str) -> None:
        """
        Log the escalation. Extend this to send emails, Slack messages,
        PagerDuty alerts, or create tickets in Jira/Zendesk.
        """
        logger.warning(
            "🚨 ESCALATION: Ticket %s | Category: %s | Priority: %s | Reason: %s",
            ticket.ticket_id, ticket.category.value,
            ticket.priority.value, reason.value,
        )
        # TODO: integrate with alerting system
        # e.g. slack_client.send(channel="#support-escalations", text=handoff_summary)
        # e.g. zendesk_client.create_ticket(ticket, handoff_summary)
