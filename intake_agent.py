"""
agents/intake_agent.py
----------------------
Intake (Triage) Agent — the first agent in the pipeline.

Responsibilities:
  - Classify the ticket into a category and priority
  - Detect sentiment
  - Decide if RAG is needed
  - Assign initial tags
  - Surface immediate escalation triggers (e.g. "data breach", "legal")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_ollama import ChatOllama

from models import (
    AgentResponse, AgentTier, EscalationReason, RoutingDecision,
    SupportCategory, SupportTicket, TicketPriority,
)
from agent_prompts import INTAKE_PROMPT
from settings import Settings

logger = logging.getLogger(__name__)

# Hard keywords that always trigger immediate escalation regardless of LLM output
CRITICAL_ESCALATION_KEYWORDS = {
    "data breach", "security incident", "lawsuit", "legal action",
    "attorney", "solicitor", "court", "regulatory", "gdpr violation",
    "ccpa violation", "ransomware", "hack", "compromised account",
}


class IntakeAgent:
    """
    Classifies an incoming support message and produces a RoutingDecision.

    Uses a fast LLM call with a strict JSON-output prompt.
    Falls back to rule-based classification if the LLM call fails.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.llm = ChatOllama(
            model=self.settings.fast_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.0,
        )
        logger.info("IntakeAgent initialised (model=%s).", self.settings.fast_model)

    def classify(self, ticket: SupportTicket) -> RoutingDecision:
        """
        Classify the ticket's most recent user message.
        Returns a RoutingDecision with category, priority, and target agent.
        """
        query = self._get_latest_user_message(ticket)
        if not query:
            return self._fallback_decision(ticket, "No user message found")

        # Hard-keyword check first (fastest path)
        lower = query.lower()
        for kw in CRITICAL_ESCALATION_KEYWORDS:
            if kw in lower:
                logger.warning("Critical keyword '%s' detected in ticket %s.", kw, ticket.ticket_id)
                return RoutingDecision(
                    target_agent=AgentTier.ESCALATION,
                    category=SupportCategory.COMPLAINT,
                    priority=TicketPriority.CRITICAL,
                    confidence=0.99,
                    reason=f"Critical keyword detected: '{kw}'",
                    requires_rag=False,
                    metadata={"triggered_by": "keyword", "keyword": kw},
                )

        # LLM classification
        try:
            chain = INTAKE_PROMPT | self.llm
            raw = chain.invoke({
                "company_name": self.settings.company_name,
                "query": query,
            })
            result = self._parse_llm_output(raw.content)
            return self._build_routing_decision(result, ticket)
        except Exception as exc:
            logger.warning("IntakeAgent LLM call failed: %s. Using fallback.", exc)
            return self._fallback_decision(ticket, str(exc))

    def build_agent_response(self, ticket: SupportTicket,
                             decision: RoutingDecision) -> AgentResponse:
        """Package the routing decision as an AgentResponse."""
        return AgentResponse(
            content=(
                f"Thank you for contacting {self.settings.company_name} support. "
                f"I've received your request and I'm routing it to the right specialist. "
                f"Your ticket ID is **{ticket.ticket_id}**. "
                f"We'll have this resolved as quickly as possible."
            ),
            agent_tier=AgentTier.INTAKE,
            ticket_id=ticket.ticket_id,
            confidence=decision.confidence,
            should_escalate=(decision.target_agent == AgentTier.ESCALATION),
            escalation_reason=(
                EscalationReason.CRITICAL_ISSUE
                if decision.target_agent == AgentTier.ESCALATION else None
            ),
            suggested_category=decision.category,
            suggested_priority=decision.priority,
            metadata=decision.metadata,
        )

    # ── Private Helpers ────────────────────────────────────────────────

    def _get_latest_user_message(self, ticket: SupportTicket) -> str:
        for msg in reversed(ticket.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def _parse_llm_output(self, raw: str) -> dict:
        """Extract JSON from the LLM response, handling common formatting issues."""
        raw = raw.strip()
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        # Find first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM output: {raw[:200]}")
        return json.loads(match.group())

    def _build_routing_decision(self, result: dict, ticket: SupportTicket) -> RoutingDecision:
        raw_cat = result.get("category", "unknown").lower()
        raw_pri = result.get("priority", "medium").lower()
        sentiment = float(result.get("sentiment_score", 0.0))
        requires_rag = bool(result.get("requires_rag", True))
        tags = result.get("tags", [])
        subject = result.get("subject", ticket.subject or "Support request")
        reason = result.get("reason", "")

        try:
            category = SupportCategory(raw_cat)
        except ValueError:
            category = SupportCategory.GENERAL

        try:
            priority = TicketPriority(raw_pri)
        except ValueError:
            priority = TicketPriority.MEDIUM

        # Override priority to CRITICAL on critical category keywords
        if priority == TicketPriority.CRITICAL:
            target = AgentTier.ESCALATION
        elif category == SupportCategory.BILLING and priority in (TicketPriority.HIGH, TicketPriority.CRITICAL):
            target = AgentTier.TIER2
        elif category == SupportCategory.TECHNICAL:
            target = AgentTier.TIER1  # Tier-1 handles technical first
        else:
            target = AgentTier.TIER1

        # Sentiment-based escalation override
        if sentiment <= self.settings.sentiment_escalation_threshold:
            logger.info("Negative sentiment (%.2f) triggers escalation for ticket %s.",
                        sentiment, ticket.ticket_id)
            target = AgentTier.TIER2
            priority = TicketPriority.HIGH

        # Update ticket with discovered info
        ticket.subject = ticket.subject or subject
        ticket.tags = list(set(ticket.tags + tags))

        return RoutingDecision(
            target_agent=target,
            category=category,
            priority=priority,
            confidence=0.85,
            reason=reason,
            requires_rag=requires_rag,
            metadata={
                "sentiment_score": sentiment,
                "raw_classification": result,
            },
        )

    def _fallback_decision(self, ticket: SupportTicket, reason: str) -> RoutingDecision:
        """Rule-based fallback when LLM classification fails."""
        query = self._get_latest_user_message(ticket).lower()

        if any(w in query for w in ["bill", "charge", "refund", "payment", "invoice"]):
            category, priority = SupportCategory.BILLING, TicketPriority.HIGH
        elif any(w in query for w in ["error", "bug", "crash", "not working", "broken"]):
            category, priority = SupportCategory.TECHNICAL, TicketPriority.MEDIUM
        elif any(w in query for w in ["account", "login", "password", "access"]):
            category, priority = SupportCategory.ACCOUNT, TicketPriority.MEDIUM
        else:
            category, priority = SupportCategory.GENERAL, TicketPriority.LOW

        return RoutingDecision(
            target_agent=AgentTier.TIER1,
            category=category,
            priority=priority,
            confidence=0.5,
            reason=f"Fallback classification (LLM error: {reason})",
            requires_rag=True,
            metadata={"fallback": True},
        )
