"""
agents/tier2_agent.py
---------------------
Tier-2 Domain Specialist Agent.

Responsibilities:
  - Handle escalated tickets that Tier-1 couldn't resolve
  - Domain-specific reasoning (billing / technical / account)
  - Access to similar resolved tickets for precedent
  - Deeper RAG with broader context window
  - Decides whether to involve a human (Tier-3 escalation)
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from ..kb import KnowledgeBase
from ..models import (
    AgentResponse, AgentTier, EscalationReason,
    SupportCategory, SupportTicket,
)
from ..prompts import (
    TIER2_PROMPT,
    TIER2_BILLING_SYSTEM, TIER2_TECHNICAL_SYSTEM, TIER2_ACCOUNT_SYSTEM,
)
from ..settings import Settings

logger = logging.getLogger(__name__)

# Which categories each Tier-2 specialisation handles
TIER2_SYSTEM_MAP = {
    SupportCategory.BILLING:   TIER2_BILLING_SYSTEM,
    SupportCategory.REFUND:    TIER2_BILLING_SYSTEM,
    SupportCategory.TECHNICAL: TIER2_TECHNICAL_SYSTEM,
    SupportCategory.PRODUCT:   TIER2_TECHNICAL_SYSTEM,
    SupportCategory.ACCOUNT:   TIER2_ACCOUNT_SYSTEM,
    SupportCategory.COMPLAINT: TIER2_ACCOUNT_SYSTEM,
}


class Tier2Agent:
    """
    Domain-specialist agent for escalated support cases.

    Combines a domain-specific system prompt with:
      - Broader KB retrieval (top_k * 2)
      - Similar resolved cases from the ticket index
      - Full conversation history
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        settings: Optional[Settings] = None,
    ):
        self.kb = kb
        self.settings = settings or Settings()
        self.llm = ChatOllama(
            model=self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.temperature,
        )
        logger.info("Tier2Agent initialised (model=%s).", self.settings.llm_model)

    def respond(
        self,
        ticket: SupportTicket,
        query: str,
        chat_history: Optional[list] = None,
    ) -> AgentResponse:
        """
        Generate a domain-specialist response.

        Args:
            ticket:       The escalated support ticket
            query:        The customer's latest message
            chat_history: LangChain message list

        Returns:
            AgentResponse — may still escalate to human if unresolvable
        """
        history = chat_history or []

        # Pick the right specialist system prompt
        system_prompt = TIER2_SYSTEM_MAP.get(ticket.category, TIER2_TECHNICAL_SYSTEM)
        system_prompt = system_prompt.replace("{company_name}", self.settings.company_name)

        # Broader KB retrieval for Tier-2
        top_k = (self.settings.retriever_top_k or 4) * 2
        docs = self.kb.search(query, top_k=top_k, category_filter=ticket.category)
        if not docs:
            docs = self.kb.search(query, top_k=top_k)
        context = self.kb.format_context(docs)
        kb_sources = [d.metadata.get("source", "") for d in docs]

        # Similar resolved cases
        similar_docs = self.kb.search_similar_cases(
            query, top_k=3, category_filter=ticket.category
        )
        similar_cases = self.kb.format_similar_cases(similar_docs)

        try:
            chain = TIER2_PROMPT | self.llm | StrOutputParser()
            answer = chain.invoke({
                "system_prompt": system_prompt,
                "context": context,
                "similar_cases": similar_cases,
                "chat_history": history,
                "question": query,
            })
        except Exception as exc:
            logger.error("Tier2 chain failed for ticket %s: %s", ticket.ticket_id, exc)
            return self._human_handoff_response(
                ticket, reason=f"Specialist agent error: {exc}"
            )

        # Tier-2 escalation to human: if both KB and case history are empty
        needs_human = (
            not self.kb.has_kb_documents()
            and self.kb.resolved_ticket_count() == 0
        )

        if needs_human:
            return self._human_handoff_response(ticket, reason="No KB or case history available.")

        return AgentResponse(
            content=answer,
            agent_tier=AgentTier.TIER2,
            ticket_id=ticket.ticket_id,
            confidence=0.75,
            should_escalate=False,
            kb_sources=kb_sources,
            metadata={
                "specialist": ticket.category.value,
                "similar_cases_found": len(similar_docs),
            },
        )

    def _human_handoff_response(self, ticket: SupportTicket, reason: str) -> AgentResponse:
        return AgentResponse(
            content=(
                "I'm a Tier-2 specialist and I've reviewed your case carefully. "
                "This issue requires direct human intervention to resolve properly. "
                "I'm escalating this to our senior team now — you'll hear from a "
                "human agent very shortly. We sincerely apologise for the inconvenience."
            ),
            agent_tier=AgentTier.TIER2,
            ticket_id=ticket.ticket_id,
            confidence=0.0,
            should_escalate=True,
            escalation_reason=EscalationReason.REPEATED_FAILURE,
            escalation_notes=reason,
        )
