"""
agents/tier1_agent.py
---------------------
Tier-1 Automated Resolution Agent.

Responsibilities:
  - First-line automated support using RAG
  - Retrieves relevant KB articles and generates a grounded answer
  - Self-assesses answer confidence
  - Decides whether to escalate to Tier-2 or resolve
  - Tracks resolution attempts
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from ..kb import KnowledgeBase
from ..models import (
    AgentResponse, AgentTier, EscalationReason,
    SupportCategory, SupportTicket,
)
from ..prompts import TIER1_PROMPT, CONFIDENCE_PROMPT
from ..settings import Settings

logger = logging.getLogger(__name__)


class Tier1Agent:
    """
    RAG-powered first-line support agent.

    Resolution pipeline:
      1. Retrieve relevant KB chunks
      2. Generate answer using RAG prompt + conversation history
      3. Self-score confidence
      4. If confidence < threshold → flag for escalation
      5. If resolution_attempts >= max → force escalation
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
        self.fast_llm = ChatOllama(
            model=self.settings.fast_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.0,
        )
        logger.info("Tier1Agent initialised (model=%s).", self.settings.llm_model)

    def respond(self, ticket: SupportTicket, query: str,
                chat_history: Optional[list] = None) -> AgentResponse:
        """
        Generate a RAG-based response to the customer's query.

        Args:
            ticket:       The support ticket being handled
            query:        The customer's latest message
            chat_history: LangChain-format message list

        Returns:
            AgentResponse with content, confidence, and escalation flag
        """
        ticket.resolution_attempts += 1
        history = chat_history or []

        # Force escalation if too many failed attempts
        if ticket.resolution_attempts > self.settings.max_auto_resolution_attempts:
            logger.info("Ticket %s exceeded max resolution attempts, escalating.", ticket.ticket_id)
            return AgentResponse(
                content=(
                    "I've been unable to fully resolve your issue through our automated system. "
                    "I'm connecting you with a specialist who can give this the attention it deserves."
                ),
                agent_tier=AgentTier.TIER1,
                ticket_id=ticket.ticket_id,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=EscalationReason.REPEATED_FAILURE,
                escalation_notes=(
                    f"Tier-1 failed to resolve after {ticket.resolution_attempts} attempts. "
                    f"Category: {ticket.category.value}"
                ),
            )

        # Retrieve KB context
        docs = self.kb.search(query, category_filter=ticket.category)
        if not docs:
            # Try without category filter
            docs = self.kb.search(query)

        context = self.kb.format_context(docs)
        kb_sources = [d.metadata.get("source", "") for d in docs]

        # No KB content at all → escalate
        if not self.kb.has_kb_documents():
            logger.warning("KB is empty for ticket %s.", ticket.ticket_id)
            return AgentResponse(
                content=(
                    "I don't have sufficient information to answer your question right now. "
                    "Let me connect you with a specialist."
                ),
                agent_tier=AgentTier.TIER1,
                ticket_id=ticket.ticket_id,
                confidence=0.1,
                should_escalate=True,
                escalation_reason=EscalationReason.LOW_CONFIDENCE,
                escalation_notes="Knowledge base is empty.",
            )

        # Build RAG chain
        try:
            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: history,
                    "category": lambda _: ticket.category.value,
                    "priority": lambda _: ticket.priority.value,
                    "company_name": lambda _: self.settings.company_name,
                }
                | TIER1_PROMPT
                | self.llm
                | StrOutputParser()
            )
            answer = chain.invoke(query)
        except Exception as exc:
            logger.error("Tier1 RAG chain failed: %s", exc)
            return AgentResponse(
                content="I encountered an error processing your request. Please try again.",
                agent_tier=AgentTier.TIER1,
                ticket_id=ticket.ticket_id,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=EscalationReason.REPEATED_FAILURE,
                escalation_notes=f"RAG chain error: {exc}",
            )

        # Score confidence
        confidence = self._score_confidence(query, context, answer)
        should_escalate = confidence < self.settings.escalation_confidence_threshold

        escalation_reason = None
        escalation_notes = ""
        if should_escalate:
            escalation_reason = EscalationReason.LOW_CONFIDENCE
            escalation_notes = (
                f"Confidence score {confidence:.2f} below threshold "
                f"{self.settings.escalation_confidence_threshold}. "
                f"Category: {ticket.category.value}. Attempt #{ticket.resolution_attempts}."
            )
            logger.info("Low confidence (%.2f) for ticket %s, flagging escalation.",
                        confidence, ticket.ticket_id)

        return AgentResponse(
            content=answer,
            agent_tier=AgentTier.TIER1,
            ticket_id=ticket.ticket_id,
            confidence=confidence,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            escalation_notes=escalation_notes,
            kb_sources=kb_sources,
        )

    def _score_confidence(self, question: str, context: str, response: str) -> float:
        """
        Use a fast LLM call to estimate how well the response
        is grounded in the provided KB context.
        """
        try:
            chain = CONFIDENCE_PROMPT | self.fast_llm
            raw = chain.invoke({
                "question": question,
                "context": context[:2000],   # truncate to avoid token overflow
                "response": response,
            })
            raw_text = raw.content.strip()
            raw_text = re.sub(r"```(?:json)?\s*", "", raw_text)
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data.get("confidence", 0.5))
        except Exception as exc:
            logger.debug("Confidence scoring failed: %s", exc)
        return 0.6   # neutral fallback
