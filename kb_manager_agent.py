"""
agents/kb_manager_agent.py
--------------------------
Knowledge Base Manager Agent.

Responsibilities:
  - Analyse resolved tickets for new KB-worthy content
  - Draft, polish, and ingest new KB articles
  - Detect KB gaps (common questions with no good KB answer)
  - Manage KB document versions and staleness
  - Run on a schedule or triggered post-resolution
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from knowledge_base import KnowledgeBase
from models import KBDocument, SupportCategory, SupportTicket
from agent_prompts import KB_GAP_PROMPT, KB_UPDATE_PROMPT
from settings import Settings

logger = logging.getLogger(__name__)


class KBManagerAgent:
    """
    Autonomous KB maintenance agent.

    Post-resolution flow:
      1. `analyse_ticket(ticket)` → decide if KB needs updating
      2. If yes: draft a new article from the resolution
      3. `polish_article(draft)` → LLM-improved, structured article
      4. `ingest_article(article, ticket)` → add to ChromaDB KB
      5. `index_resolved_ticket(ticket)` → add to similar-case index

    Run `run_kb_update_cycle(tickets)` to process a batch.
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
            temperature=0.3,
        )
        self.fast_llm = ChatOllama(
            model=self.settings.fast_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.0,
        )
        logger.info("KBManagerAgent initialised.")

    # ── Main Entry Points ──────────────────────────────────────────────

    def analyse_ticket(self, ticket: SupportTicket) -> dict:
        """
        Analyse a resolved ticket and decide whether its resolution
        should be added to the knowledge base.

        Returns a dict with keys:
          should_add_to_kb, reason, article_title,
          article_content, category, tags
        """
        if not ticket.resolution_summary and not ticket.messages:
            return {"should_add_to_kb": False, "reason": "Empty ticket"}

        # Build full conversation text
        conversation = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in ticket.messages
        )

        try:
            chain = KB_GAP_PROMPT | self.fast_llm
            raw = chain.invoke({
                "subject": ticket.subject,
                "category": ticket.category.value,
                "resolution_summary": ticket.resolution_summary or "(no summary)",
                "conversation": conversation[:3000],
            })
            raw_text = raw.content.strip()
            raw_text = re.sub(r"```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"```\s*$", "", raw_text)
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("KB gap analysis failed: %s", exc)

        return {"should_add_to_kb": False, "reason": f"Analysis failed"}

    def polish_article(self, draft: str) -> str:
        """Run a draft KB article through the LLM polisher."""
        try:
            chain = KB_UPDATE_PROMPT | self.llm | StrOutputParser()
            return chain.invoke({
                "company_name": self.settings.company_name,
                "draft": draft,
            })
        except Exception as exc:
            logger.warning("Article polishing failed: %s", exc)
            return draft   # Return unpolished draft on failure

    def ingest_article(
        self,
        title: str,
        content: str,
        category: SupportCategory,
        tags: list[str],
        source: str = "auto-generated",
    ) -> KBDocument:
        """Polish and ingest a new article into the KB."""
        polished = self.polish_article(content)
        doc = KBDocument(
            title=title,
            content=polished,
            category=category,
            source=source,
            source_type="ticket",
            tags=tags,
        )
        self.kb.ingest_kb_document(doc)
        logger.info("Ingested new KB article: '%s' (category=%s).", title, category.value)
        return doc

    def process_resolved_ticket(self, ticket: SupportTicket) -> dict:
        """
        Full post-resolution KB update cycle for a single ticket.

        Returns a summary of what was done.
        """
        result = {
            "ticket_id": ticket.ticket_id,
            "kb_article_added": False,
            "ticket_indexed": False,
            "article_title": None,
        }

        # Always index resolved tickets (for Tier-2 similar-case lookup)
        self.kb.index_resolved_ticket(ticket)
        result["ticket_indexed"] = True

        # Optionally add a new KB article
        if self.settings.auto_kb_update:
            analysis = self.analyse_ticket(ticket)
            if analysis.get("should_add_to_kb") and analysis.get("article_content"):
                title = analysis.get("article_title", f"Article from {ticket.ticket_id}")
                content = analysis["article_content"]
                try:
                    raw_cat = analysis.get("category", ticket.category.value)
                    category = SupportCategory(raw_cat)
                except ValueError:
                    category = ticket.category

                tags = analysis.get("tags", []) + ticket.tags
                self.ingest_article(
                    title=title,
                    content=content,
                    category=category,
                    tags=list(set(tags)),
                    source=f"ticket:{ticket.ticket_id}",
                )
                result["kb_article_added"] = True
                result["article_title"] = title
                logger.info(
                    "KB updated from ticket %s: '%s'.", ticket.ticket_id, title
                )
            else:
                logger.debug(
                    "Ticket %s: no KB update needed (%s).",
                    ticket.ticket_id, analysis.get("reason"),
                )

        return result

    def run_kb_update_cycle(self, tickets: list[SupportTicket]) -> list[dict]:
        """
        Process a batch of resolved tickets for KB updates.
        Returns a list of per-ticket result dicts.
        """
        results = []
        for ticket in tickets:
            try:
                result = self.process_resolved_ticket(ticket)
                results.append(result)
            except Exception as exc:
                logger.error("KB update failed for ticket %s: %s", ticket.ticket_id, exc)
                results.append({"ticket_id": ticket.ticket_id, "error": str(exc)})
        return results

    def kb_stats(self) -> dict:
        """Return current KB statistics."""
        return {
            "kb_articles": self.kb.kb_document_count(),
            "indexed_tickets": self.kb.resolved_ticket_count(),
            "kb_has_documents": self.kb.has_kb_documents(),
        }
