"""
tests/test_system.py
--------------------
Unit and integration tests for the Multi-Agent RAG Customer Support System.

Run with:
    pytest tests/ -v
    pytest tests/ -v -k "not integration"   # skip integration tests (require Ollama)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    AgentResponse, AgentTier, EscalationReason,
    KBDocument, Message, RoutingDecision,
    SupportCategory, SupportTicket,
    TicketPriority, TicketStatus,
)
from ticket_store import TicketStore
from session_memory import SessionMemory
from settings import Settings


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_ticket(
    category: SupportCategory = SupportCategory.BILLING,
    priority: TicketPriority = TicketPriority.MEDIUM,
    status: TicketStatus = TicketStatus.OPEN,
) -> SupportTicket:
    t = SupportTicket(
        customer_email="test@example.com",
        customer_id="user-123",
        subject="Test ticket",
        category=category,
        priority=priority,
        status=status,
    )
    t.add_message("user", "Hello, I need help with my billing.")
    return t


def make_settings(tmp_dir: str) -> Settings:
    s = Settings()
    s.ticket_db_path = str(Path(tmp_dir) / "tickets.db")
    s.chroma_persist_dir = str(Path(tmp_dir) / "chroma")
    s.session_ttl_seconds = 5   # Short TTL for tests
    return s


# ── Model Tests ────────────────────────────────────────────────────────────────

class TestSupportTicket(unittest.TestCase):

    def test_ticket_id_generated(self):
        t = SupportTicket()
        self.assertTrue(t.ticket_id.startswith("TKT-"))
        self.assertEqual(len(t.ticket_id), 12)

    def test_add_message(self):
        t = make_ticket()
        initial_count = len(t.messages)
        t.add_message("assistant", "Hello, I can help!")
        self.assertEqual(len(t.messages), initial_count + 1)
        self.assertEqual(t.messages[-1].role, "assistant")

    def test_add_message_updates_timestamp(self):
        t = make_ticket()
        old_ts = t.updated_at
        time.sleep(0.01)
        t.add_message("user", "Follow-up message")
        self.assertGreaterEqual(t.updated_at, old_ts)

    def test_to_lc_history(self):
        from langchain_core.messages import HumanMessage, AIMessage
        t = make_ticket()
        t.add_message("assistant", "I can help with that.")
        history = t.to_lc_history()
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], HumanMessage)
        self.assertIsInstance(history[1], AIMessage)

    def test_summary_dict_keys(self):
        t = make_ticket()
        s = t.summary_dict()
        for key in ("ticket_id", "category", "priority", "status",
                    "subject", "resolution_summary", "was_auto_resolved"):
            self.assertIn(key, s)


class TestKBDocument(unittest.TestCase):

    def test_doc_id_generated(self):
        doc = KBDocument(title="Test", content="Hello")
        self.assertTrue(doc.doc_id.startswith("KB-"))

    def test_to_langchain_document(self):
        doc = KBDocument(
            title="Billing Guide",
            content="Refunds take 3-5 days.",
            category=SupportCategory.BILLING,
            tags=["billing", "refund"],
        )
        lc = doc.to_langchain_document()
        self.assertEqual(lc.page_content, "Refunds take 3-5 days.")
        self.assertEqual(lc.metadata["category"], "billing")
        self.assertIn("billing", lc.metadata["tags"])


class TestMessage(unittest.TestCase):

    def test_user_message_to_lc(self):
        from langchain_core.messages import HumanMessage
        m = Message(role="user", content="Test")
        self.assertIsInstance(m.to_lc_message(), HumanMessage)

    def test_assistant_message_to_lc(self):
        from langchain_core.messages import AIMessage
        m = Message(role="assistant", content="Test")
        self.assertIsInstance(m.to_lc_message(), AIMessage)

    def test_system_message_to_lc(self):
        from langchain_core.messages import SystemMessage
        m = Message(role="system", content="Test")
        self.assertIsInstance(m.to_lc_message(), SystemMessage)


# ── TicketStore Tests ──────────────────────────────────────────────────────────

class TestTicketStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.settings = make_settings(self.tmp)
        self.store = TicketStore(self.settings)

    def test_save_and_get(self):
        t = make_ticket()
        self.store.save(t)
        loaded = self.store.get(t.ticket_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.ticket_id, t.ticket_id)
        self.assertEqual(loaded.customer_email, "test@example.com")

    def test_messages_persisted(self):
        t = make_ticket()
        t.add_message("assistant", "How can I help?", agent_tier=AgentTier.TIER1)
        self.store.save(t)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.messages[1].role, "assistant")
        self.assertEqual(loaded.messages[1].agent_tier, AgentTier.TIER1)

    def test_update_on_second_save(self):
        t = make_ticket()
        self.store.save(t)
        t.subject = "Updated subject"
        t.status = TicketStatus.IN_PROGRESS
        self.store.save(t)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(loaded.subject, "Updated subject")
        self.assertEqual(loaded.status, TicketStatus.IN_PROGRESS)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.store.get("TKT-DOESNOTEXIST"))

    def test_update_status(self):
        t = make_ticket()
        self.store.save(t)
        self.store.update_status(t.ticket_id, TicketStatus.RESOLVED)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(loaded.status, TicketStatus.RESOLVED)
        self.assertIsNotNone(loaded.resolved_at)

    def test_update_feedback(self):
        t = make_ticket()
        self.store.save(t)
        self.store.update_feedback(t.ticket_id, 4)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(loaded.feedback_score, 4)

    def test_feedback_clamped(self):
        t = make_ticket()
        self.store.save(t)
        self.store.update_feedback(t.ticket_id, 99)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(loaded.feedback_score, 5)

    def test_list_tickets_by_status(self):
        t1 = make_ticket(status=TicketStatus.OPEN)
        t2 = make_ticket(status=TicketStatus.RESOLVED)
        self.store.save(t1)
        self.store.save(t2)
        open_tickets = self.store.list_tickets(status=TicketStatus.OPEN)
        resolved_tickets = self.store.list_tickets(status=TicketStatus.RESOLVED)
        open_ids = [t.ticket_id for t in open_tickets]
        resolved_ids = [t.ticket_id for t in resolved_tickets]
        self.assertIn(t1.ticket_id, open_ids)
        self.assertIn(t2.ticket_id, resolved_ids)
        self.assertNotIn(t1.ticket_id, resolved_ids)

    def test_list_tickets_by_category(self):
        t1 = make_ticket(category=SupportCategory.BILLING)
        t2 = make_ticket(category=SupportCategory.TECHNICAL)
        self.store.save(t1)
        self.store.save(t2)
        billing = self.store.list_tickets(category=SupportCategory.BILLING)
        billing_ids = [t.ticket_id for t in billing]
        self.assertIn(t1.ticket_id, billing_ids)
        self.assertNotIn(t2.ticket_id, billing_ids)

    def test_analytics_basic(self):
        for _ in range(3):
            t = make_ticket(status=TicketStatus.OPEN)
            self.store.save(t)
        t_resolved = make_ticket(status=TicketStatus.RESOLVED)
        t_resolved.was_auto_resolved = True
        self.store.save(t_resolved)

        analytics = self.store.get_analytics(days=30)
        self.assertGreaterEqual(analytics["total_tickets"], 4)
        self.assertIn("resolution_rate", analytics)
        self.assertIn("by_category", analytics)

    def test_kb_fields_roundtrip(self):
        t = make_ticket()
        t.kb_sources_used = ["doc1.pdf", "doc2.txt"]
        t.tags = ["billing", "refund"]
        t.confidence_score = 0.87
        self.store.save(t)
        loaded = self.store.get(t.ticket_id)
        self.assertEqual(loaded.kb_sources_used, ["doc1.pdf", "doc2.txt"])
        self.assertEqual(loaded.tags, ["billing", "refund"])
        self.assertAlmostEqual(loaded.confidence_score, 0.87, places=2)


# ── SessionMemory Tests ────────────────────────────────────────────────────────

class TestSessionMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.settings = make_settings(self.tmp)
        self.memory = SessionMemory(self.settings)

    def test_create_and_get_session(self):
        t = make_ticket()
        session = self.memory.create_session(t)
        self.assertIsNotNone(session)
        retrieved = self.memory.get_session(t.ticket_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.ticket.ticket_id, t.ticket_id)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.memory.get_session("TKT-FAKE"))

    def test_session_ttl_expiry(self):
        self.memory.settings.session_ttl_seconds = 0   # Expire immediately
        t = make_ticket()
        self.memory.create_session(t)
        time.sleep(0.01)
        result = self.memory.get_session(t.ticket_id)
        self.assertIsNone(result)

    def test_close_session_returns_ticket(self):
        t = make_ticket()
        self.memory.create_session(t)
        returned = self.memory.close_session(t.ticket_id)
        self.assertEqual(returned.ticket_id, t.ticket_id)
        self.assertIsNone(self.memory.get_session(t.ticket_id))

    def test_active_count(self):
        self.assertEqual(self.memory.active_count(), 0)
        t1 = make_ticket()
        t2 = make_ticket()
        self.memory.create_session(t1)
        self.memory.create_session(t2)
        self.assertEqual(self.memory.active_count(), 2)

    def test_evict_expired(self):
        self.memory.settings.session_ttl_seconds = 0
        for _ in range(3):
            self.memory.create_session(make_ticket())
        time.sleep(0.01)
        evicted = self.memory.evict_expired()
        self.assertEqual(evicted, 3)
        self.assertEqual(self.memory.active_count(), 0)

    def test_get_lc_history(self):
        t = make_ticket()
        t.add_message("assistant", "Hello!", agent_tier=AgentTier.TIER1)
        self.memory.create_session(t)
        history = self.memory.get_lc_history(t.ticket_id)
        self.assertEqual(len(history), 2)

    def test_get_lc_history_missing_session(self):
        history = self.memory.get_lc_history("TKT-NOTFOUND")
        self.assertEqual(history, [])


# ── IntakeAgent Tests (mocked LLM) ────────────────────────────────────────────

class TestIntakeAgent(unittest.TestCase):

    def _make_agent(self):
        from intake_agent import IntakeAgent
        with patch("intake_agent.ChatOllama"):
            agent = IntakeAgent()
        return agent

    def test_critical_keyword_triggers_escalation(self):
        from intake_agent import IntakeAgent
        from models import AgentTier
        with patch("intake_agent.ChatOllama"):
            agent = IntakeAgent()
        t = make_ticket()
        t.messages[0] = Message(role="user", content="There's been a data breach on my account!")
        decision = agent.classify(t)
        self.assertEqual(decision.target_agent, AgentTier.ESCALATION)
        self.assertEqual(decision.priority, TicketPriority.CRITICAL)

    def test_fallback_billing_classification(self):
        from intake_agent import IntakeAgent
        with patch("intake_agent.ChatOllama") as mock_llama:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            mock_llama.return_value = mock_llm
            agent = IntakeAgent()

        t = make_ticket()
        t.messages[0] = Message(role="user", content="I was charged the wrong amount on my bill.")
        decision = agent.classify(t)
        self.assertEqual(decision.category, SupportCategory.BILLING)
        self.assertTrue(decision.metadata.get("fallback"))

    def test_fallback_technical_classification(self):
        from intake_agent import IntakeAgent
        with patch("intake_agent.ChatOllama") as mock_llama:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("unavailable")
            mock_llama.return_value = mock_llm
            agent = IntakeAgent()

        t = make_ticket()
        t.messages[0] = Message(role="user", content="The app keeps crashing with an error.")
        decision = agent.classify(t)
        self.assertEqual(decision.category, SupportCategory.TECHNICAL)

    def test_json_parsing_strips_markdown(self):
        from intake_agent import IntakeAgent
        with patch("intake_agent.ChatOllama"):
            agent = IntakeAgent()
        raw = '```json\n{"category": "billing", "priority": "high", "requires_rag": true, "sentiment_score": -0.2, "tags": [], "subject": "Test", "reason": "billing issue"}\n```'
        result = agent._parse_llm_output(raw)
        self.assertEqual(result["category"], "billing")

    def test_user_wants_human_detection(self):
        from supervisor import Supervisor
        with patch("supervisor.KnowledgeBase"), \
             patch("supervisor.TicketStore"), \
             patch("supervisor.SessionMemory"), \
             patch("supervisor.IntakeAgent"), \
             patch("supervisor.Tier1Agent"), \
             patch("supervisor.Tier2Agent"), \
             patch("supervisor.EscalationAgent"), \
             patch("supervisor.KBManagerAgent"):
            sv = Supervisor()
        self.assertTrue(sv._user_wants_human("I want to speak to a human agent"))
        self.assertTrue(sv._user_wants_human("Can I talk to a real person?"))
        self.assertFalse(sv._user_wants_human("How do I reset my password?"))


# ── AgentResponse Tests ────────────────────────────────────────────────────────

class TestAgentResponse(unittest.TestCase):

    def test_default_values(self):
        r = AgentResponse(
            content="Test", agent_tier=AgentTier.TIER1, ticket_id="TKT-001"
        )
        self.assertEqual(r.confidence, 1.0)
        self.assertFalse(r.should_escalate)
        self.assertIsNone(r.escalation_reason)
        self.assertEqual(r.kb_sources, [])

    def test_escalation_fields(self):
        r = AgentResponse(
            content="Escalating",
            agent_tier=AgentTier.TIER1,
            ticket_id="TKT-001",
            confidence=0.2,
            should_escalate=True,
            escalation_reason=EscalationReason.LOW_CONFIDENCE,
            escalation_notes="Confidence too low.",
        )
        self.assertTrue(r.should_escalate)
        self.assertEqual(r.escalation_reason, EscalationReason.LOW_CONFIDENCE)


# ── Settings Tests ─────────────────────────────────────────────────────────────

class TestSettings(unittest.TestCase):

    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.llm_model, "llama3")
        self.assertEqual(s.embedding_model, "nomic-embed-text")
        self.assertEqual(s.chunk_size, 512)
        # retriever_top_k default is 5 (from settings.py)
        self.assertEqual(s.retriever_top_k, 5)

    def test_env_override(self, monkeypatch=None):
        """Settings reads env vars via field default_factory at instantiation time."""
        import os, importlib
        os.environ["LLM_MODEL"] = "mistral"
        # chunk_size uses int(os.getenv(...)) at class-body eval time,
        # so we test the string-based fields that use default_factory
        s = Settings()
        self.assertEqual(s.llm_model, "mistral")
        # Clean up
        del os.environ["LLM_MODEL"]


# ── Routing Decision Tests ─────────────────────────────────────────────────────

class TestRoutingDecision(unittest.TestCase):

    def test_fields(self):
        rd = RoutingDecision(
            target_agent=AgentTier.TIER1,
            category=SupportCategory.BILLING,
            priority=TicketPriority.HIGH,
            confidence=0.9,
            reason="Billing keyword match",
        )
        self.assertEqual(rd.target_agent, AgentTier.TIER1)
        self.assertTrue(rd.requires_rag)


# ── Run all tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
