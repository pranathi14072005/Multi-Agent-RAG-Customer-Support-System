"""
settings.py
-----------
Central configuration for the Multi-Agent RAG Customer Support System.
All values are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Ollama / LLM ──────────────────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    fast_model: str = field(
        default_factory=lambda: os.getenv("FAST_MODEL", "llama3")
    )
    """Lighter model used for classification/routing tasks."""

    # ── Vector Store (ChromaDB) ────────────────────────────────────────
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    chroma_kb_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_KB_COLLECTION", "knowledge_base")
    )
    chroma_tickets_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_TICKETS_COLLECTION", "resolved_tickets")
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5"))

    # ── Memory ─────────────────────────────────────────────────────────
    max_memory_turns: int = int(os.getenv("MAX_MEMORY_TURNS", "20"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

    # ── Escalation Thresholds ──────────────────────────────────────────
    escalation_confidence_threshold: float = float(
        os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.4")
    )
    """If RAG answer confidence drops below this, escalate to Tier-2."""
    max_auto_resolution_attempts: int = int(
        os.getenv("MAX_AUTO_RESOLUTION_ATTEMPTS", "3")
    )
    """Max times Tier-1 tries to resolve before auto-escalating."""
    sentiment_escalation_threshold: float = float(
        os.getenv("SENTIMENT_ESCALATION_THRESHOLD", "-0.6")
    )
    """If user sentiment drops below this (range -1..1), escalate immediately."""

    # ── Ticket / KB Management ─────────────────────────────────────────
    ticket_db_path: str = field(
        default_factory=lambda: os.getenv("TICKET_DB_PATH", "./tickets.db")
    )
    kb_source_dir: str = field(
        default_factory=lambda: os.getenv("KB_SOURCE_DIR", "./kb_sources")
    )
    auto_kb_update: bool = os.getenv("AUTO_KB_UPDATE", "true").lower() == "true"
    """Automatically add resolved tickets to the KB after human review."""

    # ── Agent Personas ─────────────────────────────────────────────────
    company_name: str = field(
        default_factory=lambda: os.getenv("COMPANY_NAME", "Acme Corp")
    )
    support_email: str = field(
        default_factory=lambda: os.getenv("SUPPORT_EMAIL", "support@acme.com")
    )
