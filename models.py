"""
models.py
---------
Shared data models (dataclasses + enums) used across all agents and services.
No external dependencies — pure Python stdlib.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ── Enumerations ───────────────────────────────────────────────────────────────

class TicketStatus(str, Enum):
    OPEN       = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED  = "escalated"
    PENDING_HUMAN = "pending_human"
    RESOLVED   = "resolved"
    CLOSED     = "closed"


class TicketPriority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class AgentTier(str, Enum):
    INTAKE    = "intake"      # Classifier / triage agent
    TIER1     = "tier1"       # Automated RAG-based resolver
    TIER2     = "tier2"       # Specialised domain agent
    ESCALATION = "escalation" # Human handoff coordinator
    KB_MANAGER = "kb_manager" # Knowledge-base maintenance agent
    SUPERVISOR = "supervisor" # Orchestration + routing


class SupportCategory(str, Enum):
    BILLING      = "billing"
    TECHNICAL    = "technical"
    ACCOUNT      = "account"
    PRODUCT      = "product"
    COMPLAINT    = "complaint"
    REFUND       = "refund"
    GENERAL      = "general"
    UNKNOWN      = "unknown"


class EscalationReason(str, Enum):
    LOW_CONFIDENCE   = "low_rag_confidence"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    REPEATED_FAILURE = "repeated_resolution_failure"
    CRITICAL_ISSUE   = "critical_issue_detected"
    USER_REQUESTED   = "user_requested_human"
    POLICY_VIOLATION = "policy_violation"
    COMPLEX_BILLING  = "complex_billing"


# ── Core Data Models ───────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single turn in a support conversation."""
    role: str                  # "user" | "assistant" | "system" | "agent"
    content: str
    agent_tier: Optional[AgentTier] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_lc_message(self):
        """Convert to a LangChain BaseMessage."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role in ("assistant", "agent"):
            return AIMessage(content=self.content)
        else:
            return SystemMessage(content=self.content)


@dataclass
class SupportTicket:
    """Represents a customer support ticket through its full lifecycle."""
    ticket_id: str = field(default_factory=lambda: f"TKT-{uuid.uuid4().hex[:8].upper()}")
    customer_id: str = ""
    customer_email: str = ""
    subject: str = ""
    category: SupportCategory = SupportCategory.UNKNOWN
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    # Conversation history
    messages: list[Message] = field(default_factory=list)

    # Routing & escalation
    current_agent: AgentTier = AgentTier.INTAKE
    assigned_human: Optional[str] = None
    escalation_reason: Optional[EscalationReason] = None
    escalation_notes: str = ""
    resolution_attempts: int = 0

    # RAG metadata
    kb_sources_used: list[str] = field(default_factory=list)
    confidence_score: float = 1.0

    # Resolution
    resolution_summary: str = ""
    was_auto_resolved: bool = False
    feedback_score: Optional[int] = None   # 1-5 CSAT

    # Extra
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str,
                    agent_tier: Optional[AgentTier] = None,
                    metadata: Optional[dict] = None) -> None:
        self.messages.append(Message(
            role=role,
            content=content,
            agent_tier=agent_tier,
            metadata=metadata or {},
        ))
        self.updated_at = datetime.utcnow()

    def to_lc_history(self) -> list:
        """Return conversation as LangChain message list."""
        return [m.to_lc_message() for m in self.messages]

    def summary_dict(self) -> dict:
        """Compact dict for logging / KB indexing."""
        return {
            "ticket_id": self.ticket_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "subject": self.subject,
            "resolution_summary": self.resolution_summary,
            "was_auto_resolved": self.was_auto_resolved,
            "feedback_score": self.feedback_score,
            "tags": self.tags,
        }


@dataclass
class AgentResponse:
    """Structured response returned by any agent."""
    content: str
    agent_tier: AgentTier
    ticket_id: str
    confidence: float = 1.0
    should_escalate: bool = False
    escalation_reason: Optional[EscalationReason] = None
    escalation_notes: str = ""
    kb_sources: list[str] = field(default_factory=list)
    suggested_category: Optional[SupportCategory] = None
    suggested_priority: Optional[TicketPriority] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KBDocument:
    """A document stored in the knowledge base."""
    doc_id: str = field(default_factory=lambda: f"KB-{uuid.uuid4().hex[:10].upper()}")
    title: str = ""
    content: str = ""
    category: SupportCategory = SupportCategory.GENERAL
    source: str = ""                    # file path, URL, ticket_id, etc.
    source_type: str = "manual"         # "manual" | "ticket" | "url" | "file"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    tags: list[str] = field(default_factory=list)
    is_active: bool = True

    def to_langchain_document(self):
        from langchain_core.documents import Document
        return Document(
            page_content=self.content,
            metadata={
                "doc_id": self.doc_id,
                "title": self.title,
                "category": self.category.value,
                "source": self.source,
                "source_type": self.source_type,
                "version": self.version,
                "tags": ",".join(self.tags),
            },
        )


@dataclass
class RoutingDecision:
    """Output from the router / supervisor agent."""
    target_agent: AgentTier
    category: SupportCategory
    priority: TicketPriority
    confidence: float
    reason: str
    requires_rag: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
