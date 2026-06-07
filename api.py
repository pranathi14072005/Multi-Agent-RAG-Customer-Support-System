"""
api.py
------
FastAPI REST interface for the Multi-Agent RAG Customer Support System.

Endpoints:
  POST   /tickets                   — Start a new support conversation
  POST   /tickets/{id}/messages     — Continue an existing conversation
  GET    /tickets/{id}              — Get full ticket details
  GET    /tickets                   — List tickets (with filters)
  POST   /tickets/{id}/resolve      — Resolve a ticket
  POST   /tickets/{id}/feedback     — Submit CSAT score
  POST   /tickets/{id}/close        — Close a resolved ticket
  POST   /kb/ingest/text            — Add a text article to the KB
  POST   /kb/ingest/file            — Upload a file to the KB
  POST   /kb/update-batch           — Run KB update cycle from resolved tickets
  GET    /analytics                 — Get system analytics
  GET    /health                    — Health check

Install FastAPI:
  pip install fastapi uvicorn python-multipart

Run:
  uvicorn multi_agent_rag_support.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from supervisor import Supervisor
from models import (
    SupportCategory, TicketStatus, TicketPriority, AgentTier,
)
from settings import Settings

logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent RAG Customer Support API",
    description="Enterprise support system with intelligent routing, RAG, and auto-escalation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton supervisor (initialised at startup)
_supervisor: Optional[Supervisor] = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor(Settings())
    return _supervisor


@app.on_event("startup")
async def startup():
    logger.info("Initialising Supervisor...")
    get_supervisor()
    logger.info("API ready.")


# ── Request / Response Models ──────────────────────────────────────────────────

class NewTicketRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000,
                         description="The customer's first message")
    customer_email: str = Field(default="", description="Customer email address")
    customer_id: str = Field(default="", description="Internal customer ID")


class ContinueTicketRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ResolveRequest(BaseModel):
    resolution_summary: str = Field(..., min_length=1)
    was_auto: bool = False


class FeedbackRequest(BaseModel):
    score: int = Field(..., ge=1, le=5, description="CSAT score from 1 (worst) to 5 (best)")


class IngestTextRequest(BaseModel):
    title: str
    content: str
    category: str = "general"
    tags: list[str] = []


class MessageResponse(BaseModel):
    ticket_id: str
    response: str
    agent_tier: str
    status: str
    priority: str
    category: str
    confidence: float
    is_escalated: bool


class TicketSummaryResponse(BaseModel):
    ticket_id: str
    subject: str
    status: str
    priority: str
    category: str
    customer_email: str
    created_at: str
    updated_at: str
    resolution_summary: str
    was_auto_resolved: bool
    confidence_score: float
    feedback_score: Optional[int]
    message_count: int
    kb_sources: list[str]
    tags: list[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ticket_to_summary(ticket) -> TicketSummaryResponse:
    return TicketSummaryResponse(
        ticket_id=ticket.ticket_id,
        subject=ticket.subject,
        status=ticket.status.value,
        priority=ticket.priority.value,
        category=ticket.category.value,
        customer_email=ticket.customer_email,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        resolution_summary=ticket.resolution_summary,
        was_auto_resolved=ticket.was_auto_resolved,
        confidence_score=ticket.confidence_score,
        feedback_score=ticket.feedback_score,
        message_count=len(ticket.messages),
        kb_sources=ticket.kb_sources_used,
        tags=ticket.tags,
    )


def _build_message_response(response_text: str, ticket) -> MessageResponse:
    return MessageResponse(
        ticket_id=ticket.ticket_id,
        response=response_text,
        agent_tier=ticket.current_agent.value,
        status=ticket.status.value,
        priority=ticket.priority.value,
        category=ticket.category.value,
        confidence=ticket.confidence_score,
        is_escalated=(ticket.status.value in ("escalated", "pending_human")),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Welcome root endpoint."""
    return {
        "message": "Welcome to the Multi-Agent RAG Customer Support API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Check system health and KB status."""
    sv = get_supervisor()
    kb_stats = sv.kb_manager.kb_stats()
    return {
        "status": "ok",
        "kb_articles": kb_stats["kb_articles"],
        "indexed_tickets": kb_stats["indexed_tickets"],
        "active_sessions": sv.session_memory.active_count(),
        "company": sv.settings.company_name,
        "model": sv.settings.llm_model,
    }


@app.post("/tickets", response_model=MessageResponse, tags=["Tickets"])
async def create_ticket(req: NewTicketRequest):
    """
    Start a new support conversation.
    Returns the first agent response and a ticket_id for follow-ups.
    """
    try:
        sv = get_supervisor()
        response_text, ticket = sv.handle(
            message=req.message,
            customer_email=req.customer_email,
            customer_id=req.customer_id,
        )
        return _build_message_response(response_text, ticket)
    except Exception as exc:
        logger.exception("Error creating ticket")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/tickets/{ticket_id}/messages", response_model=MessageResponse, tags=["Tickets"])
async def continue_ticket(ticket_id: str, req: ContinueTicketRequest):
    """Send a follow-up message on an existing ticket."""
    try:
        sv = get_supervisor()
        response_text, ticket = sv.handle(
            message=req.message,
            ticket_id=ticket_id,
        )
        return _build_message_response(response_text, ticket)
    except Exception as exc:
        logger.exception("Error continuing ticket %s", ticket_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tickets/{ticket_id}", response_model=TicketSummaryResponse, tags=["Tickets"])
async def get_ticket(ticket_id: str):
    """Retrieve full ticket details."""
    sv = get_supervisor()
    ticket = sv.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return _ticket_to_summary(ticket)


@app.get("/tickets/{ticket_id}/messages", tags=["Tickets"])
async def get_ticket_messages(ticket_id: str):
    """Retrieve the full conversation thread for a ticket."""
    sv = get_supervisor()
    ticket = sv.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return {
        "ticket_id": ticket_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "agent_tier": m.agent_tier.value if m.agent_tier else None,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in ticket.messages
        ],
    }


@app.get("/tickets", tags=["Tickets"])
async def list_tickets(
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(50, ge=1, le=500),
):
    """List tickets with optional filters."""
    sv = get_supervisor()
    try:
        status_filter = TicketStatus(status) if status else None
        category_filter = SupportCategory(category) if category else None
        priority_filter = TicketPriority(priority) if priority else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {exc}")

    tickets = sv.ticket_store.list_tickets(
        status=status_filter,
        category=category_filter,
        priority=priority_filter,
        limit=limit,
    )
    return {"tickets": [_ticket_to_summary(t) for t in tickets], "count": len(tickets)}


@app.post("/tickets/{ticket_id}/resolve", tags=["Tickets"])
async def resolve_ticket(ticket_id: str, req: ResolveRequest):
    """Mark a ticket as resolved and trigger KB update."""
    sv = get_supervisor()
    ticket = sv.resolve(ticket_id, req.resolution_summary, was_auto=req.was_auto)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return {"message": "Ticket resolved successfully", "ticket_id": ticket_id}


@app.post("/tickets/{ticket_id}/feedback", tags=["Tickets"])
async def submit_feedback(ticket_id: str, req: FeedbackRequest):
    """Submit CSAT feedback (1–5) for a resolved ticket."""
    sv = get_supervisor()
    ok = sv.record_feedback(ticket_id, req.score)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    return {"message": "Feedback recorded", "score": req.score}


@app.post("/tickets/{ticket_id}/close", tags=["Tickets"])
async def close_ticket(ticket_id: str):
    """Close a resolved ticket."""
    sv = get_supervisor()
    ok = sv.close_ticket(ticket_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to close ticket")
    return {"message": "Ticket closed", "ticket_id": ticket_id}


# ── KB Endpoints ───────────────────────────────────────────────────────────────

@app.post("/kb/ingest/text", tags=["Knowledge Base"])
async def ingest_text(req: IngestTextRequest):
    """Add a text article to the knowledge base."""
    try:
        cat = SupportCategory(req.category)
    except ValueError:
        cat = SupportCategory.GENERAL

    sv = get_supervisor()
    chunks = sv.ingest_kb_text(req.content, title=req.title, category=cat, tags=req.tags)
    return {"message": "Article ingested", "chunks_added": chunks, "title": req.title}


@app.post("/kb/ingest/file", tags=["Knowledge Base"])
async def ingest_file(
    file: UploadFile = File(...),
    category: str = Query("general"),
    tags: str = Query("", description="Comma-separated tags"),
):
    """Upload a file (PDF, txt, md) to the knowledge base."""
    import tempfile, shutil

    try:
        cat = SupportCategory(category)
    except ValueError:
        cat = SupportCategory.GENERAL

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or "upload.txt")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        sv = get_supervisor()
        chunks = sv.ingest_kb_file(tmp_path, category=cat, tags=tag_list)
        return {"message": "File ingested", "chunks_added": chunks, "filename": file.filename}
    finally:
        os.unlink(tmp_path)


@app.post("/kb/update-batch", tags=["Knowledge Base"])
async def run_kb_update(days: int = Query(30, ge=1, le=365)):
    """Run KB update cycle from recently resolved tickets."""
    sv = get_supervisor()
    results = sv.run_kb_update_batch(days=days)
    added = sum(1 for r in results if r.get("kb_article_added"))
    indexed = sum(1 for r in results if r.get("ticket_indexed"))
    return {
        "tickets_processed": len(results),
        "kb_articles_added": added,
        "tickets_indexed": indexed,
        "details": results,
    }


@app.get("/kb/stats", tags=["Knowledge Base"])
async def kb_stats():
    """Return knowledge base statistics."""
    sv = get_supervisor()
    return sv.kb_manager.kb_stats()


# ── Analytics ──────────────────────────────────────────────────────────────────

@app.get("/analytics", tags=["Analytics"])
async def get_analytics(days: int = Query(30, ge=1, le=365)):
    """Return support system analytics for the past N days."""
    sv = get_supervisor()
    return sv.get_analytics(days=days)
