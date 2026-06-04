# Multi-Agent RAG Customer Support System

An enterprise-grade, fully offline customer support system built on LangChain + Ollama. Combines intelligent routing, retrieval-augmented generation (RAG), and multi-tier escalation to automatically resolve support tickets — and learn from them.

---

## Architecture

```
Customer Message
      │
      ▼
┌─────────────────┐
│  IntakeAgent    │  Classifies category, priority, sentiment (fast LLM)
└────────┬────────┘
         │ RoutingDecision
         ▼
┌─────────────────┐
│   Tier1Agent    │  RAG-based auto-resolver (ChromaDB + Ollama)
│  (automated)    │  Self-scores confidence → escalates if low
└────────┬────────┘
         │ if confidence < threshold OR attempts exceeded
         ▼
┌─────────────────┐
│   Tier2Agent    │  Domain specialist: billing / technical / account
│  (specialist)   │  Broader KB + similar-case retrieval
└────────┬────────┘
         │ if still unresolved
         ▼
┌─────────────────┐
│ EscalationAgent │  Writes handoff summary → human agent queue
└─────────────────┘
         │ post-resolution
         ▼
┌─────────────────┐
│ KBManagerAgent  │  Analyses resolved tickets → auto-updates KB
└─────────────────┘
```

All agents are coordinated by the **Supervisor**, which is the single public entry point.

---

## File Structure

```
multi_agent_rag_support/
├── __init__.py              # Public API exports
├── settings.py              # Central config (env-var driven)
├── models.py                # All data models and enums
├── supervisor.py            # Master orchestrator
│
├── agents/
│   ├── intake_agent.py      # Triage & classification
│   ├── tier1_agent.py       # RAG-powered auto-resolver
│   ├── tier2_agent.py       # Domain specialist
│   ├── escalation_agent.py  # Human handoff coordinator
│   └── kb_manager_agent.py  # KB auto-maintenance
│
├── kb/
│   └── knowledge_base.py    # ChromaDB dual-collection manager
│
├── memory/
│   ├── ticket_store.py      # SQLite-backed ticket persistence
│   └── session_memory.py    # In-process session/TTL management
│
├── prompts/
│   └── agent_prompts.py     # All LangChain prompt templates
│
├── api.py                   # FastAPI REST interface
├── cli.py                   # Interactive CLI
├── demo.py                  # End-to-end demo script
├── requirements.txt
└── tests/
    └── test_system.py       # 39-test unit suite
```

---

## Prerequisites

1. **Install Ollama** — https://ollama.com/download

2. **Pull required models:**
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### As a Python library

```python
from multi_agent_rag_support import Supervisor, SupportCategory

# Initialise (connects to Ollama at localhost:11434)
supervisor = Supervisor()

# Add knowledge base articles
supervisor.ingest_kb_text(
    text="Refunds are processed within 3-5 business days...",
    title="Refund Policy",
    category=SupportCategory.BILLING,
    tags=["billing", "refund"],
)
supervisor.ingest_kb_file("./support_docs/", category=SupportCategory.TECHNICAL)

# Handle a customer message (new ticket)
response, ticket = supervisor.handle(
    "I was charged twice this month!",
    customer_email="customer@example.com",
)
print(f"[{ticket.ticket_id}] {response}")

# Continue the conversation
response, ticket = supervisor.handle(
    "I see two charges of $29.99 on the 5th and 6th.",
    ticket_id=ticket.ticket_id,
)

# Resolve
supervisor.resolve(ticket.ticket_id, "Confirmed duplicate charge, issued refund.", was_auto=True)
supervisor.record_feedback(ticket.ticket_id, score=5)
```

### Interactive CLI

```bash
# New conversation
python -m multi_agent_rag_support.cli --email user@example.com

# Resume an existing ticket
python -m multi_agent_rag_support.cli --resume TKT-ABC12345

# Ingest a directory of docs first
python -m multi_agent_rag_support.cli --ingest ./docs/ --category technical

# Use a different model
python -m multi_agent_rag_support.cli --model mistral
```

CLI commands (type while chatting):
| Command | Description |
|---|---|
| `/resolve <summary>` | Mark ticket resolved |
| `/feedback <1-5>` | Submit CSAT rating |
| `/status` | Show ticket status |
| `/history` | Print conversation |
| `/analytics` | System analytics |
| `/human` | Request human agent |
| `/new` | Start fresh ticket |

### REST API

```bash
uvicorn multi_agent_rag_support.api:app --reload --port 8000
```

Interactive docs at http://localhost:8000/docs

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/tickets` | Create ticket / first message |
| `POST` | `/tickets/{id}/messages` | Send follow-up message |
| `GET` | `/tickets/{id}` | Get ticket details |
| `GET` | `/tickets` | List tickets (filterable) |
| `POST` | `/tickets/{id}/resolve` | Resolve ticket |
| `POST` | `/tickets/{id}/feedback` | Submit CSAT |
| `POST` | `/kb/ingest/text` | Add KB article |
| `POST` | `/kb/ingest/file` | Upload file to KB |
| `GET` | `/analytics` | System analytics |
| `GET` | `/health` | Health check |

### Run the demo

```bash
python -m multi_agent_rag_support.demo
```

---

## Configuration

All settings are readable from environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3` | Main generation model |
| `FAST_MODEL` | `llama3` | Lightweight model for classification |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `TICKET_DB_PATH` | `./tickets.db` | SQLite database path |
| `CHUNK_SIZE` | `512` | KB chunk size (chars) |
| `CHUNK_OVERLAP` | `64` | KB chunk overlap (chars) |
| `RETRIEVER_TOP_K` | `5` | KB chunks per query |
| `MAX_MEMORY_TURNS` | `20` | Conversation turns to keep |
| `ESCALATION_CONFIDENCE_THRESHOLD` | `0.4` | Confidence below which Tier-1 escalates |
| `MAX_AUTO_RESOLUTION_ATTEMPTS` | `3` | Tier-1 attempts before forced escalation |
| `SENTIMENT_ESCALATION_THRESHOLD` | `-0.6` | Sentiment score below which to escalate |
| `AUTO_KB_UPDATE` | `true` | Auto-add resolved tickets to KB |
| `COMPANY_NAME` | `Acme Corp` | Used in agent prompts |

Example `.env`:
```bash
OLLAMA_BASE_URL=http://ollama-server:11434
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
COMPANY_NAME=TechFlow Inc
TICKET_DB_PATH=/data/tickets.db
CHROMA_PERSIST_DIR=/data/chroma
ESCALATION_CONFIDENCE_THRESHOLD=0.45
AUTO_KB_UPDATE=true
```

---

## Escalation Rules

The system escalates automatically when:

| Trigger | Rule |
|---|---|
| **Low RAG confidence** | Tier-1 answer confidence < `ESCALATION_CONFIDENCE_THRESHOLD` |
| **Repeated failures** | > `MAX_AUTO_RESOLUTION_ATTEMPTS` failed attempts |
| **Negative sentiment** | Sentiment score < `SENTIMENT_ESCALATION_THRESHOLD` |
| **Critical keywords** | "data breach", "lawsuit", "ransomware", etc. in message |
| **User request** | "I want to speak to a human" detected |
| **Empty KB** | Knowledge base has no documents |

Tier-1 → Tier-2 transitions happen silently in the same turn. Tier-2 → human escalation generates a handoff summary and changes ticket status to `pending_human`.

---

## Knowledge Base Management

The KB uses **two ChromaDB collections**:

1. **`knowledge_base`** — Curated support articles (used by Tier-1 + Tier-2)
2. **`resolved_tickets`** — Indexed past resolutions (used by Tier-2 for similar-case lookup)

### Manual ingestion
```python
supervisor.ingest_kb_file("./docs/billing-faq.pdf", category=SupportCategory.BILLING)
supervisor.ingest_kb_text(article_text, title="Password Reset Guide", category=SupportCategory.ACCOUNT)
```

### Auto-learning from resolved tickets
When `AUTO_KB_UPDATE=true`, every resolved ticket is:
1. Analysed by `KBManagerAgent` — decides if resolution contains new knowledge
2. If yes: a polished KB article is drafted and ingested
3. Always: the ticket is indexed in the `resolved_tickets` collection for Tier-2 precedent lookup

Run a batch update on historical tickets:
```python
results = supervisor.run_kb_update_batch(days=90)
```

---

## Extending the System

### Add a new Tier-2 domain
1. Add a system prompt to `prompts/agent_prompts.py`
2. Map your new `SupportCategory` → prompt in `agents/tier2_agent.py`'s `TIER2_SYSTEM_MAP`

### Add alerting on escalation
Extend `EscalationAgent._alert()` in `agents/escalation_agent.py`:
```python
def _alert(self, ticket, reason, handoff_summary):
    slack_client.send(channel="#support-escalations", text=handoff_summary)
    zendesk_client.create_ticket(ticket, handoff_summary)
```

### Swap routing strategy
Replace the keyword heuristic in `Supervisor._route()` with any of the routers from your existing `routing.py`:
```python
from routing import RouterFactory
router = RouterFactory.create(strategy="embedding")
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v -k "not integration"   # skip tests requiring Ollama
```

All 39 unit tests run offline (Ollama LLM calls are mocked).

---

## Production Checklist

- [ ] Set `COMPANY_NAME` to your company name
- [ ] Load your support documentation into the KB before going live
- [ ] Tune `ESCALATION_CONFIDENCE_THRESHOLD` based on your tolerance for false escalations
- [ ] Implement real alerting in `EscalationAgent._alert()` (Slack, email, Zendesk, etc.)
- [ ] Set up a cron job to run `supervisor.run_kb_update_batch()` nightly
- [ ] Back up `./tickets.db` and `./chroma_db/` regularly
- [ ] Run `ollama serve` as a system service
- [ ] Consider `mistral` or `llama3.1:8b` for better quality at acceptable speed
