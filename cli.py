"""
cli.py
------
Interactive CLI for the Multi-Agent RAG Customer Support System.

Usage:
    python -m multi_agent_rag_support.cli
    python -m multi_agent_rag_support.cli --email user@example.com
    python -m multi_agent_rag_support.cli --resume TKT-XXXXXXXX
    python -m multi_agent_rag_support.cli --ingest ./docs/ --category technical
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

from supervisor import Supervisor
from models import SupportCategory, TicketStatus
from settings import Settings


# ── ANSI colours ───────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GREY   = "\033[90m"
    BLUE   = "\033[94m"


def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════╗
║   Multi-Agent RAG Customer Support System  v1.0      ║
║   Type  /help  for commands                          ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")


def print_help():
    print(f"""
{C.BOLD}Available commands:{C.RESET}
  {C.YELLOW}/resolve <summary>{C.RESET}   Mark ticket resolved with a summary
  {C.YELLOW}/feedback <1-5>{C.RESET}      Submit CSAT rating
  {C.YELLOW}/status{C.RESET}              Show current ticket status
  {C.YELLOW}/history{C.RESET}             Print full conversation history
  {C.YELLOW}/analytics{C.RESET}           Show system analytics
  {C.YELLOW}/human{C.RESET}               Request a human agent
  {C.YELLOW}/new{C.RESET}                 Start a new ticket
  {C.YELLOW}/quit{C.RESET}                Exit
""")


def print_ticket_status(ticket):
    status_colours = {
        "open": C.BLUE, "in_progress": C.CYAN,
        "escalated": C.RED, "pending_human": C.RED,
        "resolved": C.GREEN, "closed": C.GREY,
    }
    sc = status_colours.get(ticket.status.value, C.RESET)
    print(f"""
{C.GREY}┌─ Ticket Status ─────────────────────────────────┐{C.RESET}
{C.GREY}│{C.RESET}  ID       : {C.BOLD}{ticket.ticket_id}{C.RESET}
{C.GREY}│{C.RESET}  Status   : {sc}{ticket.status.value.upper()}{C.RESET}
{C.GREY}│{C.RESET}  Category : {ticket.category.value}
{C.GREY}│{C.RESET}  Priority : {ticket.priority.value}
{C.GREY}│{C.RESET}  Agent    : {ticket.current_agent.value}
{C.GREY}│{C.RESET}  Conf.    : {ticket.confidence_score:.0%}
{C.GREY}│{C.RESET}  Subject  : {ticket.subject or '(pending)'}
{C.GREY}└─────────────────────────────────────────────────┘{C.RESET}
""")


def print_agent_response(response_text: str, ticket):
    tier_label = ticket.current_agent.value.upper()
    is_escalated = ticket.status.value in ("escalated", "pending_human")

    label_colour = C.RED if is_escalated else C.GREEN
    print(f"\n{label_colour}{C.BOLD}[{tier_label}]{C.RESET}")
    print(f"{C.CYAN}{'─' * 60}{C.RESET}")

    # Word-wrap at 60 chars
    import textwrap
    for line in response_text.split("\n"):
        wrapped = textwrap.fill(line, width=70) if line.strip() else ""
        print(f"  {wrapped}" if wrapped else "")

    print(f"{C.CYAN}{'─' * 60}{C.RESET}")
    if is_escalated:
        print(f"  {C.RED}⚠  Escalated to human queue{C.RESET}")
    else:
        print(f"  {C.GREY}Confidence: {ticket.confidence_score:.0%} | "
              f"Attempt: {ticket.resolution_attempts}{C.RESET}")
    print()


def run_chat(supervisor: Supervisor, email: str, resume_id: str | None):
    print_banner()

    ticket_id = resume_id
    ticket = None

    if resume_id:
        ticket = supervisor.get_ticket(resume_id)
        if not ticket:
            print(f"{C.RED}Ticket {resume_id} not found.{C.RESET}")
            return
        print(f"{C.GREEN}Resuming ticket {ticket.ticket_id} ({ticket.status.value}){C.RESET}\n")
        print_ticket_status(ticket)

    print(f"{C.GREY}Type your message, or use /help for commands.{C.RESET}\n")

    while True:
        try:
            user_input = input(f"{C.YELLOW}You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("Goodbye!")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/new":
                ticket_id = None
                ticket = None
                print(f"{C.CYAN}Started new session.{C.RESET}\n")

            elif cmd == "/status":
                if ticket:
                    print_ticket_status(ticket)
                else:
                    print(f"{C.GREY}No active ticket.{C.RESET}")

            elif cmd == "/history":
                if ticket:
                    print(f"\n{C.BOLD}Conversation history:{C.RESET}")
                    for m in ticket.messages:
                        tier = f" [{m.agent_tier.value}]" if m.agent_tier else ""
                        role_col = C.YELLOW if m.role == "user" else C.GREEN
                        print(f"  {role_col}{m.role.upper()}{tier}:{C.RESET} {m.content[:120]}")
                    print()
                else:
                    print(f"{C.GREY}No active ticket.{C.RESET}")

            elif cmd == "/analytics":
                analytics = supervisor.get_analytics(days=30)
                print(f"""
{C.BOLD}System Analytics (last 30 days):{C.RESET}
  Total Tickets    : {analytics['total_tickets']}
  Resolved         : {analytics['resolved']}
  Auto-Resolved    : {analytics['auto_resolved']}
  Escalated        : {analytics['escalated']}
  Resolution Rate  : {analytics['resolution_rate']:.0%}
  Automation Rate  : {analytics['automation_rate']:.0%}
  Avg CSAT         : {analytics['avg_csat'] or 'N/A'}
  KB Articles      : {analytics['kb_stats']['kb_articles']}
  Active Sessions  : {analytics['active_sessions']}
""")

            elif cmd == "/resolve":
                if not ticket_id:
                    print(f"{C.RED}No active ticket to resolve.{C.RESET}")
                else:
                    summary = arg or "Resolved via CLI."
                    supervisor.resolve(ticket_id, summary, was_auto=False)
                    print(f"{C.GREEN}✅  Ticket {ticket_id} resolved.{C.RESET}")
                    ticket_id = None
                    ticket = None

            elif cmd == "/feedback":
                if not ticket_id:
                    print(f"{C.RED}No active ticket.{C.RESET}")
                elif not arg.isdigit() or not (1 <= int(arg) <= 5):
                    print(f"{C.RED}Score must be 1–5.{C.RESET}")
                else:
                    supervisor.record_feedback(ticket_id, int(arg))
                    print(f"{C.GREEN}⭐  Feedback recorded: {arg}/5{C.RESET}")

            elif cmd == "/human":
                user_input = "I want to speak to a human agent please."
                # Fall through to normal message handling below
                # (don't continue, let the message be sent)
            else:
                print(f"{C.RED}Unknown command: {cmd}. Type /help for help.{C.RESET}")
                continue

            # If /human, fall through to message handling
            if cmd != "/human":
                continue

        # ── Normal message ─────────────────────────────────────────────
        try:
            response_text, ticket = supervisor.handle(
                message=user_input,
                ticket_id=ticket_id,
                customer_email=email,
            )
            ticket_id = ticket.ticket_id
            print_agent_response(response_text, ticket)

        except Exception as exc:
            logging.exception("Error handling message")
            print(f"{C.RED}Error: {exc}{C.RESET}")


def ingest_mode(supervisor: Supervisor, path: str, category: str):
    try:
        cat = SupportCategory(category)
    except ValueError:
        cat = SupportCategory.GENERAL

    print(f"Ingesting '{path}' into KB (category={cat.value})...")
    chunks = supervisor.ingest_kb_file(path, category=cat)
    print(f"✅  Ingested {chunks} chunks into the knowledge base.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Customer Support CLI"
    )
    parser.add_argument("--email", default="", help="Customer email address")
    parser.add_argument("--resume", default=None, help="Resume an existing ticket ID")
    parser.add_argument("--ingest", default=None, help="Ingest a file/dir into the KB and exit")
    parser.add_argument("--category", default="general",
                        help="KB category for --ingest (billing|technical|account|product|general)")
    parser.add_argument("--model", default=None, help="Override LLM model (e.g. mistral, gemma)")

    args = parser.parse_args()

    settings = Settings()
    if args.model:
        settings.llm_model = args.model
        settings.fast_model = args.model

    supervisor = Supervisor(settings)

    if args.ingest:
        ingest_mode(supervisor, args.ingest, args.category)
    else:
        run_chat(supervisor, email=args.email, resume_id=args.resume)


if __name__ == "__main__":
    main()
