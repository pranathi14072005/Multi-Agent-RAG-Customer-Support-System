"""
demo.py
-------
End-to-end demonstration of the Multi-Agent RAG Customer Support System.

Run with:
    python demo.py

Requires Ollama running with:
    ollama pull llama3
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import logging
import textwrap

logging.basicConfig(
    level=logging.WARNING,   # Set to INFO/DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from multi_agent_rag_support import Supervisor, Settings, SupportCategory


def banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def show_response(response: str, ticket) -> None:
    print(f"\n🤖 Agent ({ticket.current_agent.value.upper()}):")
    for line in textwrap.wrap(response, 68):
        print(f"   {line}")
    print(f"\n   📋 Ticket: {ticket.ticket_id} | "
          f"Status: {ticket.status.value} | "
          f"Priority: {ticket.priority.value} | "
          f"Category: {ticket.category.value} | "
          f"Confidence: {ticket.confidence_score:.2f}")


def main():
    # ── Initialise ─────────────────────────────────────────────────────
    banner("Initialising Supervisor")
    settings = Settings(company_name="TechFlow Inc")
    supervisor = Supervisor(settings)

    # ── Seed Knowledge Base ────────────────────────────────────────────
    banner("Seeding Knowledge Base")

    billing_article = """
# Billing & Refund Policy — TechFlow Inc

## Subscription Charges
- Monthly subscriptions are billed on the 1st of each month.
- Annual subscriptions are billed on the anniversary of signup.
- Charges appear on statements as "TECHFLOW INC".

## Duplicate Charges
If you see a duplicate charge:
1. Check your bank statement for the transaction dates.
2. Contact support with your account email and the charge amount.
3. Provide a screenshot of both transactions if possible.
4. Refunds for confirmed duplicates are processed within 3-5 business days.

## Refund Policy
- Refunds are available within 30 days of purchase.
- Annual plan refunds are prorated.
- Partial-month refunds are not available for monthly plans.
- Refunds are issued to the original payment method.

## Cancellation
- Cancel any time from Account Settings > Billing > Cancel Subscription.
- Access continues until the end of the current billing period.
- Cancellations take effect at the next renewal date.
"""

    technical_article = """
# Common Technical Issues — TechFlow Inc

## Login Problems

### Forgot Password
1. Go to login page and click "Forgot Password"
2. Enter your registered email address
3. Check your inbox for a reset link (check spam folder too)
4. Link expires in 24 hours

### Two-Factor Authentication Issues
- If your 2FA app is not syncing: ensure your device clock is set to automatic
- Backup codes can be used if you've lost access to your 2FA device
- Contact support to disable 2FA if you've lost all access methods

## API Integration Issues

### 401 Unauthorized
- Verify your API key is correct and not expired
- API keys from the sandbox environment cannot be used in production
- Regenerate your API key at Settings > API > Keys

### Rate Limiting (429 errors)
- Default rate limit: 100 requests/minute
- Enterprise plans: 1000 requests/minute
- Implement exponential backoff in your code

## Data Export
- Export your data at any time from Settings > Data > Export
- CSV and JSON formats available
- Large exports may take up to 30 minutes to prepare
"""

    account_article = """
# Account Management — TechFlow Inc

## Account Recovery
To recover a locked account:
1. Visit account-recovery.techflow.com
2. Enter your email address
3. Answer security questions or use backup email
4. If all else fails, submit an ID verification request

## Data Privacy (GDPR / CCPA)
- Right to Access: Request a full data export at Settings > Privacy > Request Data
- Right to Erasure: Submit a deletion request at Settings > Privacy > Delete Account
- Data requests are processed within 30 days
- Contact dpo@techflow.com for data protection queries

## Account Sharing Policy
- Accounts are for individual use only
- Team plans allow multiple seats — upgrade at Settings > Team
- Sharing login credentials violates our Terms of Service

## Closing Your Account
- Delete your account permanently at Settings > Account > Close Account
- This action is irreversible — download your data first
- Closed accounts cannot be reactivated
"""

    supervisor.ingest_kb_text(billing_article, "Billing & Refund Policy",
                               category=SupportCategory.BILLING,
                               tags=["billing", "refund", "subscription"])
    supervisor.ingest_kb_text(technical_article, "Common Technical Issues",
                               category=SupportCategory.TECHNICAL,
                               tags=["login", "api", "2fa", "password"])
    supervisor.ingest_kb_text(account_article, "Account Management Guide",
                               category=SupportCategory.ACCOUNT,
                               tags=["account", "gdpr", "privacy", "recovery"])

    print("✅  KB seeded with 3 articles.")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 1: Simple billing question (Tier-1 auto-resolves)
    # ─────────────────────────────────────────────────────────────────
    banner("Scenario 1: Billing Question (Tier-1 Resolution)")

    print("\n👤 Customer: Hi, I think I was charged twice this month. What should I do?")
    response, ticket = supervisor.handle(
        "Hi, I think I was charged twice this month. What should I do?",
        customer_email="alice@example.com",
    )
    show_response(response, ticket)

    # Follow-up
    print("\n👤 Customer: I see two charges of $29.99 on the 5th and 6th of this month.")
    response, ticket = supervisor.handle(
        "I see two charges of $29.99 on the 5th and 6th of this month.",
        ticket_id=ticket.ticket_id,
    )
    show_response(response, ticket)

    # Resolve
    supervisor.resolve(
        ticket.ticket_id,
        "Advised customer to contact support with account email and charge screenshots. "
        "Confirmed refund policy: 3-5 business days for verified duplicates.",
        was_auto=True,
    )
    supervisor.record_feedback(ticket.ticket_id, score=5)
    print("\n✅  Ticket resolved. CSAT: 5/5")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 2: Technical issue (Tier-1, possibly Tier-2)
    # ─────────────────────────────────────────────────────────────────
    banner("Scenario 2: Technical Issue — API 401 Error")

    print("\n👤 Customer: My API integration keeps returning 401 errors. "
          "I've checked the key and it looks correct.")
    response, ticket2 = supervisor.handle(
        "My API integration keeps returning 401 errors. "
        "I've checked the key and it looks correct.",
        customer_email="dev@startup.io",
    )
    show_response(response, ticket2)

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 3: Explicit human request (immediate escalation)
    # ─────────────────────────────────────────────────────────────────
    banner("Scenario 3: Customer Requests Human Agent")

    print("\n👤 Customer: I need to speak to a human agent immediately.")
    response, ticket3 = supervisor.handle(
        "I need to speak to a human agent immediately about a serious billing dispute.",
        customer_email="bob@company.com",
    )
    show_response(response, ticket3)

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 4: Analytics
    # ─────────────────────────────────────────────────────────────────
    banner("System Analytics (Last 30 Days)")

    analytics = supervisor.get_analytics(days=30)
    print(f"\n  📊 Total Tickets   : {analytics['total_tickets']}")
    print(f"  ✅  Resolved        : {analytics['resolved']}")
    print(f"  🤖 Auto-Resolved   : {analytics['auto_resolved']}")
    print(f"  🚨 Escalated       : {analytics['escalated']}")
    print(f"  📈 Resolution Rate : {analytics['resolution_rate']:.0%}")
    print(f"  ⚡ Automation Rate : {analytics['automation_rate']:.0%}")
    print(f"  ⭐ Avg CSAT        : {analytics['avg_csat'] or 'N/A'}")
    print(f"\n  KB Articles        : {analytics['kb_stats']['kb_articles']}")
    print(f"  Indexed Tickets    : {analytics['kb_stats']['indexed_tickets']}")
    print(f"  Active Sessions    : {analytics['active_sessions']}")

    if analytics['by_category']:
        print(f"\n  By Category:")
        for cat, cnt in sorted(analytics['by_category'].items()):
            print(f"    {cat:<15} {cnt}")

    banner("Demo Complete")
    print("\n  The system is fully operational.\n"
          "  Integrate Supervisor.handle() into your API/UI layer.\n"
          "  See README.md for full configuration options.\n")


if __name__ == "__main__":
    main()
