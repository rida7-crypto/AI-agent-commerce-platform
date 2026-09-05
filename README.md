# Agent Commerce — an AI agent that can actually *buy* things

**A bounded, explainable checkout agent that goes from "gift for mom, ₹1500, skincare" to a verified Razorpay payment — with a full audit trail of every decision it made along the way.**

> Every LLM demo can *chat*. This one can *transact* — safely, provably, and without you ever having to trust it blindly.

---

## The pitch

Most "AI shopping agent" demos stop at a recommendation. The scary part — and the interesting part — is what happens *after*: does the agent get to spend real money on its own judgment?

**Agent Commerce says no — not without you.** It's an agent that can search, reason, and negotiate its way to a great recommendation, but is architecturally *incapable* of completing a payment without an explicit human confirmation and a server-side guardrail check. Every single step it takes — every search, every guardrail, every LLM draft and self-critique, every payment event — is written to a timestamped, read-only audit log that streams live to the UI. Nothing about this agent is a black box.

## Why it's more than a chatbot with a checkout button

- **Retrieval-grounded, not hallucinated.** Product recommendations come from a real vector search (Chroma + sentence-transformers) over an actual catalog — the agent can never recommend something that doesn't exist.
- **Self-critiquing generation loop.** A LangGraph state machine drafts the user-facing reply, then a *second, independent* LLM pass critiques it against the real options before it's shown — capped so a stubborn model can never hang the request.
- **Guardrails that actually gate money.** A hard spend ceiling, a "must be one of the options actually shown" check before any order is created, and an idempotency guard so a double-click can't spin up two real Razorpay orders — all enforced server-side, not just prompted for.
- **Radical explainability.** Every node in the agent's graph — intent parsing, catalog search, guardrail pass/block, draft, critique, order creation, payment verification — writes to `audit.db` and renders live in a receipt-style trail the user can watch in real time.
- **Agent-to-agent ready.** `/api/catalog` exposes the product catalog in a stable, machine-readable schema, so this isn't just a human-in-a-chat-window demo — another AI buyer speaking a protocol like ACP/AP2 could shop here too.
- **Real payment rails, test-mode safety.** Full Razorpay Checkout + signature verification flow, wired end-to-end — just running against test keys so no real money ever moves.

## How it works

```
User message
    │
    ▼
understand_intent   → parses budget + category, remembers context across turns
    │
    ▼
search_catalog       → RAG search over the real catalog (Chroma)
    │
    ▼
apply_guardrails     → hard budget cap · no-match handling · top-3 ranked options
    │
    ▼
draft_reply ⇄ critique_reply   → LLM writes the pitch, a second LLM checks it's honest
    │
    ▼
[ user confirms an option ]
    │
    ▼
/api/confirm          → re-validates the item was actually offered → creates Razorpay order
    │
    ▼
Razorpay Checkout      → user pays (or cancels, or it fails) in test mode
    │
    ▼
/api/verify-payment    → signature-verified, logged, and reflected back to the user
```

Every arrow above is a line in the audit trail.

## Stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph state machine |
| Retrieval | ChromaDB + `sentence-transformers` (local, no API key needed) |
| Generation | Llama 3.1 8B via Hugging Face Inference API (graceful templated fallback if unset) |
| Payments | Razorpay (test mode) |
| Backend | FastAPI |
| Audit trail | SQLite, timestamped, append-only |
| Frontend | Single-file HTML/CSS/JS chat + live receipt UI |

## Run it

```bash
pip install -r requirements.txt

# .env (backend/)
RAZORPAY_KEY_ID=your_test_key_id
RAZORPAY_KEY_SECRET=your_test_key_secret
HUGGINGFACE_TOKEN=your_hf_token   # optional — falls back to templated replies without it

uvicorn main:app --reload
```

Then open `index.html`. Try:

> *"gift for mom, budget 1500, skincare"*

Watch the agent search, guard, draft, and offer three real products — then confirm one and pay with a Razorpay test card. Watch the right-hand panel log every single step as it happens.

## What makes this hackathon-worthy

It's not "an agent that uses an LLM." It's a **system design answer to the actual hard question in agentic commerce**: how do you let an autonomous agent act in the world — search, decide, spend — while keeping a human (or an auditor) able to see and stop every step of it? Guardrails aren't a system prompt here; they're code paths a determined jailbreak can't talk its way around. That's the demo.

---

*Test mode only. No real payments. Every action logged.*
