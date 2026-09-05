"""
Phase 2 + Phase 5: The agent's brain (LangGraph) and its guardrails.

Flow: understand_intent -> search_catalog -> apply_guardrails ->
  [draft_reply <-> critique_reply, looping up to MAX_DRAFT_ATTEMPTS times] -> END

Payment itself never happens in this graph - after apply_guardrails sets
status="needs_confirmation", the graph only drafts and critiques the TEXT
shown to the user. The actual yes/no and payment live in main.py's
/api/confirm endpoint, kept separate on purpose so money NEVER moves here.
"""

from typing import TypedDict, List, Optional
import re
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

import rag_store
import audit_log

load_dotenv()

# LLM provider: a Llama model via the Hugging Face Inference API. If no
# token is set, the reply-generating functions below fall back to plain
# templated sentences - the agent still works end-to-end, just less chatty.
_llm_client = None
_llm_provider = None
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

if os.getenv("HUGGINGFACE_TOKEN"):
    from huggingface_hub import InferenceClient
    _llm_client = InferenceClient(provider="auto", api_key=os.getenv("HUGGINGFACE_TOKEN"))
    _llm_provider = "huggingface"

# Loud and clear at startup - so you never have to guess why replies look
# templated instead of LLM-generated.
if _llm_provider:
    print(f"[agent_graph] LLM provider active: {_llm_provider}")
else:
    print(
        "[agent_graph] WARNING: no HUGGINGFACE_TOKEN found in the environment "
        "- replies will use plain templated sentences, not an LLM. Check that "
        "backend/.env exists, is named exactly '.env', and that uvicorn was "
        "started from inside backend/ (or restarted after editing .env)."
    )


def _call_llm(prompt: str, session_id: str, max_tokens: int = 150) -> Optional[str]:
    """Shared helper: sends a prompt to whichever LLM provider is active.
    Returns None (never raises) if no provider is configured or the call
    fails, so callers can fall back to a fixed sentence."""
    if _llm_client is None:
        return None
    try:
        if _llm_provider == "huggingface":
            resp = _llm_client.chat.completions.create(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
    except Exception as e:
        # Log the REAL reason (bad token, gated model not accepted, rate
        # limit, etc.) instead of failing the request or silently going quiet.
        audit_log.log_step(
            session_id,
            "llm_call_failed",
            {"provider": _llm_provider, "error": str(e)},
        )
        print(f"[agent_graph] LLM call via '{_llm_provider}' failed: {e}")
    return None


def _clean_reply(text: str) -> str:
    """Strips common small-model leakage: leading meta-commentary like
    'Here is a response:', wrapping quote marks, and stray markdown bold."""
    text = text.strip()
    # Drop a leading "Here('s/ are) ... :" style preamble line, if present.
    text = re.sub(
        r'^\s*(here\'?s?|here are)\b[^:\n]*:\s*',
        '',
        text,
        flags=re.IGNORECASE,
    )
    # Strip wrapping quotes the model sometimes adds around the whole reply.
    text = text.strip('"\u201c\u201d')
    # Remove markdown bold/italic asterisks.
    text = text.replace('**', '').replace('*', '')
    return text.strip()


# Hard safety ceiling - no matter what the user (or a compromised prompt) says,
# the agent will never treat a budget above this as valid.
MAX_BUDGET_CAP = 5000

# How many times draft_reply <-> critique_reply is allowed to loop before
# the graph just accepts the latest draft as-is, so a stubborn small model
# can never hang the request forever.
MAX_DRAFT_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    budget: Optional[float]
    category: Optional[str]
    matches: List[dict]
    options: List[dict]
    cart_item: Optional[dict]
    status: str
    reply: str
    reply_attempt: int
    reply_ok: bool
    critique_feedback: Optional[str]




# describing a new product to search for. Without this check, a message
# like "nah" has no budget or category in it, so the search runs with no
# filters at all and returns near-random catalog matches - confusing and
# wrong, since the user wasn't asking for anything.
_CANCEL_PHRASES = {
    "no", "nah", "nope", "no thanks", "not really", "nevermind", "never mind",
    "cancel", "stop", "skip", "none", "n/a", "na",
}


def understand_intent(state: AgentState) -> AgentState:
    message = state["user_message"]
    normalized = re.sub(r"[^\w\s]", "", message).strip().lower()

    if normalized in _CANCEL_PHRASES:
        audit_log.log_step(
            state["session_id"],
            "understand_intent",
            {"detected": "cancel_intent", "raw_message": message},
        )
        state["status"] = "declined"
        state["reply"] = "No problem! Whenever you're ready, tell me what you're looking for and your budget."
        state["reply_ok"] = True
        return state

    budget_match = re.search(r"(\d{2,6})", message.replace(",", ""))
    new_budget = float(budget_match.group(1)) if budget_match else None

    new_category = None
    for cat in ["skincare", "electronics", "books", "home", "fashion"]:
        if cat in message.lower():
            new_category = cat
            break

    # Memory: only overwrite budget/category if THIS message actually
    # mentions one - otherwise keep whatever carried over from the previous
    # turn (passed in via state by main.py). This is what lets "make it 900"
    # work as a follow-up instead of wiping the category the user already gave.
    budget = new_budget if new_budget is not None else state.get("budget")
    category = new_category if new_category is not None else state.get("category")

    audit_log.log_step(
        state["session_id"],
        "understand_intent",
        {
            "budget_detected": new_budget, "category_detected": new_category,
            "budget_used": budget, "category_used": category,
            "raw_message": message,
        },
    )
    state["budget"] = budget
    state["category"] = category
    return state


def _route_after_intent(state: AgentState) -> str:
    return "end" if state.get("status") == "declined" else "search"


def search_catalog(state: AgentState) -> AgentState:
    matches = rag_store.search(
        query=state["user_message"],
        top_k=5,
        budget=state.get("budget"),
        category=state.get("category"),
    )
    audit_log.log_step(
        state["session_id"],
        "search_catalog",
        {"matches_found": len(matches), "matches": matches},
    )
    state["matches"] = matches
    return state


def apply_guardrails(state: AgentState) -> AgentState:
    matches = state.get("matches", [])
    budget = state.get("budget")

    if budget is not None and budget > MAX_BUDGET_CAP:
        audit_log.log_step(
            state["session_id"],
            "guardrail_block",
            {"reason": f"Requested budget {budget} exceeds hard cap {MAX_BUDGET_CAP}"},
        )
        state["status"] = "blocked"
        state["reply"] = f"That budget is above my allowed limit of Rs.{MAX_BUDGET_CAP}. Please give a smaller budget."
        state["reply_ok"] = True  # fixed, safe message - no need to critique it
        return state

    if not matches:
        audit_log.log_step(state["session_id"], "guardrail_block", {"reason": "No matching products found"})
        state["status"] = "no_match"
        state["reply"] = "I couldn't find anything matching that in the catalog. Try different words or a higher budget."
        state["reply_ok"] = True
        return state

    # Offer up to 3 options, best match first, instead of silently picking
    # for the user - lets them (or an AI buyer) choose, and sets up
    # upsell/cross-sell without any extra guardrail surface area, since
    # every option shown here still came from the real catalog via rag_store.
    ranked = sorted(matches, key=lambda m: m["match_score"], reverse=True)
    options = ranked[:3]

    state["options"] = options
    state["cart_item"] = options[0]  # default / top pick, kept for back-compat
    state["status"] = "needs_confirmation"
    state["reply_ok"] = False  # still needs a draft + critique pass

    audit_log.log_step(
        state["session_id"],
        "guardrail_pass",
        {"options": options, "status": "needs_confirmation"},
    )
    return state


def _route_after_guardrails(state: AgentState) -> str:
    return "end" if state.get("reply_ok") else "draft"


def draft_reply(state: AgentState) -> AgentState:
    """Writes (or rewrites, if the critic rejected the last attempt) the
    user-facing message describing the offered options."""
    attempt = state.get("reply_attempt", 0) + 1
    state["reply_attempt"] = attempt

    options = state["options"]
    listing = "; ".join(f"{o['name']} (Rs.{o['price']}, match {o['match_score']})" for o in options)

    feedback = state.get("critique_feedback")
    revision_note = f"\nYour previous draft had a problem: {feedback}\nFix that and rewrite it.\n" if feedback else ""

    prompt = (
        f"User asked: {state['user_message']}\n"
        f"Top matching products from the real catalog, best first: {listing}\n"
        f"{revision_note}"
        "Write a brief, friendly message (max 2 short sentences) presenting "
        "these options by name and price, and asking the user to pick one to "
        "confirm before payment. Do not invent details not given above.\n"
        "Respond with ONLY the exact message to show the user - no preamble "
        "like 'Here's a response' or 'Here are three sentences', no "
        "markdown formatting, no surrounding quotation marks."
    )

    raw = _call_llm(prompt, state["session_id"])
    if raw:
        state["reply"] = _clean_reply(raw)
        # A fresh (uncritiqued) draft is never auto-accepted here - the
        # critique_reply node decides that next.
    else:
        # No LLM configured, or the call failed - fixed fallback, skip critique.
        lines = [f"'{o['name']}' - Rs.{o['price']}" for o in options]
        state["reply"] = "Options: " + "; ".join(lines) + ". Which one should I pay for?"
        state["reply_ok"] = True

    audit_log.log_step(
        state["session_id"],
        "draft_reply",
        {"attempt": attempt, "reply": state["reply"], "used_llm": bool(raw)},
    )
    return state


def critique_reply(state: AgentState) -> AgentState:
    """A second, independent LLM call that checks the draft is user-friendly
    and grounded in the actual options - not the same call grading its own
    homework in one shot, but a separate pass with only the draft + options
    as input."""
    if state.get("reply_ok"):
        return state  # fixed fallback message already accepted, nothing to do

    if state["reply_attempt"] >= MAX_DRAFT_ATTEMPTS:
        # Give up looping so a stubborn small model can't hang the request;
        # log it clearly so this is visible in the audit trail, not silent.
        audit_log.log_step(
            state["session_id"],
            "critique_gave_up",
            {"reason": f"Reached MAX_DRAFT_ATTEMPTS={MAX_DRAFT_ATTEMPTS}, accepting latest draft."},
        )
        state["reply_ok"] = True
        return state

    options = state["options"]
    listing = "; ".join(f"{o['name']} (Rs.{o['price']})" for o in options)
    critique_prompt = (
        f"User's request: {state['user_message']}\n"
        f"Real options offered: {listing}\n"
        f"Draft reply to check: \"{state['reply']}\"\n\n"
        "Judge the draft against two rules only:\n"
        "1. It reads like a direct message to the user - no meta-commentary "
        "such as 'Here is a response', no leftover markdown, no preamble.\n"
        "2. It only mentions the products/prices listed above - nothing invented.\n\n"
        "Respond in EXACTLY this format, nothing else:\n"
        "VERDICT: PASS\n"
        "or\n"
        "VERDICT: FAIL\nFEEDBACK: <one short sentence saying what's wrong>"
    )

    verdict_text = _call_llm(critique_prompt, state["session_id"], max_tokens=80)

    if not verdict_text:
        # Critic call itself failed - fail open rather than looping forever
        # or blocking the user on an infra hiccup.
        audit_log.log_step(state["session_id"], "critique_reply", {"attempt": state["reply_attempt"], "result": "critic_unavailable"})
        state["reply_ok"] = True
        return state

    passed = bool(re.search(r"VERDICT:\s*PASS", verdict_text, re.IGNORECASE))
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", verdict_text, re.IGNORECASE | re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else None

    audit_log.log_step(
        state["session_id"],
        "critique_reply",
        {"attempt": state["reply_attempt"], "passed": passed, "feedback": feedback},
    )

    state["reply_ok"] = passed
    state["critique_feedback"] = None if passed else feedback
    return state


def _route_after_critique(state: AgentState) -> str:
    return "end" if state.get("reply_ok") else "retry"


def generate_payment_failed_reply(session_id: str, item_name: str, reason: str) -> str:
    """Standalone helper (not part of the graph loop) for the payment-failed
    message shown in main.py's /api/verify-payment. Falls back to a fixed
    sentence if no LLM is configured or the call fails."""
    prompt = (
        f"A payment for '{item_name}' just failed. Reason: {reason}\n"
        "Write one short, reassuring sentence telling the user the payment "
        "didn't go through, and ask if they'd like to try again or pick a "
        "different item within budget. Do not invent a different reason "
        "than the one given.\n"
        "Respond with ONLY the message to show the user - no preamble, no "
        "markdown formatting, no surrounding quotation marks."
    )
    reply = _call_llm(prompt, session_id)
    if reply:
        return _clean_reply(reply)

    return "That payment didn't go through. Want to try again, or pick a different item within budget?"


def build_recommendation_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("search_catalog", search_catalog)
    graph.add_node("apply_guardrails", apply_guardrails)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("critique_reply", critique_reply)

    graph.set_entry_point("understand_intent")
    graph.add_conditional_edges(
        "understand_intent", _route_after_intent, {"search": "search_catalog", "end": END}
    )
    graph.add_edge("search_catalog", "apply_guardrails")
    graph.add_conditional_edges(
        "apply_guardrails", _route_after_guardrails, {"draft": "draft_reply", "end": END}
    )
    graph.add_edge("draft_reply", "critique_reply")
    graph.add_conditional_edges(
        "critique_reply", _route_after_critique, {"retry": "draft_reply", "end": END}
    )
    return graph.compile()


recommendation_graph = build_recommendation_graph()