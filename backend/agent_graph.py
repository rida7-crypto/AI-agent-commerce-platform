"""
Phase 2 + Phase 5: The agent's brain (LangGraph) and its guardrails.

Flow: understand_intent -> search_catalog -> apply_guardrails -> (stop and
wait for the user's yes/no before any payment happens - that part lives
in main.py's /api/confirm endpoint, kept separate on purpose so money
NEVER moves inside this graph).
"""

from typing import TypedDict, List, Optional
import re
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

import rag_store
import audit_log

load_dotenv()

_llm_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    import anthropic
    _llm_client = anthropic.Anthropic()

# Hard safety ceiling - no matter what the user (or a compromised prompt) says,
# the agent will never treat a budget above this as valid.
MAX_BUDGET_CAP = 5000


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    budget: Optional[float]
    category: Optional[str]
    matches: List[dict]
    cart_item: Optional[dict]
    status: str
    reply: str


def understand_intent(state: AgentState) -> AgentState:
    message = state["user_message"]
    budget_match = re.search(r"(\d{2,6})", message.replace(",", ""))
    budget = float(budget_match.group(1)) if budget_match else None

    category = None
    for cat in ["skincare", "electronics", "books", "home", "fashion"]:
        if cat in message.lower():
            category = cat
            break

    audit_log.log_step(
        state["session_id"],
        "understand_intent",
        {"budget_detected": budget, "category_detected": category, "raw_message": message},
    )
    state["budget"] = budget
    state["category"] = category
    return state


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
        return state

    if not matches:
        audit_log.log_step(state["session_id"], "guardrail_block", {"reason": "No matching products found"})
        state["status"] = "no_match"
        state["reply"] = "I couldn't find anything matching that in the catalog. Try different words or a higher budget."
        return state

    best = max(matches, key=lambda m: m["match_score"])
    state["cart_item"] = best
    state["status"] = "needs_confirmation"
    state["reply"] = generate_reply(state)

    audit_log.log_step(
        state["session_id"],
        "guardrail_pass",
        {"picked": best, "status": "needs_confirmation"},
    )
    return state


def generate_reply(state: AgentState) -> str:
    best = state["cart_item"]
    if _llm_client is None:
        # Works even without an Anthropic API key, just less chatty.
        return (
            f"I found '{best['name']}' for Rs.{best['price']} "
            f"(match score {best['match_score']}). Want me to go ahead and pay for it?"
        )
    prompt = (
        f"User asked: {state['user_message']}\n"
        f"Best matching product from the real catalog: {best}\n"
        "Write one short, friendly sentence recommending this product and asking "
        "the user to confirm before payment. Do not invent details not given above."
    )
    resp = _llm_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def build_recommendation_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("search_catalog", search_catalog)
    graph.add_node("apply_guardrails", apply_guardrails)

    graph.set_entry_point("understand_intent")
    graph.add_edge("understand_intent", "search_catalog")
    graph.add_edge("search_catalog", "apply_guardrails")
    graph.add_edge("apply_guardrails", END)
    return graph.compile()


recommendation_graph = build_recommendation_graph()
