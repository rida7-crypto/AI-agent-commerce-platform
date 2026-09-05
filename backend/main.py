"""
Phase 4 backend / Phase 6 failure handling.
Run with: uvicorn main:app --reload
"""

import os
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent_graph
import razorpay_client
import audit_log
import rag_store

app = FastAPI(
    title="Agentic Commerce Demo",
    description=(
        "A bounded, explainable checkout agent for Razorpay test-mode. "
        "GET /api/catalog exposes the full product catalog in a stable, "
        "machine-readable schema so an external agent (not just a human in "
        "the chat UI) can browse products programmatically before deciding "
        "what to recommend or buy."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store - fine for a demo. Swap for Redis/DB for production.
SESSIONS: dict[str, dict] = {}


@app.on_event("startup")
def startup():
    rag_store.build_index()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float
    stock: int
    rating: float
    description: str


@app.get("/api/catalog", response_model=list[Product])
def get_catalog():
    """Agent-readable catalog: any external agent (human-driven chat UI,
    or another AI buyer speaking a protocol like ACP/AP2) can call this
    directly to see exactly what's for sale, at what price, with no need
    to go through natural-language chat first."""
    return rag_store.load_catalog()


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    # Carry forward budget/category from the previous turn in this session,
    # so a short follow-up ("make it 900") doesn't lose context the way it
    # would if every message were treated as a totally fresh, memoryless
    # request. understand_intent only overwrites these if the new message
    # actually mentions a new budget/category - otherwise it keeps these.
    prior = SESSIONS.get(session_id, {})
    state = {
        "session_id": session_id,
        "user_message": req.message,
        "budget": prior.get("budget"),
        "category": prior.get("category"),
    }
    result = agent_graph.recommendation_graph.invoke(state)
    SESSIONS[session_id] = result
    return {
        "session_id": session_id,
        "reply": result.get("reply"),
        "status": result.get("status"),
        "cart_item": result.get("cart_item"),
        "options": result.get("options"),
    }


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool
    item_id: str | None = None  # which offered option the user picked


@app.post("/api/confirm")
def confirm(req: ConfirmRequest):
    state = SESSIONS.get(req.session_id)
    if not state:
        return {"error": "Nothing pending confirmation for this session."}

    # Idempotency guard: if this session already has an order, don't create
    # a second one - just hand back the original. Without this, a double
    # click or a client retry after a slow/dropped response would create two
    # real Razorpay orders for the same confirmed intent.
    if state.get("status") == "order_created" and state.get("order"):
        existing = state["order"]
        audit_log.log_step(req.session_id, "order_reused", {"order": existing})
        return {
            "status": "order_created",
            "razorpay_order_id": existing["id"],
            "amount": existing["amount"],
            "currency": existing["currency"],
            "razorpay_key_id": os.getenv("RAZORPAY_KEY_ID"),
            "item": state["cart_item"],
        }

    if state.get("status") != "needs_confirmation":
        return {"error": "Nothing pending confirmation for this session."}

    if not req.confirm:
        audit_log.log_step(req.session_id, "user_declined", {})
        return {"status": "cancelled", "message": "No problem, order cancelled."}

    # Guardrail: the chosen item must be one that was actually offered in
    # this session - never trust an item_id/price coming straight off the
    # wire, since that would let a client (or a compromised agent) confirm
    # payment for something never shown to the user.
    options = state.get("options") or [state["cart_item"]]
    if req.item_id is not None:
        item = next((o for o in options if str(o["id"]) == str(req.item_id)), None)
        if item is None:
            audit_log.log_step(
                req.session_id,
                "guardrail_block",
                {"reason": f"item_id {req.item_id} was not among the offered options"},
            )
            return {"error": "That item wasn't one of the options offered - please pick from the list shown."}
    else:
        item = options[0]  # no explicit choice -> default to the top pick

    order = razorpay_client.create_order(
        amount_rupees=item["price"], receipt=f"receipt_{req.session_id[:8]}"
    )
    audit_log.log_step(req.session_id, "order_created", {"order": order, "item": item})
    state["cart_item"] = item  # remember the chosen item for verify-payment
    state["order"] = order      # remember the order itself, for the idempotency check above
    state["status"] = "order_created"

    return {
        "status": "order_created",
        "razorpay_order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "razorpay_key_id": os.getenv("RAZORPAY_KEY_ID"),
        "item": item,
    }


class VerifyRequest(BaseModel):
    session_id: str
    razorpay_order_id: str
    razorpay_payment_id: str | None = None
    razorpay_signature: str | None = None
    failed: bool = False
    error_description: str | None = None


@app.post("/api/verify-payment")
def verify_payment(req: VerifyRequest):
    # Phase 6: the one failure, handled gracefully instead of crashing.
    if req.failed:
        reason = req.error_description or "Payment was not completed"
        audit_log.log_step(req.session_id, "payment_failed", {"reason": reason})
        state = SESSIONS.get(req.session_id)
        item = state.get("cart_item") if state else None
        item_name = item["name"] if item else "your order"
        message = agent_graph.generate_payment_failed_reply(req.session_id, item_name, reason)
        return {"status": "failed", "message": message}

    ok = razorpay_client.verify_payment(
        req.razorpay_order_id, req.razorpay_payment_id, req.razorpay_signature
    )
    if ok:
        audit_log.log_step(req.session_id, "payment_verified", {"payment_id": req.razorpay_payment_id})
        return {"status": "success", "message": "Payment verified. Order placed!"}

    audit_log.log_step(
        req.session_id, "payment_verification_failed", {"payment_id": req.razorpay_payment_id}
    )
    return {"status": "failed", "message": "Payment signature didn't match. Please retry."}


@app.get("/api/audit/{session_id}")
def get_audit(session_id: str):
    return {"session_id": session_id, "trail": audit_log.get_trail(session_id)}