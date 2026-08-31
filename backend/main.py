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

app = FastAPI(title="Agentic Commerce Demo")

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


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    state = {"session_id": session_id, "user_message": req.message}
    result = agent_graph.recommendation_graph.invoke(state)
    SESSIONS[session_id] = result
    return {
        "session_id": session_id,
        "reply": result.get("reply"),
        "status": result.get("status"),
        "cart_item": result.get("cart_item"),
    }


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool


@app.post("/api/confirm")
def confirm(req: ConfirmRequest):
    state = SESSIONS.get(req.session_id)
    if not state or state.get("status") != "needs_confirmation":
        return {"error": "Nothing pending confirmation for this session."}

    if not req.confirm:
        audit_log.log_step(req.session_id, "user_declined", {})
        return {"status": "cancelled", "message": "No problem, order cancelled."}

    item = state["cart_item"]
    order = razorpay_client.create_order(
        amount_rupees=item["price"], receipt=f"receipt_{req.session_id[:8]}"
    )
    audit_log.log_step(req.session_id, "order_created", {"order": order})

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
        audit_log.log_step(
            req.session_id,
            "payment_failed",
            {"reason": req.error_description or "Payment was not completed"},
        )
        return {
            "status": "failed",
            "message": "That payment didn't go through. Want to try again, or pick a different item within budget?",
        }

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
