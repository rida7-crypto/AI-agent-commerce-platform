"""
Phase 3: Razorpay test-mode payments.
Uses your test key id/secret from .env - no real money ever moves.
"""

import os
import razorpay
from dotenv import load_dotenv

load_dotenv()

_client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)


def create_order(amount_rupees: float, receipt: str):
    """Razorpay wants the amount in paise (1 INR = 100 paise)."""
    order = _client.order.create(
        {
            "amount": int(round(amount_rupees * 100)),
            "currency": "INR",
            "receipt": receipt,
            "payment_capture": 1,
        }
    )
    return order


def verify_payment(order_id: str, payment_id: str, signature: str) -> bool:
    """Confirms the payment really came from Razorpay and wasn't tampered with."""
    try:
        _client.utility.verify_payment_signature(
            {
                "razorpay_order_id": order_id,
                "razorpay_payment_id": payment_id,
                "razorpay_signature": signature,
            }
        )
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
