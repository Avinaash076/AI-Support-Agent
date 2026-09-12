import sys
import os
from typing import Dict, Any, List

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import LLMClient

class EscalationEngine:
    """Decides AUTO_HANDLE vs ESCALATE_TO_HUMAN with explicit rationale"""
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def evaluate(self, query: str, intent: str, max_similarity: float, drafted_reply: str) -> Dict[str, Any]:
        reasons = []
        should_escalate = False

        # Rule 1: Security & Apple ID Lockouts
        if intent == "apple_id_security" or any(kw in query.lower() for kw in ["password", "hacked", "stolen", "unauthorized", "locked out"]):
            should_escalate = True
            reasons.append("Account security and Apple ID credentials require human identity verification.")

        # Rule 2: Low Knowledge Base Confidence
        if max_similarity < 0.35:
            should_escalate = True
            reasons.append(f"Low retrieval confidence score ({max_similarity:.2f} < 0.35) — issue has no historical resolution precedent.")

        # Rule 3: High Customer Escalation Keywords / Frustration
        frustration_words = ["lawsuit", "legal", "supervisor", "manager", "terrible service", "scam", "frauds"]
        if any(w in query.lower() for w in frustration_words):
            should_escalate = True
            reasons.append("Customer sentiment indicates high frustration or legal/escalation risk.")

        # Rule 4: Refund requests
        if "refund" in query.lower() or "refund" in drafted_reply.lower():
            should_escalate = True
            reasons.append("Refund requests require human account review and cannot be auto-approved.")

        action = "ESCALATE_TO_HUMAN" if should_escalate else "AUTO_HANDLE"
        primary_reason = " | ".join(reasons) if reasons else "High retrieval confidence and standard resolution protocol."

        return {
            "action": action,
            "reason": primary_reason,
            "confidence_score": max_similarity,
            "escalation_recommended": should_escalate
        }

if __name__ == "__main__":
    engine = EscalationEngine()
    
    # Test case 1: Standard query with high similarity
    res1 = engine.evaluate(
        query="battery draining fast on 11.0.2",
        intent="software_bug_update",
        max_similarity=0.92,
        drafted_reply="Send us a DM..."
    )
    print("--- Test Case 1 (Standard Query) ---")
    print("Action:", res1["action"])
    print("Reason:", res1["reason"])

    # Test case 2: Security locked account query
    res2 = engine.evaluate(
        query="My Apple ID was hacked and password changed without my permission!",
        intent="apple_id_security",
        max_similarity=0.75,
        drafted_reply="Send us a DM..."
    )
    print("\n--- Test Case 2 (Security Query) ---")
    print("Action:", res2["action"])
    print("Reason:", res2["reason"])
