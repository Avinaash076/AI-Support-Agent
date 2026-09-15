class EscalationEngine:
    def __init__(self, llm_client=None):
        self.client = llm_client

    def evaluate(self, query: str, intent: str, max_similarity: float, drafted_reply: str) -> dict:
        reason = ""
        should_escalate = False

        # Explicit referrals for security and billing per fix requirement
        if intent == "apple_id_security":
            should_escalate = True
            reason = "Security/Account issues require human verification."
        elif intent == "billing_subscription":
            should_escalate = True
            reason = "Billing/Refund requests require human verification."
        elif any(word in query.lower() for word in ["hacked", "stolen", "lawsuit", "sue"]):
            should_escalate = True
            reason = "High-risk or legal keywords detected."
        elif max_similarity < 0.35:
            should_escalate = True
            reason = "Retrieval confidence too low (< 0.35)."

        action = "ESCALATE_TO_HUMAN" if should_escalate else "AUTO_HANDLE"
        return {"action": action, "reason": reason}
