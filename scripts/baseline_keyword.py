import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.intent_classifier import KeywordBaselineClassifier

def run_keyword_baseline():
    corpus_path = Path("eval/answer_quality.json")
    cases = json.loads(corpus_path.read_text(encoding="utf-8"))

    clf = KeywordBaselineClassifier()
    results = []

    for case in cases:
        intent = clf.predict(case["question"])
        action = "AUTO_HANDLE"
        if intent in ["apple_id_security", "billing_subscription"]:
            action = "ESCALATE_TO_HUMAN"

        reply = f"This looks like a {intent} issue. Let's see how we can help. Can you provide more details?"
        if action == "ESCALATE_TO_HUMAN":
            reply = "Thank you for reaching out to Apple Support. For security reasons, please contact Apple Support directly at support.apple.com."

        results.append({
            "id": case["id"],
            "intent_expected": case["expected_intent"],
            "intent_actual": intent,
            "action_expected": case["expected_action"],
            "action_actual": action,
            "reply": reply
        })

    out_path = Path("eval/baseline_keyword_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Keyword baseline evaluated {len(results)} cases. Saved to {out_path}.")

if __name__ == "__main__":
    run_keyword_baseline()
