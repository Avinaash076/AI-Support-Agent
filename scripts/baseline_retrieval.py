import sys
import os
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# We will just mock the retrieval directly to bypass LLMClient init
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.intent_classifier import KeywordBaselineClassifier

def run_retrieval_baseline():
    corpus_path = Path("eval/answer_quality.json")
    cases = json.loads(corpus_path.read_text(encoding="utf-8"))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "applesupport_pairs.csv"
    if not csv_path.exists():
        df = pd.DataFrame([
            {"customer_query": "battery drain", "brand_response_text": "Update your iOS."},
            {"customer_query": "icloud full", "brand_response_text": "Buy more storage."},
            {"customer_query": "billing issue", "brand_response_text": "Check your receipt."}
        ])
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    queries = df["customer_query"].fillna("").tolist()
    replies = df["brand_response_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    matrix = vectorizer.fit_transform(queries)

    kw_clf = KeywordBaselineClassifier()
    results = []

    for case in cases:
        intent = kw_clf.predict(case["question"])

        q_vec = vectorizer.transform([case["question"]])
        sim = cosine_similarity(q_vec, matrix).flatten()
        best_idx = sim.argmax()
        best_score = float(sim[best_idx])

        action = "AUTO_HANDLE"
        if best_score < 0.35:
            action = "ESCALATE_TO_HUMAN"
        elif intent in ["apple_id_security", "billing_subscription"]:
            action = "ESCALATE_TO_HUMAN"

        if best_score > 0:
            reply = replies[best_idx]
        else:
            reply = "No similar historical cases found. Please visit support.apple.com."

        results.append({
            "id": case["id"],
            "intent_expected": case["expected_intent"],
            "intent_actual": intent,
            "action_expected": case["expected_action"],
            "action_actual": action,
            "reply": reply
        })

    out_path = Path("eval/baseline_retrieval_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Retrieval baseline evaluated {len(results)} cases. Saved to {out_path}.")

if __name__ == "__main__":
    run_retrieval_baseline()
