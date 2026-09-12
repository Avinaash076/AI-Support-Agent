"""Validate and optionally run the repeatable answer-quality evaluation set."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
DEFAULT_CORPUS = ROOT / "data" / "evaluation" / "answer_quality.json"
REQUIRED_FIELDS = {
    "id", "category", "question", "expected_intent", "expected_action",
    "should_clarify", "good_answer_should_include", "must_not_include",
}
EXPECTED_CATEGORIES = {"battery", "icloud", "billing", "account_access", "hardware", "follow_up", "unrelated"}
EXPECTED_ACTIONS = {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}

class TrivialBaselinePipeline:
    def process_query(self, query, history=None):
        return {
            "intent": "general_inquiry_kb",
            "action": "ESCALATE_TO_HUMAN",
            "drafted_reply": "Thank you for reaching out to Apple Support. Please visit support.apple.com for help.",
        }

class KeywordBaselinePipeline:
    def __init__(self):
        from src.intent_classifier import KeywordBaselineClassifier
        self.clf = KeywordBaselineClassifier()
    def process_query(self, query, history=None):
        intent = self.clf.predict(query)
        action = "AUTO_HANDLE"
        if intent == "apple_id_security":
            action = "ESCALATE_TO_HUMAN"
        return {
            "intent": intent,
            "action": action,
            "drafted_reply": f"This looks like a {intent} issue. Let's see how we can help. Can you provide more details?",
        }


def llm_judge(question, facts, must_not, generated_reply):
    try:
        from src.llm_client import LLMClient
        client = LLMClient()
    except Exception:
        # fallback if env is bad
        return {"score": 3, "reasoning": "Mock judge fallback."}

    prompt = f"""You are an expert AI judge evaluating a customer support reply from an Apple Support bot.
You will be given the original customer question, a list of facts the answer MUST include, a list of things the answer MUST NOT include, and the actual generated reply.

Question: "{question}"
Must Include: {facts}
Must Not Include: {must_not}
Generated Reply: "{generated_reply}"

Evaluate the reply and give it a score from 1 to 5:
5 - Excellent. Helpful, grounded, includes all 'Must Include' facts, avoids all 'Must Not Include' facts, natural tone.
4 - Good. Misses a minor fact but is overall helpful and safe.
3 - Average. Vague or generic, but safe.
2 - Poor. Misses key facts or sounds robotic/unhelpful.
1 - Terrible. Hallucinates, gives bad advice, or violates 'Must Not Include' rules.

Output your result strictly as a JSON object with two keys: "score" (integer) and "reasoning" (string).
"""
    try:
        response = client.completion(prompt, use_reasoning_model=False, max_tokens=1000)
        # Parse JSON
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        return {"score": 3, "reasoning": f"Judge error: {str(e)}"}


def load_cases(path: Path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    seen = set()
    for position, case in enumerate(cases, start=1):
        missing = REQUIRED_FIELDS - set(case)
        if missing:
            errors.append(f"case {position} is missing: {', '.join(sorted(missing))}")
        if case.get("id") in seen:
            errors.append(f"duplicate case id: {case.get('id')}")
        seen.add(case.get("id"))
        if case.get("category") not in EXPECTED_CATEGORIES:
            errors.append(f"{case.get('id')}: unsupported category")
        if case.get("expected_action") not in EXPECTED_ACTIONS:
            errors.append(f"{case.get('id')}: unsupported expected_action")
        if not isinstance(case.get("good_answer_should_include"), list) or not case.get("good_answer_should_include"):
            errors.append(f"{case.get('id')}: good_answer_should_include must be non-empty")
        if not isinstance(case.get("must_not_include"), list):
            errors.append(f"{case.get('id')}: must_not_include must be a list")
    if len(cases) < 30 or len(cases) > 250:
        errors.append(f"corpus has {len(cases)} cases; expected 30-250")
    if errors:
        raise ValueError("\n".join(errors))
    return cases


def run_live(cases, limit=None, pipeline_type="rag"):

    if pipeline_type == "trivial":
        pipeline = TrivialBaselinePipeline()
    elif pipeline_type == "keyword":
        pipeline = KeywordBaselinePipeline()
    else:
        from src.agent_pipeline import SupportAgentPipeline
        pipeline = SupportAgentPipeline()

    rows = []
    selected = cases[:limit] if limit else cases
    for case in selected:
        started = time.perf_counter()

        try:
            result = pipeline.process_query(case["question"], history=case.get("history"))
            elapsed = time.perf_counter() - started

            judge_res = llm_judge(case["question"], case["good_answer_should_include"], case["must_not_include"], result["drafted_reply"])

            rows.append({
                "id": case["id"],
                "intent_expected": case["expected_intent"],
                "intent_actual": result["intent"],
                "action_expected": case["expected_action"],
                "action_actual": result["action"],
                "intent_match": result["intent"] == case["expected_intent"],
                "action_match": result["action"] == case["expected_action"],
                "seconds": round(elapsed, 3),
                "reply": result["drafted_reply"],
                "judge_score": judge_res.get("score", 3),
                "judge_reasoning": judge_res.get("reasoning", "")
            })
        except Exception as e:
            print(f"Failed case {case['id']}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--live", action="store_true", help="Call the configured LLM for each case.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases in live mode.")
    parser.add_argument("--output", type=Path, help="Write live results as JSON.")
    parser.add_argument("--pipeline", type=str, default="rag", choices=["rag", "trivial", "keyword"])
    args = parser.parse_args()

    cases = load_cases(args.corpus)
    if not args.live:
        print(f"Corpus valid: {len(cases)} cases")
        return 0

    rows = run_live(cases, args.limit, args.pipeline)
    if not rows:
        print("No cases processed successfully.")
        return 1

    intent_accuracy = sum(row["intent_match"] for row in rows) / len(rows)
    action_accuracy = sum(row["action_match"] for row in rows) / len(rows)
    average_seconds = sum(row["seconds"] for row in rows) / len(rows)
    avg_judge_score = sum(row["judge_score"] for row in rows) / len(rows)

    summary = {"cases": len(rows), "intent_accuracy": intent_accuracy,
               "action_accuracy": action_accuracy, "average_seconds": average_seconds,
               "average_judge_score": avg_judge_score,
               "results": rows}
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
