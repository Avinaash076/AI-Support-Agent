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
    if len(cases) < 30 or len(cases) > 50:
        errors.append(f"corpus has {len(cases)} cases; expected 30-50")
    missing_categories = EXPECTED_CATEGORIES - {case.get("category") for case in cases}
    if missing_categories:
        errors.append(f"missing categories: {', '.join(sorted(missing_categories))}")
    if errors:
        raise ValueError("\n".join(errors))
    return cases


def run_live(cases, limit=None):
    from src.agent_pipeline import SupportAgentPipeline

    pipeline = SupportAgentPipeline()
    rows = []
    selected = cases[:limit] if limit else cases
    for case in selected:
        started = time.perf_counter()
        result = pipeline.process_query(case["question"], history=case.get("history"))
        elapsed = time.perf_counter() - started
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
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--live", action="store_true", help="Call the configured LLM for each case.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases in live mode.")
    parser.add_argument("--output", type=Path, help="Write live results as JSON.")
    args = parser.parse_args()

    cases = load_cases(args.corpus)
    counts = {category: sum(case["category"] == category for case in cases) for category in sorted(EXPECTED_CATEGORIES)}
    print(f"Corpus valid: {len(cases)} cases")
    print("Categories: " + ", ".join(f"{category}={count}" for category, count in counts.items()))
    if not args.live:
        return 0

    rows = run_live(cases, args.limit)
    intent_accuracy = sum(row["intent_match"] for row in rows) / len(rows)
    action_accuracy = sum(row["action_match"] for row in rows) / len(rows)
    average_seconds = sum(row["seconds"] for row in rows) / len(rows)
    summary = {"cases": len(rows), "intent_accuracy": intent_accuracy,
               "action_accuracy": action_accuracy, "average_seconds": average_seconds,
               "results": rows}
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())