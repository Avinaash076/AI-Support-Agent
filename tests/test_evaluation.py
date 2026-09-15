import json
import unittest
from pathlib import Path

from scripts.evaluate_answers import load_cases


class EvaluationCorpusTests(unittest.TestCase):
    def test_answer_quality_corpus_is_valid_and_complete(self):
        path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "answer_quality.json"
        cases = load_cases(path)
        self.assertEqual(len(cases), 202)
        self.assertEqual({case["category"] for case in cases}, {
            "battery", "icloud", "billing", "account_access", "hardware",
            "follow_up", "unrelated",
        })

    def test_history_is_json_serializable(self):
        path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "answer_quality.json"
        cases = json.loads(path.read_text(encoding="utf-8"))
        for case in cases:
            json.dumps(case.get("history", []))


if __name__ == "__main__":
    unittest.main()