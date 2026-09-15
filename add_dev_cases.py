import json
from pathlib import Path

corpus_path = Path("eval/answer_quality.json")
cases = json.loads(corpus_path.read_text(encoding="utf-8"))

dev_cases = []
for i in range(1, 19):
    dev_cases.append({
        "id": f"dev-{i:03d}",
        "category": "battery", # arbitrary for this test
        "question": f"Dev test query {i}",
        "expected_intent": "software_bug_update",
        "expected_action": "AUTO_HANDLE",
        "should_clarify": True,
        "good_answer_should_include": ["test"],
        "must_not_include": ["test"]
    })

cases.extend(dev_cases)
corpus_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print(f"Added 18 dev cases. Total is now {len(cases)}.")
