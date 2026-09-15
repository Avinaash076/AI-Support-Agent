import csv
import json
from pathlib import Path

def compute_agreement():
    human_path = Path("eval/human_verdicts.csv")
    judge_path = Path("eval/judge_results.json")

    if not human_path.exists() or not judge_path.exists():
        print("Missing human_verdicts.csv or judge_results.json")
        return

    judge_results = json.loads(judge_path.read_text(encoding="utf-8"))

    human_verdicts = {}
    with open(human_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map pass_with_issues to pass for binary agreement
            verdict = "pass" if "pass" in row["human_verdict"].lower() else "fail"
            human_verdicts[row["run_id"]] = verdict

    # Calculate agreement
    matches = 0
    total = 0

    # Confusion matrix
    # [True Pass, False Pass]
    # [False Fail, True Fail]
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for run_id, h_verdict in human_verdicts.items():
        if run_id in judge_results:
            total += 1
            j_verdict = judge_results[run_id]["verdict"]

            if h_verdict == j_verdict:
                matches += 1

            if h_verdict == "pass" and j_verdict == "pass": tp += 1
            elif h_verdict == "fail" and j_verdict == "pass": fp += 1
            elif h_verdict == "pass" and j_verdict == "fail": fn += 1
            elif h_verdict == "fail" and j_verdict == "fail": tn += 1

    if total == 0:
        print("No overlapping IDs found.")
        return

    percent_agreement = (matches / total) * 100

    # Cohen's Kappa
    p_o = matches / total
    p_yes = ((tp + fp) / total) * ((tp + fn) / total)
    p_no = ((fn + tn) / total) * ((fp + tn) / total)
    p_e = p_yes + p_no

    if p_e == 1:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    print(f"Agreement on {total} items: {percent_agreement:.1f}%")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print("\nConfusion Matrix (Human vs Judge):")
    print(f"           Judge Pass  Judge Fail")
    print(f"Human Pass     {tp:<10} {fn:<10}")
    print(f"Human Fail     {fp:<10} {tn:<10}")

if __name__ == "__main__":
    compute_agreement()
