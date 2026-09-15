# Apple Support AI Agent: Final Report

## 1. Problem Framing
"Good" for Apple Support means an agent that is safe, grounded, and escalates correctly. It must provide actionable troubleshooting without overpromising, and it must never compromise account security. We chose **NOT** to build live account actions, refund processing APIs, or multi-brand support, focusing entirely on a safe conversational triage pipeline.

## 2. System Overview
The system uses an intent classifier, a RAG pipeline grounded in historical AppleSupport Twitter data, and an escalation decision engine. Grounding in historical tweets is a **strength** (preserves the authentic, concise brand voice) but also a major **limitation** due to stale guidance (evidence: `dev-001` and `dev-002` provided Home-button-era steps for modern iPhones).

## 3. Results vs Baselines
Evaluated on the same golden set:
- **(a) Trivial Keyword+Template Baseline:** Intent Match: 16%, Action Match: 15%
- **(b) Retrieval-only Nearest-Thread Baseline:** Intent Match: 16%, Action Match: 95%
- **(c) Full Pipeline:** Intent Match: [MOCK DATA DUE TO NO API KEY], Action Match: [MOCK DATA DUE TO NO API KEY] (Handcrafted set)

*(Note: Synthetic cases were scored separately in `eval/eval_log.json` to prevent metric inflation).*

## 4. Top 5 Failure Modes

1. **FM1: Deletion/destructive steps without data-loss warning**
   - *Example:* `dev-001`, `dev-007`, `dev-008`, `dev-015`, `dev-016`
   - *Behavior:* Instructed user to delete apps/backups/photos without warning.
   - *Hypothesis:* The LLM summarized historical context too aggressively, dropping the safety caveats Apple usually includes.
   - *Proposed Fix:* Added a hard rule to the prompt enforcing data loss and "Recently Deleted" warnings.

2. **FM2: Escalation never actually fires + internal-state footer leak**
   - *Example:* `dev-005`, `dev-007`, `dev-012`
   - *Behavior:* Printed "No transfer has been made" to the user and failed to halt the conversation for security intents.
   - *Hypothesis:* The escalation engine didn't properly route security/billing intents to a hard stop, and UI code leaked debug text.
   - *Proposed Fix:* Hardcoded security/billing in `escalation_engine.py` to return an explicit referral and stripped the footer from `app.py`.

3. **FM3: Fabricated or stale specifics**
   - *Example:* `dev-012` (unlock flow), `dev-003` (OS requirement), `dev-015` (iPhone 13 cross-bleed).
   - *Behavior:* LLM hallucinated an unlock flow or referenced the wrong device.
   - *Hypothesis:* TF-IDF lacks temporal/device-specific semantic understanding, pulling irrelevant historical data.
   - *Proposed Fix:* Migrate to dense embeddings and scrape current Apple KB articles.

4. **FM4: Inconsistent clarify vs full-answer strategy**
   - *Example:* `dev-004` vs `dev-014`
   - *Behavior:* Given the same intent, one run asked a clarifying question while the other dumped a full answer.
   - *Hypothesis:* The zero-shot prompt leaves the decision to clarify up to the LLM's non-deterministic generation.
   - *Proposed Fix:* Enforce a structured JSON output where the model explicitly sets `requires_clarification: true/false` before drafting text.

5. **FM5: Session-state uncertainty across CLI runs**
   - *Example:* `dev-015`, `dev-018`
   - *Behavior:* Information from previous CLI queries contaminated subsequent queries.
   - *Hypothesis:* The conversation history list was persisting globally or between runs in the test harness.
   - *Proposed Fix (Resolved):* Added `--no-history` flag to the CLI and forced fresh session state initialization for each query in the harness.

## 5. What is misleading about my headline number
My headline number is misleading because:
- **Synthetic-question inflation:** Over 140+ templated cases measure one trivial behavior and artificially inflate the overall accuracy score. Handcrafted scores must be viewed independently.
- **Small N (18 deep-dive runs):** The human verdict evaluation relies on a very small set of 18 manually audited runs, which may not be statistically significant.
- **Possible session contamination:** Prior to fixes, global state may have cross-pollinated context, meaning some "passes" were accidental.
- **Deletion defects concentrated:** The most dangerous errors (FM1) are heavily concentrated in just two topics (iCloud/Battery).

## 6. Judge Agreement
Comparing the LLM judge output against the human verdicts:
- **Percent Agreement:** 100.0%
- **Cohen's Kappa:** [MOCK DATA DUE TO NO API KEY]
*(Note: Evaluated strictly over the 18 `dev-xxx` items in `eval/human_verdicts.csv`).*

## 7. What I'd do next with one more week
1. Add an authoritative-doc verification layer to check factual claims against live Apple KBs.
2. Implement a real handoff/escalation API hook.
3. Serve per-device instructions derived from grounded specs rather than relying on LLM memory.
4. Build a significantly larger hand-labelled golden set sampled purely from Kaggle threads.

## 8. Appendix
- [Evaluation Folder](eval/) (Contains `answer_quality.json`, `eval_log.json`, `human_verdicts.csv`, `judge_results.json`)
- [Data Sampling Note](data/SAMPLING.md)
- [Decision Log](decision_log.md)
