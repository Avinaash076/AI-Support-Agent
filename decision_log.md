# Decision Log

1. **Grounding in historical tweets**: Using real `@AppleSupport` tweets provides an authentic brand voice but introduces a significant staleness tradeoff (e.g., suggesting Home-button resets for iPhone 16s).
2. **Synthetic eval cases separated**: Kept synthetic evaluation cases but scored them separately from the hand-labelled Kaggle subset to avoid masking real-world performance with inflated numbers on templated data.
3. **Intent set derived from data**: The 6 core intents were chosen based directly on observing the highest-frequency topics in the Kaggle support thread data rather than arbitrarily guessing what customers ask.
4. **Escalation semantics chosen**: Security, billing disputes, and low-confidence retrievals bypass LLM generation entirely and escalate, prioritizing safety over automated resolution.
5. **Deletion-safety rule added**: After identifying a major failure mode (FM1) where the LLM blindly suggested deleting backups/photos, an explicit constraint was added to the prompt to enforce data loss and 30-day "Recently Deleted" warnings.
6. **Golden-set sampling method**: Used a random uniform sample from the filtered `applesupport_pairs.csv` while ensuring representation across all 6 core intents to build `answer_quality.json`.
7. **Judge rubric design**: The LLM judge strictly evaluates against binary `must_include` and `must_not_include` facts, reducing the inherent bias of LLMs scoring their own fluency.
8. **Baselines chosen**: Selected a keyword/template baseline to measure basic routing logic, and a raw retrieval baseline to isolate the benefit of the LLM generator summarizing the historical context.
9. **Session-state logging added**: Added `--no-history` and per-run transcripts to fix and diagnose cross-run contamination where the model hallucinated contexts from previous queries.
10. **Removed internal state footer leak**: Stripped the internal debug flag "No transfer has been made" from the user-facing output to maintain a professional, production-ready tone.
11. **Avoided complex agent actions**: Explicitly chose not to implement mock API tools (like issuing refunds or password resets) because the failure risk in customer support is too high without human verification.
