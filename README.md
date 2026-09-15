# Apple Support Agent

Local prototype that classifies support queries, retrieves historical Twitter
support replies using TF-IDF, drafts a reply, and recommends human escalation.
It does not contact Apple or send replies to customers.

## Quickstart (Reproduction in < 15 minutes)

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Ensure you set your `GROQ_API_KEY` in the `.env` file. We use the Groq API and `openai/gpt-oss-20b` base LLM by default.

### 2. Dataset Setup
This project uses the Kaggle `thoughtvector/customer-support-on-twitter` dataset.
Run the subsampling script to build the local index:
```bash
python scripts/download_data.py
python scripts/filter_brand_data.py AppleSupport
```
*(See `data/SAMPLING.md` for details on how we filtered this dataset).*

### 3. Run the Agent UI
```bash
python -m streamlit run app.py
```
*(Use `--no-history` in CLI mode to ensure session hygiene across distinct runs).*

### 4. Run the Evaluation Harness
Run the baselines and the main RAG pipeline against the golden set:
```bash
python scripts/baseline_keyword.py
python scripts/baseline_retrieval.py
python scripts/run_eval.py
```
Outputs are written to `eval/baseline_keyword_results.json`, `eval/baseline_retrieval_results.json`, and `eval/eval_log.json`.

Run the LLM Judge and compute Human-Judge agreement:
```bash
python scripts/judge.py
python scripts/agreement.py
```

## Documentation
- **[Final Report](REPORT.md)**: Details on problem framing, baselines, failure modes, and what's misleading about the headline metrics.
- **[Decision Log](decision_log.md)**: 10-15 non-obvious engineering decisions made during development.
- **[Evaluation Data](eval/)**: Contains the `answer_quality.json` golden set, human verdicts, and outputs.

---
*Citations: Kaggle dataset by thoughtvector. Base LLM models provided via Groq.*
