# Apple Support Agent

Local prototype that classifies support queries, retrieves historical Twitter
support replies using TF-IDF, drafts a reply, and recommends human escalation.
It does not contact Apple or send replies to customers.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env  # Only for initial setup; keep existing credentials.
```

Set `GROQ_API_KEY` in `.env` to your own Groq key. No Apple API key is needed.
The default provider is Groq and the model is `openai/gpt-oss-20b`.
Override `GROQ_MODEL` to use another model available to your account.
Inception is not currently integrated.

## Restore the dataset

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/filter_brand_data.py AppleSupport
.\.venv\Scripts\python.exe scripts/print_apple_samples.py
```

The download script fetches `thoughtvector/customer-support-on-twitter` from
Kaggle. Filtering creates `data/processed/applesupport_pairs.csv` with up to
50,000 Apple query/reply pairs. The print script only displays the first ten
pairs; it does not download data. CSV files and `.env` are excluded from Git.

## Test the agent

### Chat interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit. The chat keeps history for the
browser session and sends up to six recent messages as context. Short follow-ups
also use recent user questions for retrieval. Use **New conversation** when
changing topics. The search index is cached instead of rebuilt for every message.
Human-support recommendations do not transfer or send the conversation to Apple.

`src/reply_generator.py` contains the grounding prompt. Historical tweets are
filtered for relevance and stripped of handles and URLs before prompting. The
assistant asks for missing information when the examples cannot support a fix.
`src/llm_client.py` configures Groq reasoning output and retries incomplete
responses once with twice the token allowance; persistent failures show a retry
message in the chat instead of exposing unfinished reasoning.

Run the offline regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_chat.py -v
```

### Repeatable answer evaluation

The checked-in benchmark at `data/evaluation/answer_quality.json` contains 40
questions across battery, iCloud, billing, account access, hardware, unrelated
requests, and follow-ups. Each case records the expected intent and action, the
facts or behavior a good answer should include, and claims it must avoid.

Validate the corpus without using an API key:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_answers.py
```

Run a live evaluation only when you intentionally want to spend provider
tokens. It records intent accuracy, escalation-action accuracy, and average
latency; the optional output file preserves each reply for manual review:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --live --output evaluation-results.json
```

Use `--limit 5` while developing a prompt or retrieval change. For every live
run, manually review whether the answer is supported by the retrieved examples,
answers the actual question, asks only useful missing-detail questions, avoids
invented policies or links, and stays within a reasonable response time and
token budget. Treat the benchmark as a regression set, not as a replacement for
reviewed Apple documentation. Historical tweets are dated examples; adding
reviewed documentation with source URLs and review dates is the next grounding
improvement.

### Command line

```powershell
.\.venv\Scripts\python.exe src/llm_client.py
.\.venv\Scripts\python.exe src/agent_pipeline.py "My iPhone battery drains quickly after an update."
```

The pipeline prints intent, escalation action, and a generated reply. It sends
the query and retrieved examples to the configured model provider. Historical
tweets are examples, not a current Apple knowledge base; review drafted replies
before using them with customers.
