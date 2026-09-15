import json
import sys
import subprocess
from pathlib import Path

# Since I do not have a working API key for LLMClient (GROQ_API_KEY is not provided),
# I cannot write an LLM judge that actually calls the LLM, nor can I actually generate real RAG replies.
# I am operating in a sandbox without an API key, so I have to mock the outputs to complete the assignment's structure.
# Let's fix the report to clarify this limitation.

report_path = Path("REPORT.md")
content = report_path.read_text()
content = content.replace("100%", "[MOCK DATA DUE TO NO API KEY]")
content = content.replace("1.000", "[MOCK DATA DUE TO NO API KEY]")
report_path.write_text(content)

print("Updated REPORT.md to explicitly state that the metrics are placeholders due to missing API keys.")
