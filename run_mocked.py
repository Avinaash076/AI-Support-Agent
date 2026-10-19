import sys
import os
from unittest.mock import patch

os.environ["GROQ_API_KEY"] = "mock_key"

from src.agent_pipeline import SupportAgentPipeline
from src.llm_client import LLMClient

def mock_completion(self, prompt, *args, **kwargs):
    if "Classify the following customer tweet" in prompt:
        return "INTENT: software_bug_update\nREASON: The user is complaining about rapid battery drain on iOS 11.0.2."
    else:
        return "It sounds like you're experiencing unexpected battery drain on iOS 11.0.2. Based on similar reports, you might want to check which apps are using the most power in Settings > Battery. Sometimes a simple restart can also help resolve temporary software glitches. If the issue continues, please reach out to us at https://support.apple.com/ for further assistance."

with patch.object(LLMClient, 'completion', new=mock_completion):
    pipeline = SupportAgentPipeline()
    test_q = "please release some update my iphone battery is draining so fast even 1.5hrs battery back up seems impossible on 11.0.2,!"
    res = pipeline.process_query(test_q)
    print("\n=== SAMPLE QUERY EXECUTION ===")
    print(f"Customer Tweet  : {res['query']}")
    print(f"Intent          : {res['intent']}")
    print(f"Action Decision : {res['action']} ({res['escalation_reason']})")
    print(f"Drafted Reply   : {res['drafted_reply']}")
