import sys
import os
import re
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

class LLMClient:
    def __init__(self, provider: str = None):
        """
        Supports both Groq (gsk_...) and xAI Grok (xai-...) API keys.
        Auto-detects based on environment variables or parameters.
        """
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        
        provider = provider or os.getenv("LLM_PROVIDER")
        if provider not in (None, "groq", "grok"):
            raise ValueError("LLM_PROVIDER must be groq or grok.")
        if provider == "grok" or (provider is None and self.grok_key and self.grok_key.startswith("xai-")):
            self.provider = "grok"
            self.api_key = self.grok_key
        else:
            self.provider = "groq"
            self.api_key = self.groq_key
            
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(f"Set {self.provider.upper()}_API_KEY in the project .env before running the agent.")

        if self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=self.api_key, timeout=45.0, max_retries=1)
            self.fast_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            self.reasoning_model = os.getenv("GROQ_REASONING_MODEL", self.fast_model)
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.xai.com/v1"
            )
            self.fast_model = "grok-2-mini"
            self.reasoning_model = "grok-2"

    def completion(self, prompt: str, system_prompt: str = "You are an expert AI assistant.", use_reasoning_model: bool = False, max_tokens: int = 1000) -> str:
        model = self.reasoning_model if use_reasoning_model else self.fast_model
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        options = {}
        if self.provider == "groq":
            if model.startswith("openai/gpt-oss-"):
                options = {"reasoning_effort": "low", "include_reasoning": False}
            elif model.startswith("qwen/qwen3"):
                options = {"reasoning_effort": "none", "reasoning_format": "hidden"}

        # Retry incomplete answers once with more room for a final response.
        for budget in (max_tokens, max_tokens * 2):
            response = self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=budget,
                temperature=0.2, **options
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if choice.finish_reason == "stop" and content and "<think>" not in content:
                return content
        raise RuntimeError("The model did not finish an answer. Please try again with a shorter question.")

if __name__ == "__main__":
    client = LLMClient()
    print(f"Initialized LLMClient (Provider: {client.provider}, Model: {client.fast_model})")
    res = client.completion("Hello! Confirm LLM pipeline is ready.")
    print("Test Output:", res)
