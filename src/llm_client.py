import sys
import os
import re
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

class LLMClient:
    def __init__(self, provider: str = None):
        """
        Supports both Groq (gsk_...) and xAI Grok (xai-...) API keys.
        Auto-detects based on environment variables or parameters.
        """
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        
        if provider == "grok" or (self.grok_key and self.grok_key.startswith("xai-")):
            self.provider = "grok"
            self.api_key = self.grok_key
        else:
            self.provider = "groq"
            self.api_key = self.groq_key
            
        if self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.fast_model = "qwen/qwen3.6-27b"
            self.reasoning_model = "qwen/qwen3.6-27b"
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
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2
        )
        content = response.choices[0].message.content or ""
        
        # Robustly handle thinking tags (e.g. from Qwen models)
        if "<think>" in content:
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            else:
                content = content.replace("<think>", "").strip()
                # If cut off inside think block before </think>, extract response if available
                if "\n\n" in content:
                    parts = content.split("\n\n")
                    if len(parts) > 1 and not parts[-1].strip().startswith(("1.", "2.", "Here", "*")):
                        content = parts[-1].strip()
        return content

if __name__ == "__main__":
    client = LLMClient()
    print(f"Initialized LLMClient (Provider: {client.provider}, Model: {client.fast_model})")
    res = client.completion("Hello! Confirm LLM pipeline is ready.")
    print("Test Output:", res)
