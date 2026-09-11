import sys
import os
import re
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.llm_client import LLMClient

# 6 Core Intents for @AppleSupport
INTENT_SCHEMA = {
    "software_bug_update": {
        "description": "iOS, macOS, watchOS update issues, battery drain, app crashes, lag, or software glitches.",
        "keywords": ["update", "ios", "macos", "battery", "drain", "bug", "crash", "glitch", "freeze", "slow", "version"]
    },
    "hardware_issue": {
        "description": "Physical device damage, screen replacement, trackpad, keyboard, charging port, speaker noise, power issues.",
        "keywords": ["screen", "display", "trackpad", "keyboard", "battery physical", "charger", "port", "speaker", "noise", "broken", "cracked", "macbook", "iphone"]
    },
    "icloud_storage_sync": {
        "description": "iCloud backup failures, storage full notifications, photo/file sync issues across devices.",
        "keywords": ["icloud", "storage", "backup", "sync", "photos", "full", "drive", "cloud"]
    },
    "apple_id_security": {
        "description": "Apple ID account locked, password reset, 2FA, verification code, unauthorized login.",
        "keywords": ["apple id", "password", "locked", "disabled", "security", "2fa", "verification", "code", "account", "login"]
    },
    "billing_subscription": {
        "description": "App Store purchases, recurring subscriptions, iTunes charges, refund requests.",
        "keywords": ["billing", "charge", "refund", "subscription", "app store", "itunes", "receipt", "money", "payment", "card"]
    },
    "general_inquiry_kb": {
        "description": "How-to queries, device specs, store appointments, warranty/AppleCare questions.",
        "keywords": ["how to", "applecare", "warranty", "appointment", "store", "trade in", "spec", "support", "help"]
    }
}

class KeywordBaselineClassifier:
    """Baseline 1: Rule-based Keyword Matching Classifier"""
    def __init__(self, schema: Dict[str, Any] = INTENT_SCHEMA):
        self.schema = schema

    def predict(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for intent, info in self.schema.items():
            count = sum(1 for kw in info["keywords"] if kw in text_lower)
            scores[intent] = count
            
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return "general_inquiry_kb"  # Fallback
        return best_intent

class TFIDFBaselineClassifier:
    """Baseline 2: TF-IDF + Logistic Regression Classifier"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500, class_weight='balanced')
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict(self, text: str) -> str:
        if not self.is_fitted:
            # Fallback to keyword if not trained yet
            return KeywordBaselineClassifier().predict(text)
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

class LLMIntentClassifier:
    """Main Classifier: LLM Zero-Shot / Few-Shot Intent Classifier using Groq/Grok"""
    def __init__(self, llm_client: LLMClient = None):
        self.client = llm_client or LLMClient()
        self.intents = list(INTENT_SCHEMA.keys())

    def predict(self, text: str) -> Dict[str, Any]:
        intents_desc = "\n".join([f"- {k}: {v['description']}" for k, v in INTENT_SCHEMA.items()])
        
        prompt = f"""You are an intent classification system for Apple Support (@AppleSupport).
Classify the following customer tweet into EXACTLY ONE of these categories:
{intents_desc}

Customer Tweet: "{text}"

Output your answer in EXACTLY this format:
INTENT: <one_of_the_keys_above>
REASON: <brief_one_sentence_reason>
"""
        raw_response = self.client.completion(prompt, use_reasoning_model=False, max_tokens=150)
        
        intent = "general_inquiry_kb"
        reason = "Default fallback classification."
        
        for line in raw_response.splitlines():
            if line.startswith("INTENT:"):
                cand = line.replace("INTENT:", "").strip().lower()
                for key in self.intents:
                    if key in cand:
                        intent = key
                        break
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
                
        return {"intent": intent, "reason": reason, "raw": raw_response}

if __name__ == "__main__":
    kw_clf = KeywordBaselineClassifier()
    tfidf_clf = TFIDFBaselineClassifier()
    llm_clf = LLMIntentClassifier()
    
    # Quick dummy fit for TF-IDF sanity check
    sample_texts = [
        "iPhone battery drains fast after iOS update",
        "Trackpad on my MacBook is broken and cracked",
        "iCloud storage is full photo sync failed",
        "Apple ID password reset locked account",
        "Refund for app store subscription billing",
        "How to make store appointment for AppleCare"
    ]
    sample_labels = [
        "software_bug_update", "hardware_issue", "icloud_storage_sync",
        "apple_id_security", "billing_subscription", "general_inquiry_kb"
    ]
    tfidf_clf.fit(sample_texts, sample_labels)
    
    test_query = "My iPhone battery is draining so fast after updating to iOS 17. Please help!"
    print("Test Query:", test_query)
    print("Keyword Baseline Prediction:", kw_clf.predict(test_query))
    print("TF-IDF Baseline Prediction:", tfidf_clf.predict(test_query))
    print("LLM Classifier Prediction:", llm_clf.predict(test_query))
