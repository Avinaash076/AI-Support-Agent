import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import LLMClient

class HistoricalKnowledgeIndex:
    """TF-IDF Retrieval Index over historical Apple Support resolution pairs"""
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join("data", "processed", "applesupport_pairs.csv")
            
        print(f"Loading historical knowledge index from {data_path}...")
        self.df = pd.read_csv(data_path).dropna(subset=["customer_query", "brand_response_text"])
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        
        print("Indexing customer queries for retrieval...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["customer_query"].values)
        print(f"Index built with {len(self.df)} historical support interactions.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            row = self.df.iloc[idx]
            results.append({
                "similarity_score": score,
                "historical_query": row["customer_query"],
                "historical_response": row["brand_response_text"]
            })
        return results

class GroundedReplyGenerator:
    """RAG-based Grounded Reply Generator for Apple Support"""
    def __init__(self, index: HistoricalKnowledgeIndex = None, llm_client: LLMClient = None):
        self.index = index or HistoricalKnowledgeIndex()
        self.llm = llm_client or LLMClient()

    def generate(self, query: str, intent: str = "general_inquiry_kb", top_k: int = 3) -> Dict[str, Any]:
        # Step 1: Retrieve similar historical context
        retrieved_docs = self.index.retrieve(query, top_k=top_k)
        
        # Format context for prompt
        context_str = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context_str += f"Example {i} (Similarity: {doc['similarity_score']:.2f}):\n"
            context_str += f"  Past Customer: {doc['historical_query']}\n"
            context_str += f"  Apple Support Reply: {doc['historical_response']}\n\n"

        # Step 2: Formulate Grounded LLM Prompt
        prompt = f"""You are an official AI Customer Support Agent for Apple Support (@AppleSupport).
Your task is to draft a response to a customer tweet.

Customer Query: "{query}"
Classified Intent: {intent}

GROUNDING CONTEXT (Historical Apple Support Resolutions for similar issues):
{context_str}

STRICT GROUNDING RULES:
1. Base your resolution steps and tone ONLY on how Apple Support historically resolved similar issues above.
2. Keep the tone helpful, empathetic, professional, and aligned with Apple Support.
3. Keep the reply concise (under 280 characters suitable for Twitter).
4. If historical replies suggest taking the conversation to DM or checking a specific support link/setting, incorporate that appropriately.
5. Do NOT invent fake warranty policies or unverified URLs.

Draft Reply:
"""
        
        drafted_reply = self.llm.completion(prompt, use_reasoning_model=False, max_tokens=200)
        
        return {
            "query": query,
            "intent": intent,
            "drafted_reply": drafted_reply,
            "retrieved_context": retrieved_docs
        }

if __name__ == "__main__":
    generator = GroundedReplyGenerator()
    
    test_query = "please release some update my iphone battery is draining so fast even 1.5hrs battery back up seems impossible on 11.0.2,!"
    intent = "software_bug_update"
    
    print("\n--- Running Grounded RAG Generator ---")
    result = generator.generate(test_query, intent=intent)
    
    print("\nCustomer Query:", result["query"])
    print("Classified Intent:", result["intent"])
    print("\nTop Retrieved Historical Resolution:")
    top_doc = result["retrieved_context"][0]
    print(f"Score: {top_doc['similarity_score']:.3f}")
    print("Past Query:", top_doc['historical_query'])
    print("Past Apple Reply:", top_doc['historical_response'])
    print("\n--- Generated Grounded Reply ---")
    print(result["drafted_reply"])
