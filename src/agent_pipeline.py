import sys
import os
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_classifier import LLMIntentClassifier, KeywordBaselineClassifier
from src.reply_generator import GroundedReplyGenerator
from src.escalation_engine import EscalationEngine

class SupportAgentPipeline:
    """Full End-to-End AI Support Agent Pipeline for Apple Support"""
    def __init__(self):
        print("Initializing AI Support Agent Pipeline for @AppleSupport...")
        self.intent_clf = LLMIntentClassifier()
        self.kw_clf = KeywordBaselineClassifier()
        self.reply_gen = GroundedReplyGenerator()
        self.escalation_eng = EscalationEngine()
        print("Pipeline initialized successfully!\n")

    def process_query(self, query: str) -> Dict[str, Any]:
        # Step 1: Intent Classification
        intent_res = self.intent_clf.predict(query)
        intent = intent_res["intent"]
        intent_reason = intent_res["reason"]

        # Step 2: Grounded Reply Generation via RAG
        rag_res = self.reply_gen.generate(query, intent=intent)
        drafted_reply = rag_res["drafted_reply"]
        retrieved_docs = rag_res["retrieved_context"]
        top_similarity = retrieved_docs[0]["similarity_score"] if retrieved_docs else 0.0

        # Step 3: Triage & Escalation Decision
        triage_res = self.escalation_eng.evaluate(
            query=query,
            intent=intent,
            max_similarity=top_similarity,
            drafted_reply=drafted_reply
        )

        return {
            "query": query,
            "intent": intent,
            "intent_reason": intent_reason,
            "drafted_reply": drafted_reply,
            "top_similarity": top_similarity,
            "historical_matches": retrieved_docs,
            "action": triage_res["action"],
            "escalation_reason": triage_res["reason"]
        }

def run_interactive_cli():
    pipeline = SupportAgentPipeline()
    print("=" * 70)
    print(" 🍎 Apple Support AI Agent Interactive CLI")
    print(" Type any customer tweet to test the agent (or 'exit' to quit).")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\nCustomer Tweet > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Apple Support Agent CLI.")
                break
                
            res = pipeline.process_query(user_input)
            
            print("\n" + "-" * 50)
            print(f"🎯 INTENT CLASSIFIED : {res['intent'].upper()}")
            print(f"   Reason            : {res['intent_reason']}")
            print(f"📈 KB RETRIEVAL SCORE : {res['top_similarity']:.3f}")
            print(f"🚨 ACTION DECISION   : {res['action']}")
            print(f"   Reason            : {res['escalation_reason']}")
            print("-" * 50)
            print("💬 DRAFTED GROUNDED REPLY:")
            print(f"   \"{res['drafted_reply']}\"")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        pipeline = SupportAgentPipeline()
        res = pipeline.process_query(query_text)
        print("\n--- Pipeline Result ---")
        print("Query:", res["query"])
        print("Intent:", res["intent"])
        print("Action:", res["action"])
        print("Reply:", res["drafted_reply"])
    else:
        # Run single test query output
        pipeline = SupportAgentPipeline()
        test_q = "please release some update my iphone battery is draining so fast even 1.5hrs battery back up seems impossible on 11.0.2,!"
        res = pipeline.process_query(test_q)
        print("\n=== SAMPLE QUERY EXECUTION ===")
        print(f"Customer Tweet  : {res['query']}")
        print(f"Intent          : {res['intent']}")
        print(f"Action Decision : {res['action']} ({res['escalation_reason']})")
        print(f"Drafted Reply   : {res['drafted_reply']}")
