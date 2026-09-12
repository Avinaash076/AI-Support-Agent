# Apple Support AI Agent: Final Report

## Problem Framing
Our goal was to build an AI support agent capable of classifying, answering, and triaging incoming customer tweets directed at `@AppleSupport`.
"Good" for this brand means:
1. **Safety & Security:** Never ask for passwords and immediately escalate security/account lockout issues.
2. **Grounded Help:** Provide actionable, step-by-step troubleshooting that Apple has previously provided, without inventing non-existent features or phantom URLs.
3. **Appropriate Tone:** Maintain a calm, helpful, and concise presence, mirroring human support staff.

**What we chose NOT to build:**
- We chose not to build an automated action executor (e.g., an agent that actively resets passwords or processes refunds via API).
- We chose not to support multi-turn complex reasoning loops, keeping it strictly to a single RAG retrieval step to ensure fast response times on social media.

## Results vs. Baselines
We evaluated our RAG pipeline against two baselines over our golden dataset of 184 examples:
1. **Trivial Baseline:** Always classifies as `general_inquiry_kb` and routes to `ESCALATE_TO_HUMAN` with a generic "Please visit support.apple.com" message.
2. **Keyword Baseline:** Uses a simple keyword-matching heuristic for intent classification, rules for escalation, and static templates for replies.
3. **RAG Pipeline (Our approach):** LLM for intent classification, TF-IDF over 50,000 tweets for context retrieval, and LLM drafting.

*Live Evaluation Results (Subset sample):*
- **Trivial Baseline:**
  - Intent Accuracy: ~0%
  - Action Accuracy: ~0%
  - LLM Judge Score (1-5): ~1.5 (Very unhelpful)
- **Keyword Baseline:**
  - Intent Accuracy: ~50%
  - Action Accuracy: ~85%
  - LLM Judge Score (1-5): ~2.0 (Too robotic/generic)
- **RAG Pipeline:**
  - Intent Accuracy: ~90%+
  - Action Accuracy: ~95%+
  - LLM Judge Score (1-5): ~4.2 (Highly grounded and specific)

## Failure Analysis
Here are the top 5 failure modes observed during development and evaluation, along with hypotheses for why they occur:

1. **Hallucinated iOS versions.**
   - *Example:* The AI suggests "Update to iOS 17.4" when the user's issue was with a generic battery problem, but the context tweets were pulled from an era when iOS 16 was the latest.
   - *Hypothesis:* TF-IDF doesn't understand temporal context. It just matches keywords, pulling in outdated advice.
2. **Over-escalation on "Locked" screens.**
   - *Example:* User says "My screen is locked in landscape mode" and the AI escalates for an `apple_id_security` breach.
   - *Hypothesis:* The keyword "locked" strongly biases the security escalation rule, missing semantic meaning.
3. **Poor handling of highly specific device hardware.**
   - *Example:* User asks about a 2012 MacBook Pro trackpad issue. The AI provides generic iPhone trackpad (non-existent) advice.
   - *Hypothesis:* Lack of dense semantic embeddings causes the retriever to miss the nuanced difference between devices.
4. **Vague follow-ups.**
   - *Example:* User says "I tried that". The AI responds "Can you explain what you tried?" instead of using context.
   - *Hypothesis:* The LLM prompt isn't forcefully constrained enough to utilize the injected conversation history for context resolution.
5. **Inventing Support URLs.**
   - *Example:* The AI outputs `support.apple.com/kb/HT1234567` (a dead link).
   - *Hypothesis:* The LLM is acting generatively to "fill in the blanks" when a historical tweet tells the user "Check out this link: [URL]".

## What is misleading about my headline number?
If I say "Our AI achieves 95% action accuracy and a 4.2 LLM Judge score," it is highly misleading for several reasons:
1. **The dataset is synthesized and heavily skewed.** A large portion of our 184-case evaluation set was synthetically generated. Real Twitter data is significantly noisier, filled with slang, misspellings, and sarcasm that our test set lacks.
2. **TF-IDF overfits to our test keywords.** Because our synthetic data uses standard terminology ("battery drain", "icloud full"), the TF-IDF retriever performs artificially well. In the wild, semantic search is necessary.
3. **The LLM Judge is inherently biased.** The LLM Judge (Grok/Groq) evaluating the LLM Generator often exhibits self-preference bias and may score highly on text that sounds fluent, even if a human support agent would find the advice slightly off-brand.

## What I'd do next with one more week
1. **Replace TF-IDF with Dense Embeddings (Sentence-Transformers):** Move to semantic search (e.g., Pinecone/Chroma) to understand the *meaning* of questions rather than exact word overlap.
2. **Implement a Cross-Encoder Reranker:** To drastically improve the quality of the top 3 examples fed into the prompt.
3. **Scrape Official Apple KB Articles:** Historical tweets are outdated. I would index the actual Apple Support documentation to ground the answers in current, factual steps.
4. **Fine-tune a smaller model (LoRA):** Instead of using zero-shot prompting on a large model, I would fine-tune a smaller model (like Llama 3 8B) on the perfect golden responses to lock in the exact tone and formatting constraints.

## Decision Log
1. **Chose TF-IDF over Dense Embeddings initially:** Faster to prototype locally without requiring GPU resources for embedding models or setting up a vector database.
2. **Hardcoded Escalation Engine:** Used basic Python logic (Regex/Keywords) for triage rather than asking the LLM, to guarantee 100% safety on critical issues like passwords or legal threats.
3. **Included "Must Not Include" in eval:** Added negative constraints to the LLM judge rubric because avoiding bad advice is more important for Apple Support than giving perfectly detailed advice.
4. **Synthetic Data Augmentation:** Expanded the golden set from 40 to 184 using programmatic templates to generate diverse scenarios quickly while controlling the exact distribution of intents.
5. **No actual API dispatch:** Did not mock up fake API endpoints (like "refund API") because the assignment focused on the NLP pipeline and evaluation, not backend integration.
6. **Stripping raw URLs in prompt:** Explicitly clean URLs from historical data before feeding to the LLM to reduce the chance of the LLM hallucinating fake `apple.com/support/...` links.
7. **Streamlit UI:** Chose Streamlit over a React frontend for rapid prototyping of the chat interface, allowing immediate qualitative testing.
8. **Groq/Grok as primary provider:** Leveraged high-speed API endpoints to ensure the evaluation loop runs quickly during development.
9. **Separate Intent vs RAG steps:** Chose to classify intent first, then generate a reply, rather than asking the LLM to do both in one pass. This allows specific business logic (escalation) based on intent before text generation.
10. **Cached Vectorizer:** Ensured the TF-IDF matrix is built once and cached in memory, preventing a 50,000-document rebuild on every chat turn.
