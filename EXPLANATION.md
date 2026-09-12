# Apple Support AI Agent: Project Brief and Explanation

## Project Brief
The Apple Support AI Agent is a local prototype application that simulates an automated customer support pipeline for `@AppleSupport` queries on Twitter. It does not actually interact with customers or Apple. Instead, it demonstrates a complete **Retrieval-Augmented Generation (RAG)** system with a fixed workflow:
1. **Classify Intent:** Determine the nature of the customer's problem.
2. **Retrieve Context:** Find similar historical problems and their resolutions from a database of 50,000 past interactions.
3. **Draft Reply:** Use a Large Language Model (LLM) to write a grounded, helpful response based on the retrieved historical context.
4. **Triage/Escalation:** Apply business logic to decide if the AI can safely handle the request (`AUTO_HANDLE`) or if it needs to be routed to a human agent (`ESCALATE_TO_HUMAN`).

---

## How Each Component Works (Line-by-Line Logic)

### 1. `src/agent_pipeline.py` (The Orchestrator)
This file connects all the individual components into a single workflow.
* **Initialization (`__init__`)**: Creates instances of the `LLMClient`, `LLMIntentClassifier`, `GroundedReplyGenerator`, and `EscalationEngine`.
* **`process_query(query, history)`**:
  * Formats the user's `query` and includes any recent conversation `history` to provide context for short follow-ups.
  * **Step 1:** Calls `intent_clf.predict()` to classify what the user wants (e.g., `software_bug_update`, `apple_id_security`).
  * **Step 2:** Calls `reply_gen.generate()` to retrieve similar historical tweets and use the LLM to draft a reply based on them. It also gets back the `top_similarity` score from the retrieval step.
  * **Step 3:** Calls `escalation_eng.evaluate()` to decide if a human needs to step in, using the intent, the similarity score, the query, and the drafted reply.
  * Returns a final dictionary containing all the results.

### 2. `src/intent_classifier.py` (The Classifier)
This file is responsible for understanding what the user is asking about.
* It defines a list of strict intents (like `hardware_repair`, `billing_subscriptions`, `apple_id_security`).
* **`predict(query)`**: It constructs a prompt asking the LLM to analyze the user's query and pick the single best matching intent from the allowed list. It enforces that the LLM output strictly matches a valid intent name and provides a brief reason for the classification.

### 3. `src/reply_generator.py` (The RAG Engine)
This is where the actual "Retrieval" and "Generation" happen.
* **`_build_index()`**: Reads the historical data (`applesupport_pairs.csv`). It uses `TfidfVectorizer` to learn the vocabulary of the historical questions and converts them into numerical vectors (`tfidf_matrix`). *Note: TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection. It is keyword-based, not semantic.*
* **`retrieve(query)`**: Converts the incoming user query into a TF-IDF vector and uses `cosine_similarity` to find the historical questions that overlap the most with the user's keywords. It returns the top `k` historical responses.
* **`generate(query, intent, history)`**:
  * Retrieves the top 5 historical responses.
  * Cleans the historical responses (removes raw URLs and "@" handles).
  * Constructs a prompt for the LLM that includes the user's query and the cleaned historical context. It strictly instructs the LLM to *only* use facts from the provided context and to ask clarifying questions if the context doesn't provide a complete fix.
  * Calls the LLM to generate the final `drafted_reply`.

### 4. `src/escalation_engine.py` (The Business Logic)
This component applies safety guardrails and routing rules without needing an LLM.
* **`evaluate(...)`**: Contains hardcoded python `if` statements (Rules).
  * **Rule 1 (Security):** If the intent is `apple_id_security` or contains words like "hacked" or "password", it instantly escalates to a human (`should_escalate = True`).
  * **Rule 2 (Low Confidence):** If the retrieval similarity score is less than 0.35 (meaning the system couldn't find good historical examples), it escalates to avoid hallucinating an answer.
  * **Rule 3 (Frustration):** If the query contains words like "lawsuit", "manager", or "scam", it escalates to handle the angry customer carefully.
  * **Rule 4 (Refunds):** Explicitly handles refund requests by checking if "refund" is in the query or drafted reply, routing them to human account review.

---

## Improving the Model Through Training

Currently, your project relies on **prompt engineering** and an off-the-shelf model (`openai/gpt-oss-20b`), while using **TF-IDF (keyword matching)** for retrieval. Here is how you can dramatically improve the answers through training and fine-tuning:

### 1. Upgrade Retrieval: Train a Dense Embedding Model
TF-IDF only matches exact words (e.g., "battery" matches "battery", but not "power").
* **How to improve:** Train or fine-tune a bi-encoder model (like `sentence-transformers`). You can train it using a "Contrastive Loss" or "Multiple Negatives Ranking Loss" on your pairs of similar questions.
* **Result:** The system will understand *semantic* similarity. A query like "My phone dies fast" will successfully retrieve historical answers for "battery drains quickly."

### 2. Add a Trained Reranker (Cross-Encoder)
Retrieving top documents fast can sometimes yield slightly irrelevant results.
* **How to improve:** Train a Cross-Encoder model that takes `[Customer Query, Historical Answer]` as input and outputs a relevance score between 0 and 1.
* **Result:** You retrieve top 50 examples quickly with embeddings, then the Cross-Encoder re-scores and filters them down to the top 3 most perfectly applicable examples to feed to your LLM generator.

### 3. Supervised Fine-Tuning (SFT) of the Generator
Instead of heavily prompting a generic LLM to act like Apple Support, you can train a smaller, custom model to naturally speak in the correct tone and format.
* **How to improve:** Take thousands of perfect `(Customer Query + Context -> Apple Support Ideal Reply)` examples. Use Parameter-Efficient Fine-Tuning (PEFT) techniques like **LoRA** (Low-Rank Adaptation) on an open-weights model like Llama 3.
* **Result:** The model learns the *style* (concise, polite, asking for iOS version), the *format* (how to correctly format Apple Support links), and the strict *boundaries* (knowing when to say "Send us a DM" instead of making up a policy).

### 4. Reinforcement Learning from Human Feedback (RLHF)
If your model still sometimes promises refunds or gives bad advice:
* **How to improve:** Have humans rate the model's drafted replies (e.g., Answer A is better than Answer B). Train a Reward Model on these ratings. Then use Proximal Policy Optimization (PPO) or Direct Preference Optimization (DPO) to update the generator model.
* **Result:** The model is mathematically penalized for generating confident but wrong answers, or for hallucinating unverified links, aligning it perfectly with Apple's strict support guidelines.
