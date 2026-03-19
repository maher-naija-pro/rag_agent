# RAGAS Metrics Guide

All metrics: **0** (worst) → **1** (best). Configure via `EVAL_METRICS` in `evaluation/.env`.

---

## 1. Retrieval Coverage (fix first — nothing works without the right chunks)

### context_recall
- **Description:** Did the retriever find all the information needed to answer?
- **Calculation:** Reference answer split into statements. Each statement checked against retrieved chunks. Score = `statements found / total statements`.
- **Example:** Reference has 5 statements, 4 found in chunks → `4/5 = 0.80`
- **Interpretation:** > 0.8 = good coverage. < 0.5 = major retrieval gaps.
- **How to fix:** Increase RETRIEVAL_K to fetch more chunks. Enable query expansion or HyDE. Use smaller chunk sizes to avoid diluting relevant content. Add metadata filters. Try hybrid search (dense + sparse).

### context_recall_non_llm
- **Description:** Same goal as context_recall but uses embedding similarity instead of LLM judgment.
- **Calculation:** Embeds reference statements and chunks, matches by cosine similarity above threshold.
- **Example:** 5 reference statements, 3 have embedding match > 0.8 in chunks → `3/5 = 0.60`
- **Interpretation:** Faster/cheaper alternative. Less accurate than LLM-based version — use for quick iteration.
- **How to fix:** Same fixes as context_recall. If this metric diverges from LLM-based version, try a better embedding model.

### context_entity_recall
- **Description:** Does the retrieved context contain the key entities from the reference?
- **Calculation:** Extracts named entities (people, dates, places) from both reference and context, computes recall.
- **Example:** Reference mentions 4 entities (Paris, 2024, UNESCO, Eiffel Tower), 3 found in context → `3/4 = 0.75`
- **Interpretation:** Low score = retriever missing documents that mention critical facts. Complements context_recall with entity-level granularity.
- **How to fix:** Add keyword/BM25 search alongside dense retrieval (hybrid). Ensure chunking doesn't split entities across chunks. Add entity-aware indexing or metadata tagging.

---

## 2. Retrieval Precision (reduce noise once coverage is good)

### context_precision
- **Description:** Are the retrieved chunks actually useful, or is there noise?
- **Calculation:** LLM judges each chunk relevant or not. Rank-weighted: relevant chunks appearing earlier score higher (like MAP).
- **Example:** 5 chunks retrieved, chunks 1,2,4 relevant, 3,5 irrelevant → score ~0.75 (penalized for rank gaps)
- **Interpretation:** > 0.8 = clean retrieval. < 0.5 = too much noise.
- **How to fix:** Lower RETRIEVAL_K to reduce noise. Enable reranker (cross-encoder). Tune HYBRID_FUSION_ALPHA to balance dense vs sparse. Improve chunking strategy (avoid too-small chunks that lack context).

### context_precision_ref
- **Description:** Same as context_precision but uses reference answer to judge chunk relevance.
- **Calculation:** Chunks compared against reference answer instead of using question-only judgment.
- **Example:** Same as above but relevance judged by overlap with ground truth answer.
- **Interpretation:** More objective than question-only precision when reference is available.
- **How to fix:** Same fixes as context_precision. Use this metric over context_precision when you have ground truth for more reliable evaluation.

### context_precision_non_llm
- **Description:** Context precision via embeddings instead of LLM.
- **Calculation:** Embeds chunks and reference, scores by cosine similarity, applies rank weighting.
- **Example:** 5 chunks, embedding similarity to reference: [0.9, 0.85, 0.3, 0.8, 0.2] → high score (top chunks are relevant)
- **Interpretation:** Fast/cheap proxy. Use for rapid experimentation before confirming with LLM-based version.
- **How to fix:** Same fixes as context_precision. If scores diverge from LLM version, upgrade embedding model.

### noise_sensitivity
- **Description:** Is the answer robust when irrelevant context is mixed in?
- **Calculation:** Measures answer quality degradation when noisy/irrelevant chunks are present alongside relevant ones.
- **Example:** Answer scores 0.9 with clean context but 0.5 with noise added → noise sensitivity = 0.5
- **Interpretation:** Low score = model easily distracted by irrelevant chunks.
- **How to fix:** Enable reranker to push irrelevant chunks down. Filter chunks below a similarity threshold before passing to LLM. Add "ignore irrelevant context" instruction in prompt. Reduce RETRIEVAL_K.

---

## 3. Generation Faithfulness (fix hallucination once retrieval is solid)

### faithfulness
- **Description:** Does the answer stick to the retrieved context, or does it hallucinate?
- **Calculation:** LLM decomposes answer into atomic claims, checks each against context. Score = `supported claims / total claims`.
- **Example:** Answer makes 8 claims, 6 are supported by context → `6/8 = 0.75`
- **Interpretation:** 1.0 = fully grounded. < 0.3 = largely hallucinated.
- **How to fix:** Add "only answer based on the provided context" to prompt. Lower temperature (0.0–0.2). Use a more instruction-following model. Improve context quality so the LLM doesn't need to fill gaps. Add chain-of-thought with citation requirements.

### faithfulness_hhem
- **Description:** Same goal as faithfulness but uses Hughes Hallucination Evaluation Model.
- **Calculation:** HHEM model scores answer-context pairs directly instead of claim decomposition.
- **Example:** HHEM returns 0.85 for an answer-context pair → score = 0.85
- **Interpretation:** Alternative hallucination detector. Can catch different patterns than LLM decomposition.
- **How to fix:** Same fixes as faithfulness. Use both metrics together — if they disagree, investigate those samples manually.

---

## 4. Answer Relevancy (usually improves when above are fixed)

### answer_relevancy
- **Description:** Does the answer actually address what was asked?
- **Calculation:** LLM generates hypothetical questions the answer would satisfy. These are compared to the original question via embedding cosine similarity.
- **Example:** Question: "What is the capital of France?" Answer talks about French cuisine → generated questions don't match original → score ~0.3
- **Interpretation:** > 0.8 = focused and on-topic. < 0.5 = answer misses the point.
- **How to fix:** Improve retrieval precision (irrelevant chunks cause off-topic answers). Add "answer the question directly" to prompt. Ensure the question is included in the LLM prompt. If retrieval metrics are good but this is low, try a more capable model.

---

## 5. Answer Quality (end-to-end correctness checks)

### answer_correctness
- **Description:** Overall correctness of the answer compared to reference.
- **Calculation:** Combines claim-level F1 (precision + recall of factual claims) with semantic similarity into a weighted score.
- **Example:** Claim F1 = 0.7, semantic similarity = 0.9, weight 0.5/0.5 → `(0.7 + 0.9) / 2 = 0.80`
- **Interpretation:** Balanced "is this answer right?" metric. Low score = either missing facts or stating wrong ones.
- **How to fix:** This is an end-to-end metric — fix upstream issues first (recall → precision → faithfulness). If still low, check if the model is summarizing too aggressively (losing claims) or adding unsupported claims.

### factual_correctness
- **Description:** Claim-level precision and recall vs reference.
- **Calculation:** Decomposes both response and reference into atomic claims. Computes how many claims match (TP), are extra (FP), or missing (FN). Returns F1.
- **Example:** Reference: 5 claims. Answer: 6 claims. 4 match → P=4/6, R=4/5, F1 ≈ 0.73
- **Interpretation:** High precision = no wrong claims. High recall = no missing claims. F1 balances both.
- **How to fix:** Low precision (wrong claims) → fix faithfulness (hallucination). Low recall (missing claims) → fix context_recall (retrieval gaps) or add "be comprehensive" to prompt. Both low → likely a retrieval + generation problem.

### semantic_similarity
- **Description:** How close is the answer's meaning to the reference?
- **Calculation:** Cosine similarity between embedding vectors of response and reference.
- **Example:** Embeddings of answer and reference have cosine similarity 0.92 → score = 0.92
- **Interpretation:** Fast, no LLM needed. High = semantically aligned. Doesn't catch factual errors if phrasing is similar.
- **How to fix:** If low despite correct facts → answer is phrased very differently from reference (may not be a real problem). If low with wrong facts → fix upstream retrieval and generation. Consider using a better embedding model for more accurate similarity.

---

## 6. Custom / Rubric-Based Metrics

### Aspect Critics (EVAL_ASPECT_*)
- **Description:** Binary pass/fail on any custom criteria you define.
- **Calculation:** LLM evaluates the response against your natural-language criterion. Returns 0 (fail) or 1 (pass).
- **Example:** Criterion: "Does the response cite page numbers?" Answer has no citations → score = 0
- **Interpretation:** Define in `.env`: `EVAL_ASPECT_CITATION=Does the response cite page numbers?`
- **How to fix:** Add explicit instructions in your prompt for the failing criteria. For citation aspects, add "cite sources" to prompt. For safety aspects, add guardrails or content filtering.

### Rubric-Based Scoring (EVAL_RUBRIC_*)
- **Description:** Custom 1–5 scale with detailed level descriptions.
- **Calculation:** LLM scores response against rubric definitions. Score normalized to 0–1 range.
- **Example:** Rubric levels 1–5 defined, LLM assigns level 4 → `(4-1)/(5-1) = 0.75`
- **Interpretation:** Define all 5 levels in `.env`: `EVAL_RUBRIC_1=Completely wrong` ... `EVAL_RUBRIC_5=Perfect with citations`
- **How to fix:** Analyze which rubric levels are most common. Tailor your prompt to address the gap between the current level and the next. Add few-shot examples in the prompt that match the desired rubric level.

### simple_criteria
- **Description:** General quality score on a simple scale.
- **Calculation:** LLM rates the response quality without specific criteria.
- **Example:** LLM judges answer as good quality → score = 0.80
- **Interpretation:** Quick sanity check. Less specific than aspect critics or rubrics.
- **How to fix:** If low, switch to aspect critics or rubrics to identify the specific quality dimension that's failing, then fix that.

### summarization
- **Description:** How well does a summary cover the source content?
- **Calculation:** Extracts keyphrases from source, checks how many appear in the summary.
- **Example:** Source has 10 keyphrases, 7 found in summary → score ≈ 0.70
- **Interpretation:** Low = summary misses key points. High = comprehensive coverage.
- **How to fix:** Add "include all key points" to prompt. Increase max output tokens if summary is being truncated. Pass more context to the LLM. Use a model with a larger context window.

---

## 7. Traditional NLP Metrics (No LLM)

Fast, deterministic, free — require reference answer.

### bleu
- **Description:** N-gram precision between response and reference.
- **Calculation:** Counts matching 1/2/3/4-grams, applies brevity penalty for short answers. Geometric mean of n-gram precisions.
- **Example:** Response shares 80% of bigrams with reference → BLEU ≈ 0.75
- **Interpretation:** High = wording closely matches reference. Originally designed for translation evaluation.
- **How to fix:** Low BLEU is expected when correct answers use different wording. Only optimize if exact phrasing matters. Add few-shot examples to guide output style.

### rouge
- **Description:** N-gram overlap with recall focus.
- **Calculation:** Measures how many n-grams from the reference appear in the response (recall-oriented).
- **Example:** Reference has 20 unigrams, 16 appear in response → ROUGE-1 = 0.80
- **Interpretation:** High = response covers reference content. Standard for summarization evaluation.
- **How to fix:** Low ROUGE-recall = answer is missing key terms from reference. Fix retrieval to surface relevant content. Add "use precise terminology" to prompt.

### exact_match
- **Description:** Strict string equality.
- **Calculation:** `1` if response == reference exactly, `0` otherwise.
- **Example:** Response: "Paris" vs Reference: "Paris" → 1. Response: "paris" vs Reference: "Paris" → 0.
- **Interpretation:** Binary. Useful for short factoid answers.
- **How to fix:** Add output formatting instructions ("answer in one word", "respond with only the answer"). Normalize casing/whitespace in post-processing if needed.

### string_presence
- **Description:** Is the reference a substring of the response?
- **Calculation:** `1` if reference string found within response, `0` otherwise.
- **Example:** Reference: "42" — Response: "The answer is 42 degrees" → 1
- **Interpretation:** Checks if the key answer appears somewhere in the response, regardless of surrounding text.
- **How to fix:** If failing, the model isn't producing the expected answer at all — fix retrieval and generation upstream. Check if the reference format matches what the model outputs (e.g., "42" vs "forty-two").

### string_similarity
- **Description:** Edit distance between response and reference.
- **Calculation:** Normalized Levenshtein, Hamming, or Jaro distance.
- **Example:** Response: "Pris" vs Reference: "Paris" → Levenshtein = 1 edit → normalized ≈ 0.80
- **Interpretation:** High = strings are close. Catches typos and minor variations. Low = very different strings.
- **How to fix:** If close but not exact → minor formatting or spelling issues, fix with output post-processing. If very low → the answer is fundamentally different, fix retrieval and generation.
