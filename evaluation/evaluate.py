"""Run RAGAS evaluation against the RAG pipeline.

Usage:
    python -m evaluation.evaluate                                        # uses generated testset
    python -m evaluation.evaluate --testset evaluation/output/testset.json
    python -m evaluation.evaluate --pdf /path/to/doc.pdf                 # custom PDF

This script:
1. Loads the testset (questions + reference answers + reference contexts)
2. Runs each question through the real RAG pipeline (graph.invoke)
3. Evaluates with RAGAS metrics (configurable via EVAL_METRICS in .env)
4. Saves detailed results to evaluation/output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add rag_agent_pipeline to sys.path so pipeline imports work
_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "rag_agent_pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── RAGAS metric imports ────────────────────────────────────────────────────
from ragas.metrics import (
    # Core RAG
    Faithfulness,
    FaithfulnesswithHHEM,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    # Answer quality
    AnswerCorrectness,
    FactualCorrectness,
    SemanticSimilarity,
    # Custom / rubric-based
    AspectCritic,
    SimpleCriteriaScore,
    RubricsScore,
    InstanceRubrics,
    # Traditional NLP
    BleuScore,
    RougeScore,
    ExactMatch,
    StringPresence,
    NonLLMStringSimilarity,
    # Summarization
    SummarizationScore,
)

from graph import build_graph
from evaluation.config import (
    EVAL_EMBEDDING_MODEL,
    EVAL_LLM_API_KEY,
    EVAL_LLM_BASE_URL,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    EVAL_METRICS,
    EVAL_ASPECT_CRITICS,
    EVAL_RUBRICS,
    OUTPUT_DIR,
    PDF_PATH,
)

# ── Metric registry ────────────────────────────────────────────────────────
# Each entry: name -> (needs_llm, needs_emb, factory)
# Factories receive (llm, emb) and return the metric instance.

_METRIC_REGISTRY: dict[str, tuple[bool, bool, callable]] = {
    # ── Core RAG (LLM-as-judge) ────────────────────────────────────────────
    "faithfulness":               (True, False, lambda llm, emb: Faithfulness(llm=llm)),
    "faithfulness_hhem":          (True, False, lambda llm, emb: FaithfulnesswithHHEM(llm=llm)),
    "answer_relevancy":           (True, True,  lambda llm, emb: ResponseRelevancy(llm=llm, embeddings=emb)),
    "context_precision":          (True, False, lambda llm, emb: LLMContextPrecisionWithoutReference(llm=llm)),
    "context_precision_ref":      (True, False, lambda llm, emb: LLMContextPrecisionWithReference(llm=llm)),
    "context_precision_non_llm":  (False, True, lambda llm, emb: NonLLMContextPrecisionWithReference(embeddings=emb)),
    "context_recall":             (True, False, lambda llm, emb: LLMContextRecall(llm=llm)),
    "context_recall_non_llm":     (False, True, lambda llm, emb: NonLLMContextRecall(embeddings=emb)),
    "context_entity_recall":      (True, False, lambda llm, emb: ContextEntityRecall(llm=llm)),
    "noise_sensitivity":          (True, False, lambda llm, emb: NoiseSensitivity(llm=llm)),

    # ── Answer quality ─────────────────────────────────────────────────────
    "answer_correctness":         (True, True,  lambda llm, emb: AnswerCorrectness(llm=llm, embeddings=emb)),
    "factual_correctness":        (True, False, lambda llm, emb: FactualCorrectness(llm=llm)),
    "semantic_similarity":        (False, True, lambda llm, emb: SemanticSimilarity(embeddings=emb)),

    # ── Custom / rubric-based ──────────────────────────────────────────────
    "simple_criteria":            (True, False, lambda llm, emb: SimpleCriteriaScore(llm=llm)),
    "summarization":              (True, False, lambda llm, emb: SummarizationScore(llm=llm)),

    # ── Traditional NLP (no LLM needed) ────────────────────────────────────
    "bleu":                       (False, False, lambda llm, emb: BleuScore()),
    "rouge":                      (False, False, lambda llm, emb: RougeScore()),
    "exact_match":                (False, False, lambda llm, emb: ExactMatch()),
    "string_presence":            (False, False, lambda llm, emb: StringPresence()),
    "string_similarity":          (False, False, lambda llm, emb: NonLLMStringSimilarity()),
}


def build_eval_llm() -> ChatOpenAI:
    """Build the LLM instance for RAGAS evaluation (judge)."""
    return ChatOpenAI(
        model=EVAL_LLM_MODEL,
        temperature=EVAL_LLM_TEMPERATURE,
        openai_api_key=EVAL_LLM_API_KEY,
        openai_api_base=EVAL_LLM_BASE_URL,
    )


def build_eval_embeddings() -> FastEmbedEmbeddings:
    """Build the embeddings instance for RAGAS evaluation (judge)."""
    return FastEmbedEmbeddings(model_name=EVAL_EMBEDDING_MODEL)


def build_metrics(eval_llm, eval_embeddings) -> list:
    """Build RAGAS metrics from the EVAL_METRICS config list + aspect critics + rubrics."""
    evaluator_llm = LangchainLLMWrapper(eval_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    metrics = []

    # Standard metrics from registry
    for name in EVAL_METRICS:
        entry = _METRIC_REGISTRY.get(name)
        if entry is None:
            print(f"  Warning: unknown metric '{name}', skipping. "
                  f"Available: {', '.join(sorted(_METRIC_REGISTRY.keys()))}")
            continue
        needs_llm, needs_emb, factory = entry
        llm_arg = evaluator_llm if needs_llm else None
        emb_arg = evaluator_embeddings if needs_emb else None
        metrics.append(factory(llm_arg, emb_arg))

    # Aspect critics (custom binary pass/fail)
    for aspect_name, definition in EVAL_ASPECT_CRITICS.items():
        metrics.append(AspectCritic(
            name=aspect_name,
            definition=definition,
            llm=evaluator_llm,
        ))

    # Rubric-based scoring (custom 1-5 scale)
    if EVAL_RUBRICS:
        metrics.append(RubricsScore(
            llm=evaluator_llm,
            rubrics=EVAL_RUBRICS,
        ))

    if not metrics:
        raise ValueError(
            f"No valid metrics configured. Set EVAL_METRICS in .env to one or more of: "
            f"{', '.join(sorted(_METRIC_REGISTRY.keys()))}"
        )

    return metrics


def run_pipeline_query(graph, question: str, source: str, thread_id: str) -> dict:
    """Run a single question through the real RAG pipeline and return answer + contexts."""
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "source": source,
            "raw_pages": [],
            "chunks": [],
            "candidates": [],
            "context": [],
            "answer": "",
            "ingested": True,  # data is already in Qdrant
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    context_docs = result.get("context", [])
    contexts = [doc.page_content for doc in context_docs]
    answer = result.get("answer", "")

    return {"answer": answer, "contexts": contexts}


def load_testset(testset_path: str) -> list[dict]:
    """Load testset from JSON file."""
    with open(testset_path) as f:
        return json.load(f)


def run_evaluation(
    testset_path: str,
    source: str,
    output_dir: Path,
) -> Path:
    """Run the full evaluation pipeline."""
    # Load testset
    testset = load_testset(testset_path)
    print(f"Loaded {len(testset)} test samples from {testset_path}")

    # Build the real pipeline graph
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer)
    print("Pipeline graph compiled")

    # Build RAGAS judge components
    eval_llm = build_eval_llm()
    eval_embeddings = build_eval_embeddings()
    metrics = build_metrics(eval_llm, eval_embeddings)

    print(f"Judge LLM: {EVAL_LLM_MODEL} @ {EVAL_LLM_BASE_URL}")
    print(f"Judge embeddings: {EVAL_EMBEDDING_MODEL}")
    metric_names = [m.name if hasattr(m, 'name') else type(m).__name__ for m in metrics]
    print(f"Metrics ({len(metrics)}): {', '.join(metric_names)}")

    # Run each question through the RAG pipeline
    results = []
    for i, sample in enumerate(testset):
        question = sample.get("user_input", sample.get("question", ""))
        reference = sample.get("reference", sample.get("ground_truth", ""))
        reference_contexts = sample.get("reference_contexts", sample.get("contexts", []))

        print(f"  [{i+1}/{len(testset)}] {question[:80]}...")

        # Use a unique thread_id per question to avoid cache/history interference
        thread_id = f"eval_{i}"

        try:
            pipeline_result = run_pipeline_query(graph, question, source, thread_id)
            answer = pipeline_result["answer"]
            contexts = pipeline_result["contexts"]
        except Exception as e:
            print(f"    Pipeline failed: {e}")
            answer = "Error generating answer."
            contexts = []

        results.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
            "reference_contexts": reference_contexts if isinstance(reference_contexts, list) else [],
        })

    # Build RAGAS evaluation dataset
    eval_dataset = Dataset.from_dict({
        "user_input": [r["user_input"] for r in results],
        "response": [r["response"] for r in results],
        "retrieved_contexts": [r["retrieved_contexts"] for r in results],
        "reference": [r["reference"] for r in results],
        "reference_contexts": [r["reference_contexts"] for r in results],
    })

    print(f"\nRunning RAGAS evaluation with {len(metrics)} metrics...")
    eval_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall scores
    scores = {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, (int, float))}
    scores_path = output_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nOverall scores saved: {scores_path}")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")

    # Detailed per-sample results
    detail_df = eval_result.to_pandas()
    detail_csv = output_dir / "evaluation_details.csv"
    detail_df.to_csv(detail_csv, index=False)
    print(f"Detailed results saved: {detail_csv}")

    detail_json = output_dir / "evaluation_details.json"
    detail_df.to_json(detail_json, orient="records", indent=2, default_handler=str)
    print(f"Detailed results JSON: {detail_json}")

    return scores_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS metrics")
    parser.add_argument(
        "--testset",
        default=str(OUTPUT_DIR / "testset.json"),
        help="Path to testset JSON",
    )
    parser.add_argument(
        "--source",
        default="",
        help="PDF source filename to filter retrieval (e.g. eu_ai_act.pdf)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Auto-detect source from PDF_PATH if not provided
    source = args.source
    if not source:
        source = Path(PDF_PATH).name

    run_evaluation(
        testset_path=args.testset,
        source=source,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
