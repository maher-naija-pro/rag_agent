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

# Permet les annotations de type modernes
from __future__ import annotations

# Module pour analyser les arguments de la ligne de commande
import argparse
# Module pour la sérialisation/désérialisation JSON
import json
# Accès aux fonctions système (path, exit)
import sys
# Manipulation portable des chemins de fichiers
from pathlib import Path

# Ajout du répertoire rag_agent_pipeline au sys.path pour que les imports du pipeline fonctionnent
_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "rag_agent_pipeline")
# Vérification pour éviter les doublons dans le chemin
if _PIPELINE_DIR not in sys.path:
    # Insertion en début de liste pour priorité maximale
    sys.path.insert(0, _PIPELINE_DIR)

# Bibliothèque de manipulation de données tabulaires
import pandas as pd
# Classe Dataset de HuggingFace pour structurer les données d'évaluation
from datasets import Dataset
# Message utilisateur pour le pipeline LangChain
from langchain_core.messages import HumanMessage
# Client LLM compatible OpenAI via LangChain
from langchain_openai import ChatOpenAI
# Modèle d'embeddings rapide et local
from langchain_community.embeddings import FastEmbedEmbeddings
# Sauvegarde de l'état du graphe en mémoire
from langgraph.checkpoint.memory import InMemorySaver
# Fonction principale d'évaluation RAGAS
from ragas import evaluate
# Adaptateur LLM LangChain pour RAGAS
from ragas.llms import LangchainLLMWrapper
# Adaptateur embeddings LangChain pour RAGAS
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Imports des métriques RAGAS ────────────────────────────────────────────
from ragas.metrics import (
    # Métriques RAG de base
    # Fidélité : la réponse est-elle fidèle au contexte récupéré ?
    Faithfulness,
    # Fidélité avec le modèle HHEM (hallucination)
    FaithfulnesswithHHEM,
    # Pertinence de la réponse par rapport à la question
    ResponseRelevancy,
    # Précision du contexte sans référence (via LLM)
    LLMContextPrecisionWithoutReference,
    # Précision du contexte avec référence (via LLM)
    LLMContextPrecisionWithReference,
    # Précision du contexte avec référence (sans LLM)
    NonLLMContextPrecisionWithReference,
    # Rappel du contexte via LLM
    LLMContextRecall,
    # Rappel du contexte sans LLM
    NonLLMContextRecall,
    # Rappel des entités dans le contexte
    ContextEntityRecall,
    # Sensibilité au bruit dans le contexte
    NoiseSensitivity,
    # Métriques de qualité de réponse
    # Exactitude de la réponse
    AnswerCorrectness,
    # Correction factuelle de la réponse
    FactualCorrectness,
    # Similarité sémantique entre réponse et référence
    SemanticSimilarity,
    # Métriques personnalisées / basées sur des grilles
    # Critique d'aspect : évaluation binaire (réussi/échoué)
    AspectCritic,
    # Score basé sur un critère simple
    SimpleCriteriaScore,
    # Score basé sur une grille de notation
    RubricsScore,
    # Grille de notation par instance
    InstanceRubrics,
    # Métriques NLP traditionnelles
    # Score BLEU (n-grammes)
    BleuScore,
    # Score ROUGE (rappel de n-grammes)
    RougeScore,
    # Correspondance exacte entre réponse et référence
    ExactMatch,
    # Vérification de la présence d'une chaîne
    StringPresence,
    # Similarité de chaînes sans LLM
    NonLLMStringSimilarity,
    # Résumé
    # Score de qualité de résumé
    SummarizationScore,
)

# Fonction de construction du graphe du pipeline RAG
from graph import build_graph
# Importation des paramètres de configuration d'évaluation
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

# ── Registre des métriques ────────────────────────────────────────────────
# Chaque entrée : nom -> (nécessite_llm, nécessite_embeddings, fonction_fabrique)
# Les fonctions fabriques reçoivent (llm, emb) et retournent l'instance de métrique.

_METRIC_REGISTRY: dict[str, tuple[bool, bool, callable]] = {
    # ── RAG de base (LLM comme juge) ──────────────────────────────────────
    # Mesure la fidélité au contexte
    "faithfulness":               (True, False, lambda llm, emb: Faithfulness(llm=llm)),
    # Fidélité avec modèle HHEM
    "faithfulness_hhem":          (True, False, lambda llm, emb: FaithfulnesswithHHEM(llm=llm)),
    # Pertinence de la réponse
    "answer_relevancy":           (True, True,  lambda llm, emb: ResponseRelevancy(llm=llm, embeddings=emb)),
    # Précision du contexte sans réf.
    "context_precision":          (True, False, lambda llm, emb: LLMContextPrecisionWithoutReference(llm=llm)),
    # Précision du contexte avec réf.
    "context_precision_ref":      (True, False, lambda llm, emb: LLMContextPrecisionWithReference(llm=llm)),
    # Précision sans LLM
    "context_precision_non_llm":  (False, True, lambda llm, emb: NonLLMContextPrecisionWithReference(embeddings=emb)),
    # Rappel du contexte via LLM
    "context_recall":             (True, False, lambda llm, emb: LLMContextRecall(llm=llm)),
    # Rappel sans LLM
    "context_recall_non_llm":     (False, True, lambda llm, emb: NonLLMContextRecall(embeddings=emb)),
    # Rappel des entités du contexte
    "context_entity_recall":      (True, False, lambda llm, emb: ContextEntityRecall(llm=llm)),
    # Sensibilité au bruit
    "noise_sensitivity":          (True, False, lambda llm, emb: NoiseSensitivity(llm=llm)),

    # ── Qualité de réponse ────────────────────────────────────────────────
    # Exactitude globale
    "answer_correctness":         (True, True,  lambda llm, emb: AnswerCorrectness(llm=llm, embeddings=emb)),
    # Correction factuelle
    "factual_correctness":        (True, False, lambda llm, emb: FactualCorrectness(llm=llm)),
    # Similarité sémantique
    "semantic_similarity":        (False, True, lambda llm, emb: SemanticSimilarity(embeddings=emb)),

    # ── Personnalisées / grilles ──────────────────────────────────────────
    # Score sur critère simple
    "simple_criteria":            (True, False, lambda llm, emb: SimpleCriteriaScore(llm=llm)),
    # Score de résumé
    "summarization":              (True, False, lambda llm, emb: SummarizationScore(llm=llm)),

    # ── NLP traditionnelle (pas de LLM nécessaire) ────────────────────────
    # Score BLEU
    "bleu":                       (False, False, lambda llm, emb: BleuScore()),
    # Score ROUGE
    "rouge":                      (False, False, lambda llm, emb: RougeScore()),
    # Correspondance exacte
    "exact_match":                (False, False, lambda llm, emb: ExactMatch()),
    # Présence de chaîne
    "string_presence":            (False, False, lambda llm, emb: StringPresence()),
    # Similarité de chaînes
    "string_similarity":          (False, False, lambda llm, emb: NonLLMStringSimilarity()),
}


def build_eval_llm() -> ChatOpenAI:
    """Build the LLM instance for RAGAS evaluation (judge)."""
    # Création du LLM juge pour l'évaluation RAGAS
    return ChatOpenAI(
        # Nom du modèle
        model=EVAL_LLM_MODEL,
        # Température (0 = déterministe)
        temperature=EVAL_LLM_TEMPERATURE,
        # Clé API
        openai_api_key=EVAL_LLM_API_KEY,
        # URL de base de l'API
        openai_api_base=EVAL_LLM_BASE_URL,
    )


def build_eval_embeddings() -> FastEmbedEmbeddings:
    """Build the embeddings instance for RAGAS evaluation (judge)."""
    # Création du modèle d'embeddings pour l'évaluation
    return FastEmbedEmbeddings(model_name=EVAL_EMBEDDING_MODEL)


def build_metrics(eval_llm, eval_embeddings) -> list:
    """Build RAGAS metrics from the EVAL_METRICS config list + aspect critics + rubrics."""
    # Encapsulation du LLM pour RAGAS
    evaluator_llm = LangchainLLMWrapper(eval_llm)
    # Encapsulation des embeddings pour RAGAS
    evaluator_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # Liste des métriques à utiliser pour l'évaluation
    metrics = []

    # Métriques standard depuis le registre
    # Parcours des noms de métriques configurées
    for name in EVAL_METRICS:
        # Recherche de la métrique dans le registre
        entry = _METRIC_REGISTRY.get(name)
        # Vérification si la métrique existe
        if entry is None:
            # Avertissement pour métrique inconnue
            print(f"  Warning: unknown metric '{name}', skipping. "
                  f"Available: {', '.join(sorted(_METRIC_REGISTRY.keys()))}")
            continue
        # Décomposition du tuple (nécessite_llm, nécessite_emb, fabrique)
        needs_llm, needs_emb, factory = entry
        # Passage du LLM uniquement si nécessaire
        llm_arg = evaluator_llm if needs_llm else None
        # Passage des embeddings uniquement si nécessaire
        emb_arg = evaluator_embeddings if needs_emb else None
        # Création et ajout de l'instance de métrique
        metrics.append(factory(llm_arg, emb_arg))

    # Critiques d'aspect (évaluation binaire réussi/échoué personnalisée)
    # Parcours des critiques d'aspect configurés
    for aspect_name, definition in EVAL_ASPECT_CRITICS.items():
        metrics.append(AspectCritic(
            # Nom du critère d'aspect
            name=aspect_name,
            # Définition textuelle du critère
            definition=definition,
            # LLM utilisé pour juger
            llm=evaluator_llm,
        ))

    # Notation par grille de critères (échelle personnalisée de 1 à 5)
    # Si des grilles de notation sont définies
    if EVAL_RUBRICS:
        metrics.append(RubricsScore(
            # LLM utilisé pour noter
            llm=evaluator_llm,
            # Dictionnaire des descriptions par niveau
            rubrics=EVAL_RUBRICS,
        ))

    # Vérification qu'au moins une métrique est configurée
    if not metrics:
        raise ValueError(
            f"No valid metrics configured. Set EVAL_METRICS in .env to one or more of: "
            f"{', '.join(sorted(_METRIC_REGISTRY.keys()))}"
        )

    # Retourne la liste des métriques configurées
    return metrics


def run_pipeline_query(graph, question: str, source: str, thread_id: str) -> dict:
    """Run a single question through the real RAG pipeline and return answer + contexts."""
    # Exécution de la question à travers le graphe du pipeline RAG
    result = graph.invoke(
        {
            # Message utilisateur avec la question
            "messages": [HumanMessage(content=question)],
            # Question brute
            "question": question,
            # Nom du fichier source pour filtrer la recherche
            "source": source,
            # Pages brutes (vide car déjà ingéré)
            "raw_pages": [],
            # Morceaux de texte (vide car déjà ingéré)
            "chunks": [],
            # Candidats pour la recherche
            "candidates": [],
            # Contexte récupéré (sera rempli par le pipeline)
            "context": [],
            # Réponse (sera générée par le pipeline)
            "answer": "",
            # Indique que les données sont déjà dans Qdrant
            "ingested": True,
        },
        # Identifiant de conversation unique
        config={"configurable": {"thread_id": thread_id}},
    )

    # Récupération des documents de contexte
    context_docs = result.get("context", [])
    # Extraction du texte de chaque document de contexte
    contexts = [doc.page_content for doc in context_docs]
    # Récupération de la réponse générée
    answer = result.get("answer", "")

    # Retourne la réponse et les contextes
    return {"answer": answer, "contexts": contexts}


def load_testset(testset_path: str) -> list[dict]:
    """Load testset from JSON file."""
    # Ouverture du fichier JSON du testset
    with open(testset_path) as f:
        # Désérialisation et retour de la liste de dictionnaires
        return json.load(f)


def run_evaluation(
    testset_path: str,
    source: str,
    output_dir: Path,
) -> Path:
    """Run the full evaluation pipeline."""
    # Chargement du jeu de test
    # Chargement des échantillons de test depuis le fichier JSON
    testset = load_testset(testset_path)
    # Affichage du nombre d'échantillons chargés
    print(f"Loaded {len(testset)} test samples from {testset_path}")

    # Construction du graphe réel du pipeline RAG
    # Création d'un gestionnaire de checkpoints en mémoire
    checkpointer = InMemorySaver()
    # Compilation du graphe du pipeline RAG
    graph = build_graph(checkpointer)
    # Confirmation de la compilation
    print("Pipeline graph compiled")

    # Construction des composants du juge RAGAS
    # Instanciation du LLM juge
    eval_llm = build_eval_llm()
    # Instanciation des embeddings juge
    eval_embeddings = build_eval_embeddings()
    # Construction de la liste des métriques
    metrics = build_metrics(eval_llm, eval_embeddings)

    # Affichage du modèle LLM juge
    print(f"Judge LLM: {EVAL_LLM_MODEL} @ {EVAL_LLM_BASE_URL}")
    # Affichage du modèle d'embeddings juge
    print(f"Judge embeddings: {EVAL_EMBEDDING_MODEL}")
    # Extraction des noms de métriques
    metric_names = [m.name if hasattr(m, 'name') else type(m).__name__ for m in metrics]
    # Affichage des métriques utilisées
    print(f"Metrics ({len(metrics)}): {', '.join(metric_names)}")

    # Exécution de chaque question à travers le pipeline RAG
    # Liste pour stocker les résultats de chaque question
    results = []
    # Parcours de chaque échantillon du testset
    for i, sample in enumerate(testset):
        # Extraction de la question (compatibilité multi-format)
        question = sample.get("user_input", sample.get("question", ""))
        # Extraction de la réponse de référence
        reference = sample.get("reference", sample.get("ground_truth", ""))
        # Extraction des contextes de référence
        reference_contexts = sample.get("reference_contexts", sample.get("contexts", []))

        # Affichage de la progression avec un aperçu de la question
        print(f"  [{i+1}/{len(testset)}] {question[:80]}...")

        # Utilisation d'un thread_id unique par question pour éviter les interférences de cache/historique
        thread_id = f"eval_{i}"

        try:
            # Exécution de la question dans le pipeline
            pipeline_result = run_pipeline_query(graph, question, source, thread_id)
            # Récupération de la réponse générée
            answer = pipeline_result["answer"]
            # Récupération des contextes récupérés
            contexts = pipeline_result["contexts"]
        # Gestion des erreurs du pipeline
        except Exception as e:
            # Affichage de l'erreur
            print(f"    Pipeline failed: {e}")
            # Réponse par défaut en cas d'erreur
            answer = "Error generating answer."
            # Pas de contexte en cas d'erreur
            contexts = []

        # Ajout du résultat à la liste
        results.append({
            # Question posée
            "user_input": question,
            # Réponse du pipeline
            "response": answer,
            # Contextes récupérés par le pipeline
            "retrieved_contexts": contexts,
            # Réponse de référence attendue
            "reference": reference,
            # Contextes de référence (avec vérification de type)
            "reference_contexts": reference_contexts if isinstance(reference_contexts, list) else [],
        })

    # Construction du dataset d'évaluation RAGAS au format HuggingFace
    eval_dataset = Dataset.from_dict({
        # Liste de toutes les questions
        "user_input": [r["user_input"] for r in results],
        # Liste de toutes les réponses du pipeline
        "response": [r["response"] for r in results],
        # Liste des contextes récupérés
        "retrieved_contexts": [r["retrieved_contexts"] for r in results],
        # Liste des réponses de référence
        "reference": [r["reference"] for r in results],
        # Liste des contextes de référence
        "reference_contexts": [r["reference_contexts"] for r in results],
    })

    # Lancement de l'évaluation
    print(f"\nRunning RAGAS evaluation with {len(metrics)} metrics...")
    # Exécution de l'évaluation RAGAS sur le dataset
    eval_result = evaluate(
        dataset=eval_dataset,
        # Liste des métriques à calculer
        metrics=metrics,
    )

    # Sauvegarde des résultats
    # Création du répertoire de sortie si nécessaire
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scores globaux (moyennes par métrique)
    # Arrondi des scores à 4 décimales
    scores = {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, (int, float))}
    # Chemin du fichier des scores
    scores_path = output_dir / "scores.json"
    # Écriture des scores en JSON
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    # Confirmation de la sauvegarde
    print(f"\nOverall scores saved: {scores_path}")
    # Affichage de chaque score
    for metric, score in scores.items():
        print(f"  {metric}: {score}")

    # Résultats détaillés par échantillon
    # Conversion des résultats en DataFrame pandas
    detail_df = eval_result.to_pandas()
    # Chemin du fichier CSV détaillé
    detail_csv = output_dir / "evaluation_details.csv"
    # Sauvegarde en CSV
    detail_df.to_csv(detail_csv, index=False)
    # Confirmation de la sauvegarde CSV
    print(f"Detailed results saved: {detail_csv}")

    # Chemin du fichier JSON détaillé
    detail_json = output_dir / "evaluation_details.json"
    # Sauvegarde en JSON
    detail_df.to_json(detail_json, orient="records", indent=2, default_handler=str)
    # Confirmation de la sauvegarde JSON
    print(f"Detailed results JSON: {detail_json}")

    # Retourne le chemin vers le fichier des scores
    return scores_path


def main():
    # Création du parseur d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS metrics")
    # Argument pour le chemin du testset
    parser.add_argument(
        "--testset",
        default=str(OUTPUT_DIR / "testset.json"),
        help="Path to testset JSON",
    )
    # Argument pour le nom du fichier source PDF
    parser.add_argument(
        "--source",
        default="",
        help="PDF source filename to filter retrieval (e.g. eu_ai_act.pdf)",
    )
    # Argument pour le répertoire de sortie
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    # Analyse des arguments fournis
    args = parser.parse_args()

    # Détection automatique du nom du fichier source depuis PDF_PATH si non fourni
    source = args.source
    # Si aucun source n'est spécifié
    if not source:
        # Extraction du nom de fichier depuis le chemin du PDF
        source = Path(PDF_PATH).name

    # Lancement de l'évaluation avec les arguments fournis
    run_evaluation(
        testset_path=args.testset,
        source=source,
        output_dir=Path(args.output),
    )


# Point d'entrée lorsqu'on exécute le script directement
if __name__ == "__main__":
    main()
