"""Evaluation configuration — fully configurable via evaluation/.env.

All settings have sensible defaults and can be overridden with environment
variables. The pipeline's .env is loaded first for shared defaults (e.g.
LLM_MODEL, EMBEDDING_MODEL), then evaluation/.env overrides with EVAL_*
prefixed variables.
"""

# Module pour accéder aux variables d'environnement
import os
# Gestion portable des chemins de fichiers
from pathlib import Path

# Chargement de variables depuis des fichiers .env
from dotenv import load_dotenv

# Chargement du fichier .env du pipeline principal pour les valeurs par défaut partagées
_PIPELINE_ENV = Path(__file__).resolve().parent.parent / "rag_agent_pipeline" / ".env"
# Charge les variables d'environnement du pipeline
load_dotenv(dotenv_path=_PIPELINE_ENV)

# Chargement du fichier .env spécifique à l'évaluation (écrase les valeurs du pipeline)
_LOCAL_ENV = Path(__file__).resolve().parent / ".env"
# override=True pour écraser les valeurs déjà définies
load_dotenv(dotenv_path=_LOCAL_ENV, override=True)


# ── PDF d'entrée ────────────────────────────────────────────────────────────
# Chemin vers le PDF à utiliser pour la génération de questions et l'évaluation
PDF_PATH = os.getenv(
    "EVAL_PDF_PATH",
    # Valeur par défaut : le PDF de l'EU AI Act
    str(Path(__file__).resolve().parent.parent / "data" / "eu_ai_act.pdf"),
)

# ── LLM juge RAGAS ─────────────────────────────────────────────────────────
# Modèle LLM utilisé par RAGAS pour noter les réponses (distinct du LLM du pipeline)
# Modèle LLM pour l'évaluation
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "mistral:7b"))
# URL de base de l'API LLM
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
# Clé API pour le LLM d'évaluation
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "ollama")
# Température du LLM (0 = déterministe)
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "0"))

# ── Embeddings juge RAGAS ──────────────────────────────────────────────────
# Modèle d'embeddings utilisé par RAGAS pour calculer la similarité sémantique
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

# ── Génération du jeu de test ──────────────────────────────────────────────
# Nombre d'échantillons de test à générer
TESTSET_SIZE = int(os.getenv("EVAL_TESTSET_SIZE", "10"))
# Taille des morceaux de texte en caractères
CHUNK_SIZE = int(os.getenv("EVAL_CHUNK_SIZE", os.getenv("CHUNK_SIZE", "800")))
# Chevauchement entre les morceaux de texte
CHUNK_OVERLAP = int(os.getenv("EVAL_CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP", "150")))

# ── Sortie ──────────────────────────────────────────────────────────────────
# Répertoire de sortie pour les résultats d'évaluation
OUTPUT_DIR = Path(os.getenv(
    "EVAL_OUTPUT_DIR",
    # Par défaut : evaluation/output/
    str(Path(__file__).resolve().parent / "output"),
))

# ── Métriques RAGAS ───────────────────────────────────────────────────────
# Liste des métriques d'évaluation à calculer, configurables via variable d'environnement
EVAL_METRICS = [
    # Suppression des espaces autour de chaque nom de métrique
    m.strip()
    for m in os.getenv(
        "EVAL_METRICS",
        # Métriques par défaut
        "faithfulness,answer_relevancy,context_precision,context_recall",
    # Découpage de la chaîne en liste par virgule
    ).split(",")
    # Exclusion des entrées vides
    if m.strip()
]

# ── Critères d'aspect (évaluation binaire réussi/échoué personnalisée) ──────
# Format : EVAL_ASPECT_<NOM>=<définition>
# Chaque variable d'environnement crée un AspectCritic qui note 0 ou 1.
# Exemple : EVAL_ASPECT_HARMFULNESS="Does the response contain harmful content?"
# Dictionnaire des critiques d'aspect personnalisés
EVAL_ASPECT_CRITICS: dict[str, str] = {}
# Préfixe des variables d'environnement pour les critiques d'aspect
_ASPECT_PREFIX = "EVAL_ASPECT_"
# Parcours de toutes les variables d'environnement
for key, value in os.environ.items():
    # Filtrage des variables commençant par le préfixe
    if key.startswith(_ASPECT_PREFIX) and value.strip():
        # Extraction du nom de l'aspect en minuscules
        aspect_name = key[len(_ASPECT_PREFIX):].lower()
        # Ajout de l'aspect au dictionnaire
        EVAL_ASPECT_CRITICS[aspect_name] = value.strip()

# ── Notation par grille de critères (échelle personnalisée de 1 à 5) ──────
# Format : EVAL_RUBRIC_<N>=<description>  (N = 1 à 5)
# Exemple :
#   EVAL_RUBRIC_1=Completely wrong or irrelevant
#   EVAL_RUBRIC_5=Perfect answer with citations
# Dictionnaire des descriptions de la grille de notation
EVAL_RUBRICS: dict[int, str] = {}
# Boucle de 1 à 5 pour chaque niveau de la grille
for i in range(1, 6):
    # Récupération de la description du niveau
    desc = os.getenv(f"EVAL_RUBRIC_{i}", "").strip()
    # Ajout uniquement si la description n'est pas vide
    if desc:
        EVAL_RUBRICS[i] = desc
