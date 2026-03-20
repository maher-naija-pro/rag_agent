"""Generate a RAGAS evaluation testset from a PDF document.

Usage:
    python -m evaluation.generate_testset                          # uses default PDF
    python -m evaluation.generate_testset --pdf /path/to/doc.pdf   # custom PDF
    python -m evaluation.generate_testset --size 20                # 20 test samples

Environment variables (or set in evaluation/.env):
    EVAL_PDF_PATH      — path to the PDF to generate questions from
    EVAL_TESTSET_SIZE  — number of test samples to generate (default: 10)
    EVAL_OUTPUT_DIR    — directory for output files (default: evaluation/output)
"""

# Permet les annotations de type modernes
from __future__ import annotations

# Analyse des arguments en ligne de commande
import argparse
# Sérialisation et désérialisation JSON
import json
# Accès aux fonctions système (stderr, exit)
import sys
# Manipulation portable des chemins de fichiers
from pathlib import Path

# PyMuPDF — bibliothèque pour lire et extraire le texte des PDF
import fitz
# Classe représentant un document avec contenu et métadonnées
from langchain_core.documents import Document
# Client LLM compatible OpenAI via LangChain
from langchain_openai import ChatOpenAI
# Découpeur de texte récursif en morceaux
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Générateur de jeux de test RAGAS
from ragas.testset import TestsetGenerator
# Adaptateur pour utiliser un LLM LangChain dans RAGAS
from ragas.llms import LangchainLLMWrapper
# Adaptateur pour utiliser des embeddings LangChain dans RAGAS
from ragas.embeddings import LangchainEmbeddingsWrapper
# Modèle d'embeddings rapide et local
from langchain_community.embeddings import FastEmbedEmbeddings

# Importation de tous les paramètres de configuration
from evaluation.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EVAL_EMBEDDING_MODEL,
    EVAL_LLM_API_KEY,
    EVAL_LLM_BASE_URL,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    OUTPUT_DIR,
    PDF_PATH,
    TESTSET_SIZE,
)


def load_pdf(pdf_path: str) -> list[Document]:
    """Extract text from PDF pages using PyMuPDF."""
    # Conversion du chemin en objet Path
    path = Path(pdf_path)
    # Vérification que le fichier PDF existe
    if not path.is_file():
        # Affichage de l'erreur sur stderr
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        # Arrêt du programme avec un code d'erreur
        sys.exit(1)

    # Ouverture du document PDF avec PyMuPDF
    doc = fitz.open(str(path))
    # Liste pour stocker les documents extraits
    documents = []

    # Parcours de chaque page du PDF
    for i, page in enumerate(doc):
        # Extraction du texte brut de la page
        text = page.get_text("text").strip()
        # Ajout uniquement si la page contient du texte
        if text:
            documents.append(Document(
                # Contenu textuel de la page
                page_content=text,
                # Métadonnées : nom du fichier et numéro de page
                metadata={"source": path.name, "page": i + 1},
            ))

    # Fermeture du document PDF
    doc.close()
    # Affichage du nombre de pages chargées
    print(f"Loaded {len(documents)} pages from {path.name}")
    # Retourne la liste des documents
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks matching pipeline settings."""
    # Création du découpeur de texte récursif avec les paramètres configurés
    splitter = RecursiveCharacterTextSplitter(
        # Taille maximale de chaque morceau
        chunk_size=CHUNK_SIZE,
        # Nombre de caractères de chevauchement entre morceaux
        chunk_overlap=CHUNK_OVERLAP,
        # Séparateurs utilisés par ordre de priorité
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # Découpage des documents en morceaux
    chunks = splitter.split_documents(documents)
    # Affichage du nombre de morceaux créés
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    # Retourne la liste des morceaux
    return chunks


def build_llm() -> ChatOpenAI:
    """Build the LLM instance for testset generation."""
    # Création d'une instance LLM avec les paramètres d'évaluation
    return ChatOpenAI(
        # Nom du modèle LLM à utiliser
        model=EVAL_LLM_MODEL,
        # Température de génération (0 = déterministe)
        temperature=EVAL_LLM_TEMPERATURE,
        # Clé API pour l'authentification
        openai_api_key=EVAL_LLM_API_KEY,
        # URL de base de l'API
        openai_api_base=EVAL_LLM_BASE_URL,
    )


def build_embeddings() -> FastEmbedEmbeddings:
    """Build the embeddings instance for testset generation."""
    # Création du modèle d'embeddings avec le modèle configuré
    return FastEmbedEmbeddings(model_name=EVAL_EMBEDDING_MODEL)


def generate_testset(
    pdf_path: str,
    testset_size: int,
    output_dir: Path,
) -> Path:
    """Generate a RAGAS testset and save it to disk."""
    # Chargement et découpage du PDF en morceaux
    # Extraction du texte du PDF
    documents = load_pdf(pdf_path)
    # Découpage du texte en morceaux
    chunks = chunk_documents(documents)

    # Construction des wrappers LLM et embeddings pour RAGAS
    # Instanciation du LLM
    llm = build_llm()
    # Instanciation du modèle d'embeddings
    embeddings = build_embeddings()

    # Encapsulation du LLM pour RAGAS
    generator_llm = LangchainLLMWrapper(llm)
    # Encapsulation des embeddings pour RAGAS
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Création du générateur de jeu de test RAGAS
    generator = TestsetGenerator(
        # LLM utilisé pour générer les questions et réponses
        llm=generator_llm,
        # Modèle d'embeddings pour la similarité sémantique
        embedding_model=generator_embeddings,
    )

    # Information sur la durée potentielle
    print(f"Generating {testset_size} test samples (this may take a while)...")
    # Génération du testset à partir des documents LangChain
    testset = generator.generate_with_langchain_docs(
        # Morceaux de texte comme source
        chunks,
        # Nombre d'échantillons à générer
        testset_size=testset_size,
    )

    # Sauvegarde des résultats
    # Création du répertoire de sortie s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde en CSV via un DataFrame pandas
    # Conversion du testset en DataFrame pandas
    df = testset.to_pandas()
    # Chemin du fichier CSV de sortie
    csv_path = output_dir / "testset.csv"
    # Écriture du CSV sans index
    df.to_csv(csv_path, index=False)
    # Confirmation de la sauvegarde CSV
    print(f"Saved testset CSV: {csv_path}")

    # Sauvegarde en JSON pour une utilisation programmatique
    # Chemin du fichier JSON de sortie
    json_path = output_dir / "testset.json"
    # Conversion en liste de dictionnaires
    records = df.to_dict(orient="records")
    # Ouverture du fichier en écriture
    with open(json_path, "w") as f:
        # Sérialisation JSON avec indentation
        json.dump(records, f, indent=2, default=str)
    # Confirmation de la sauvegarde JSON
    print(f"Saved testset JSON: {json_path}")

    # Affichage du nombre total d'échantillons générés
    print(f"\nGenerated {len(df)} test samples")
    # Affichage des colonnes du DataFrame
    print(f"Columns: {list(df.columns)}")
    # Affichage d'un exemple de question
    print(f"\nSample question: {df.iloc[0].get('user_input', df.iloc[0].get('question', 'N/A'))}")

    # Retourne le chemin vers le fichier JSON généré
    return json_path


def main():
    # Création du parseur d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Generate RAGAS evaluation testset from a PDF")
    # Chemin vers le PDF source
    parser.add_argument("--pdf", default=PDF_PATH, help=f"Path to PDF (default: {PDF_PATH})")
    # Taille du jeu de test
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help=f"Number of test samples (default: {TESTSET_SIZE})")
    # Répertoire de sortie
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    # Analyse des arguments
    args = parser.parse_args()

    # Lancement de la génération du testset avec les arguments fournis
    generate_testset(
        pdf_path=args.pdf,
        testset_size=args.size,
        output_dir=Path(args.output),
    )


# Point d'entrée lorsqu'on exécute le script directement
if __name__ == "__main__":
    main()
