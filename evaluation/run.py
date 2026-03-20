"""One-command evaluation runner: generate testset + run evaluation.

Usage:
    python -m evaluation.run                                    # full pipeline, default PDF
    python -m evaluation.run --pdf /path/to/doc.pdf --size 20   # custom PDF, 20 samples
    python -m evaluation.run --skip-generate                    # reuse existing testset
"""

# Permet d'utiliser les annotations de type modernes
from __future__ import annotations

# Module pour analyser les arguments de la ligne de commande
import argparse
# Gestion des chemins de fichiers de manière portable
from pathlib import Path

# Importation des paramètres de configuration
from evaluation.config import OUTPUT_DIR, PDF_PATH, TESTSET_SIZE


def main():
    # Création du parseur d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Generate testset and evaluate RAG pipeline")
    # Chemin vers le PDF source
    parser.add_argument("--pdf", default=PDF_PATH, help=f"Path to PDF (default: {PDF_PATH})")
    # Nombre d'échantillons de test à générer
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help=f"Testset size (default: {TESTSET_SIZE})")
    # Répertoire de sortie des résultats
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    # Option pour sauter la génération et réutiliser un testset existant
    parser.add_argument("--skip-generate", action="store_true", help="Skip testset generation, use existing")
    # Analyse des arguments fournis par l'utilisateur
    args = parser.parse_args()

    # Conversion du répertoire de sortie en objet Path
    output_dir = Path(args.output)
    # Chemin complet vers le fichier de testset JSON
    testset_path = output_dir / "testset.json"

    # Étape 1 : Génération du jeu de test
    if not args.skip_generate:
        # Affichage d'un séparateur visuel
        print("=" * 60)
        print("STEP 1: Generating evaluation testset")
        print("=" * 60)
        # Importation différée pour éviter un chargement inutile
        from evaluation.generate_testset import generate_testset
        # Lancement de la génération du testset
        generate_testset(
            pdf_path=args.pdf,
            testset_size=args.size,
            output_dir=output_dir,
        )
    else:
        # Vérification que le testset existe déjà
        if not testset_path.exists():
            # Message d'erreur si le fichier est introuvable
            print(f"ERROR: --skip-generate set but testset not found: {testset_path}")
            return
        # Confirmation de l'utilisation du testset existant
        print(f"Skipping generation, using existing testset: {testset_path}")

    # Étape 2 : Lancement de l'évaluation RAGAS
    print()
    print("=" * 60)
    print("STEP 2: Running RAGAS evaluation")
    print("=" * 60)
    # Extraction du nom du fichier PDF (sans le chemin)
    source = Path(args.pdf).name
    # Importation différée du module d'évaluation
    from evaluation.evaluate import run_evaluation
    # Exécution de l'évaluation du pipeline RAG
    run_evaluation(
        testset_path=str(testset_path),
        source=source,
        output_dir=output_dir,
    )

    # Affichage du message de fin avec le répertoire des résultats
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    # Indique où trouver les résultats
    print(f"Results in: {output_dir}")
    print("=" * 60)


# Point d'entrée lorsqu'on exécute le script directement
if __name__ == "__main__":
    main()
