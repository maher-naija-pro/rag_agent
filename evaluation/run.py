"""One-command evaluation runner: generate testset + run evaluation.

Usage:
    python -m evaluation.run                                    # full pipeline, default PDF
    python -m evaluation.run --pdf /path/to/doc.pdf --size 20   # custom PDF, 20 samples
    python -m evaluation.run --skip-generate                    # reuse existing testset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.config import OUTPUT_DIR, PDF_PATH, TESTSET_SIZE


def main():
    parser = argparse.ArgumentParser(description="Generate testset and evaluate RAG pipeline")
    parser.add_argument("--pdf", default=PDF_PATH, help=f"Path to PDF (default: {PDF_PATH})")
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help=f"Testset size (default: {TESTSET_SIZE})")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--skip-generate", action="store_true", help="Skip testset generation, use existing")
    args = parser.parse_args()

    output_dir = Path(args.output)
    testset_path = output_dir / "testset.json"

    # Step 1: Generate testset
    if not args.skip_generate:
        print("=" * 60)
        print("STEP 1: Generating evaluation testset")
        print("=" * 60)
        from evaluation.generate_testset import generate_testset
        generate_testset(
            pdf_path=args.pdf,
            testset_size=args.size,
            output_dir=output_dir,
        )
    else:
        if not testset_path.exists():
            print(f"ERROR: --skip-generate set but testset not found: {testset_path}")
            return
        print(f"Skipping generation, using existing testset: {testset_path}")

    # Step 2: Run evaluation
    print()
    print("=" * 60)
    print("STEP 2: Running RAGAS evaluation")
    print("=" * 60)
    source = Path(args.pdf).name
    from evaluation.evaluate import run_evaluation
    run_evaluation(
        testset_path=str(testset_path),
        source=source,
        output_dir=output_dir,
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
