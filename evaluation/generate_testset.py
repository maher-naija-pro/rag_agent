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

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import FastEmbedEmbeddings

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
    path = Path(pdf_path)
    if not path.is_file():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    doc = fitz.open(str(path))
    documents = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": i + 1},
            ))

    doc.close()
    print(f"Loaded {len(documents)} pages from {path.name}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks matching pipeline settings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_llm() -> ChatOpenAI:
    """Build the LLM instance for testset generation."""
    return ChatOpenAI(
        model=EVAL_LLM_MODEL,
        temperature=EVAL_LLM_TEMPERATURE,
        openai_api_key=EVAL_LLM_API_KEY,
        openai_api_base=EVAL_LLM_BASE_URL,
    )


def build_embeddings() -> FastEmbedEmbeddings:
    """Build the embeddings instance for testset generation."""
    return FastEmbedEmbeddings(model_name=EVAL_EMBEDDING_MODEL)


def generate_testset(
    pdf_path: str,
    testset_size: int,
    output_dir: Path,
) -> Path:
    """Generate a RAGAS testset and save it to disk."""
    # Load and chunk the PDF
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    # Build LLM and embeddings wrappers for RAGAS
    llm = build_llm()
    embeddings = build_embeddings()

    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Create the testset generator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    print(f"Generating {testset_size} test samples (this may take a while)...")
    testset = generator.generate_with_langchain_docs(
        chunks,
        testset_size=testset_size,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pandas DataFrame CSV
    df = testset.to_pandas()
    csv_path = output_dir / "testset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved testset CSV: {csv_path}")

    # Save as JSON for programmatic use
    json_path = output_dir / "testset.json"
    records = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved testset JSON: {json_path}")

    print(f"\nGenerated {len(df)} test samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample question: {df.iloc[0].get('user_input', df.iloc[0].get('question', 'N/A'))}")

    return json_path


def main():
    parser = argparse.ArgumentParser(description="Generate RAGAS evaluation testset from a PDF")
    parser.add_argument("--pdf", default=PDF_PATH, help=f"Path to PDF (default: {PDF_PATH})")
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help=f"Number of test samples (default: {TESTSET_SIZE})")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    generate_testset(
        pdf_path=args.pdf,
        testset_size=args.size,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
