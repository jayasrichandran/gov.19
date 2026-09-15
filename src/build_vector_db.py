"""
Vector Database Builder for GOV-19 Knowledge Base.

Loads document chunks from data/processed/chunks.json,
computes dense embeddings using sentence-transformers (all-MiniLM-L6-v2),
builds a FAISS vector index (Cosine Similarity via IndexFlatIP with L2 normalization),
and saves the vector index and chunk metadata in data/vector_db/.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output encoding in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def build_vector_database(
    chunks_file: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """
    Build FAISS vector index and save with metadata.
    """
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_file}")

    print("\n" + "=" * 75)
    print("                GOV-19 VECTOR DATABASE BUILDER")
    print("=" * 75)
    print(f"Loading chunks from: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks: List[Dict] = json.load(f)

    if not chunks:
        raise ValueError("Chunks file is empty!")

    print(f"Total chunks to embed: {len(chunks)}")
    print(f"Loading embedding model: {model_name}...")

    t0 = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f} seconds.")

    # Prepare texts for embedding with scheme context
    texts_to_embed = [
        f"Scheme: {c['scheme']}\n{c['text']}" for c in chunks
    ]

    print("Generating embeddings...")
    t1 = time.time()
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Unit length for cosine similarity
    )
    embed_time = time.time() - t1
    print(f"Embeddings generated in {embed_time:.2f} seconds.")
    print(f"Embeddings shape: {embeddings.shape} (dtype: {embeddings.dtype})")

    dimension = embeddings.shape[1]

    # Create FAISS Inner-Product index (exact cosine similarity because vectors are unit-normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index created with {index.ntotal} vectors of dimension {dimension}.")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "faiss_index.bin"
    metadata_file = output_dir / "chunks_metadata.json"

    # Save FAISS index
    faiss.write_index(index, str(index_file))
    print(f"FAISS index saved to: {index_file}")

    # Save chunks metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunk metadata saved to: {metadata_file}")

    print("-" * 75)
    print("Vector database build complete!")
    print(f"  • Vectors: {index.ntotal}")
    print(f"  • Dimensions: {dimension}")
    print(f"  • Model: {model_name}")
    print(f"  • Location: {output_dir}")
    print("=" * 75 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    default_chunks = base_dir / "data" / "processed" / "chunks.json"
    default_output = base_dir / "data" / "vector_db"

    parser = argparse.ArgumentParser(description="Build FAISS vector database for GOV-19.")
    parser.add_argument(
        "--chunks-file",
        "-c",
        type=Path,
        default=default_chunks,
        help=f"Path to chunks.json (default: {default_chunks})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_output,
        help=f"Directory to save FAISS index & metadata (default: {default_output})",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )

    args = parser.parse_args()
    build_vector_database(
        chunks_file=args.chunks_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
