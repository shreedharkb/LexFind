"""
LexFind — Local GPU Embedder (Replaces kaggle_embedder.ipynb)
==============================================================
Uses your local NVIDIA RTX 3050 GPU to embed all chunks exported
by export_for_kaggle.py.

Reads:   scripts/qdrant_ingestion/kaggle_export/chunks_for_embedding_*.jsonl
Writes:  scripts/qdrant_ingestion/embeddings.npy
         scripts/qdrant_ingestion/chunk_ids.json

Run from backend/:
    python scripts/qdrant_ingestion/local_gpu_embedder.py
    python scripts/qdrant_ingestion/local_gpu_embedder.py --batch-size 128
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Redirect HuggingFace cache to D: drive to avoid filling C: drive
os.environ["HF_HOME"] = "D:/LexFind/.cache/huggingface"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from tqdm import tqdm

MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE  = 256   # RTX 3050 (4GB) handles 256 comfortably at 512 token limit

OUTPUT_DIR  = BASE_DIR / "scripts" / "qdrant_ingestion"
INPUT_GLOB  = str(BASE_DIR / "scripts" / "qdrant_ingestion" / "kaggle_export" / "*.jsonl")

EMB_PATH    = OUTPUT_DIR / "embeddings.npy"
IDS_PATH    = OUTPUT_DIR / "chunk_ids.json"


def parse_args():
    p = argparse.ArgumentParser(description="Local GPU Embedder for LexFind")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--input-glob", type=str, default=INPUT_GLOB)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--max-chunks", type=int, default=0, help="Limit max chunks to embed")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU check
    try:
        import torch
        torch.set_num_threads(2)  # Limit CPU threads so Windows UI stays responsive
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU detected: {props.name} - {props.total_memory / 1024**3:.1f} GB VRAM")
        else:
            print("No CUDA GPU found - running on CPU (will be slow)")
    except ImportError:
        device = "cpu"
        print("torch not installed - running on CPU")

    # Load JSONL chunks
    jsonl_files = sorted(glob.glob(args.input_glob, recursive=True))
    if not jsonl_files:
        print(f"No .jsonl files found at: {args.input_glob}")
        print("Run export_for_kaggle.py first.")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL file(s):")
    chunks = []
    for fpath in jsonl_files:
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {fpath}  ({size_mb:.1f} MB)")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
                if args.max_chunks > 0 and len(chunks) >= args.max_chunks:
                    break
        if args.max_chunks > 0 and len(chunks) >= args.max_chunks:
            break

    total = len(chunks)
    print(f"Total chunks loaded: {total:,}")
    if total == 0:
        print("No chunks found! Check your export files.")
        sys.exit(1)

    texts     = [c["chunk_text"] for c in chunks]
    chunk_ids = [c["chunk_id"]   for c in chunks]
    print(f"Sample text  : {texts[0][:120]}...")
    print(f"Sample ID    : {chunk_ids[0]}")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dim : {dim}")
    print(f"Max seq len   : {model.max_seq_length}")
    print(f"Device        : {device}")

    # Embed
    print(f"Embedding {total:,} chunks (batch={args.batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )

    print(f"Embedding complete!")
    print(f"Shape : {embeddings.shape}")
    print(f"Dtype : {embeddings.dtype}")

    sample_norms = np.linalg.norm(embeddings[:100], axis=1)
    print(f"Norms (first 100): min={sample_norms.min():.4f}  max={sample_norms.max():.4f}")

    # Save outputs
    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "chunk_ids.json"

    np.save(str(emb_path), embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    emb_mb = emb_path.stat().st_size / 1024 / 1024
    ids_mb = ids_path.stat().st_size / 1024 / 1024

    print(f"Saved:")
    print(f"  {emb_path}  ({emb_mb:.1f} MB)")
    print(f"  {ids_path}  ({ids_mb:.1f} MB)")
    print(f"Next step:")
    print(f"  python scripts/hybrid/recreate_collection.py")
    print(f"  python scripts/hybrid/hybrid_ingestor.py")


if __name__ == "__main__":
    main()
