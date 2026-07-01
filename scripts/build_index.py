"""
Build the embedding index from the catalog.

Computes sentence embeddings for all catalog entries and saves them
as a numpy .npz file. With ~600 entries, this takes under a minute
and the resulting file is small enough to version-control.

Run once before starting the server:
    python scripts/build_index.py
"""

import sys
import os
import time

# allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, EMBEDDINGS_PATH
from app.catalog import CatalogStore


def build_index():
    print(f"Loading catalog...")
    store = CatalogStore()
    print(f"Loaded {len(store.entries)} catalog entries.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # each document is "name | description" -- same text the retrieval
    # layer will query against
    documents = []
    entity_ids = []

    for entry in store.entries:
        doc_text = f"{entry.name} | {entry.description}"
        documents.append(doc_text)
        entity_ids.append(entry.entity_id)

    print(f"Computing embeddings for {len(documents)} documents...")
    start_time = time.time()

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # pre-normalize for cosine similarity via dot product
    )

    elapsed = time.time() - start_time
    print(f"Embeddings computed in {elapsed:.1f}s")
    print(f"Shape: {embeddings.shape}")

    # save as compressed numpy archive
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        entity_ids=np.array(entity_ids, dtype=str),
    )

    file_size = os.path.getsize(EMBEDDINGS_PATH) / (1024 * 1024)
    print(f"Saved to {EMBEDDINGS_PATH} ({file_size:.1f} MB)")


if __name__ == "__main__":
    build_index()
