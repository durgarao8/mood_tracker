from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the locally saved embedding model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {MODEL_PATH}"
        )

    return SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )


def load_chroma_client():
    """Load the existing local ChromaDB."""

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB not found: {CHROMA_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client


def list_collections():
    """Display collections stored in ChromaDB."""

    client = load_chroma_client()

    collections = client.list_collections()

    print("\nChromaDB collections:")

    for collection in collections:
        print(f"- {collection.name}")

    return collections


def get_collection(collection_name):
    """Load an existing ChromaDB collection."""

    client = load_chroma_client()

    try:
        collection = client.get_collection(
            name=collection_name
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load collection '{collection_name}'. "
            f"Available collections: "
            f"{[c.name for c in client.list_collections()]}"
        ) from e

    return collection