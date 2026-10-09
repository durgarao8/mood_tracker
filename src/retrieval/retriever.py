from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"

COLLECTION_NAME = "langchain"


def load_retriever():
    """
    Load the local embedding model and existing ChromaDB collection.
    """

    # Load local embedding model
    embedding_model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )

    # Load existing ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return embedding_model, collection


def retrieve(query: str, top_k: int = 5):
    """
    Retrieve the most relevant DSA documents.
    """

    embedding_model, collection = load_retriever()

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results