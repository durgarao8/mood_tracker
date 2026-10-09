from pathlib import Path
from sentence_transformers import SentenceTransformer


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local embedding model
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the locally saved SentenceTransformer model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Embedding model not found at: {MODEL_PATH}"
        )

    print(f"Loading embedding model from: {MODEL_PATH}")

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )

    print("Embedding model loaded successfully.")

    return model


def generate_embedding(text: str):
    """
    Generate an embedding for a single text.
    """

    model = load_embedding_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding