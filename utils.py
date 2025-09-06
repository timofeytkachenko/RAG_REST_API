import hashlib
import json
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from gemma import GemmaEmbeddings

# -----------------------
# Config
# -----------------------
API_TOKEN = os.getenv("API_TOKEN", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
QDRANT_DATA_DIR = Path(os.getenv("QDRANT_DATA_DIR", "./qdrant_data"))

# Create directories
STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DATA_DIR.mkdir(parents=True, exist_ok=True)

HASH_DB = STATE_DIR / "file_hashes.json"
COLLECTION = os.getenv("COLLECTION", "docs_rag")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")  # "openai" or "google"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model dimensions
OPENAI_DIMENSION = 1536
GOOGLE_GEMMA_DIMENSION = 768

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))


def compute_hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hash_db() -> dict:
    if HASH_DB.exists():
        return json.loads(HASH_DB.read_text())
    return {}


def write_hash_db(d: dict):
    HASH_DB.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def load_docs_from_path(path: Path) -> List[Document]:
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif path.suffix.lower() == ".txt":
        return TextLoader(str(path), autodetect_encoding=True).load()
    else:
        return []


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )


def get_expected_dimension() -> int:
    """Get expected embedding dimension based on provider."""
    if EMBED_PROVIDER.lower() == "google":
        return GOOGLE_GEMMA_DIMENSION
    elif EMBED_PROVIDER.lower() == "openai":
        return OPENAI_DIMENSION
    else:
        raise ValueError(
            f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}. Use 'openai' or 'google'"
        )


def get_embeddings():
    """Get embeddings provider: either OpenAI or Google Embedding Gemma."""
    if EMBED_PROVIDER.lower() == "google":
        return GemmaEmbeddings()
    elif EMBED_PROVIDER.lower() == "openai":
        return OpenAIEmbeddings()
    else:
        raise ValueError(
            f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}. Use 'openai' or 'google'"
        )


def get_vectorstore() -> QdrantVectorStore:
    # Ensure Qdrant storage directories exist
    ensure_qdrant_directories()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = get_embeddings()
    expected_dim = get_expected_dimension()

    # Check if collection exists and validate dimensions
    try:
        collection_info = client.get_collection(COLLECTION)
        existing_dim = collection_info.config.params.vectors.size

        if existing_dim != expected_dim:
            print(
                f"⚠️  Collection '{COLLECTION}' exists with {existing_dim} dimensions, but {EMBED_PROVIDER} embeddings need {expected_dim} dimensions."
            )
            print(f"🔄 Attempting to recreate collection with correct dimensions...")
            print(
                f"💡 If this fails due to permissions, try using a different COLLECTION name in your .env file."
            )

            try:
                # Delete existing collection
                client.delete_collection(COLLECTION)
                print(f"🗑️  Old collection deleted successfully")

                # Create new collection with correct dimensions
                from qdrant_client.models import Distance, VectorParams

                client.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=VectorParams(
                        size=expected_dim, distance=Distance.COSINE
                    ),
                )
                print(
                    f"✅ Collection '{COLLECTION}' recreated with {expected_dim} dimensions"
                )

            except Exception as recreate_error:
                error_msg = str(recreate_error)
                print(f"❌ Failed to recreate collection: {error_msg}")

                raise ValueError(
                    f"Cannot recreate collection. "
                    f"Please use a different collection name (e.g., COLLECTION=docs_{EMBED_PROVIDER}) "
                    f"or manually delete the qdrant_data directory."
                )
        else:
            print(
                f"✅ Collection '{COLLECTION}' already exists with correct dimensions ({existing_dim})"
            )

    except Exception as e:
        # Collection doesn't exist, create it
        print(
            f"📄 Creating new collection '{COLLECTION}' with {expected_dim} dimensions for {EMBED_PROVIDER} embeddings..."
        )
        from qdrant_client.models import Distance, VectorParams

        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{COLLECTION}' created successfully")

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        distance="Cosine",
    )


def build_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


def format_docs(docs: List[Document]) -> str:
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("source_path") or "unknown"
        header = f"[{src}]"
        chunks.append(header + "\n" + d.page_content)
    return "\n\n---\n\n".join(chunks)


def get_qdrant_storage_info() -> dict:
    """Get information about Qdrant storage directory and collections."""
    storage_info = {
        "qdrant_data_dir": str(QDRANT_DATA_DIR),
        "qdrant_data_exists": QDRANT_DATA_DIR.exists(),
        "qdrant_data_size_bytes": 0,
        "collections_dir_exists": False,
        "snapshots_dir_exists": False,
        "collection_count": 0,
        "current_embed_provider": EMBED_PROVIDER,
        "expected_dimensions": get_expected_dimension(),
        "collection_info": {},
    }

    if QDRANT_DATA_DIR.exists():
        # Calculate total size
        storage_info["qdrant_data_size_bytes"] = sum(
            f.stat().st_size for f in QDRANT_DATA_DIR.rglob("*") if f.is_file()
        )

        collections_dir = QDRANT_DATA_DIR / "collections"
        snapshots_dir = QDRANT_DATA_DIR / "snapshots"

        storage_info["collections_dir_exists"] = collections_dir.exists()
        storage_info["snapshots_dir_exists"] = snapshots_dir.exists()

        if collections_dir.exists():
            storage_info["collection_count"] = len(
                [d for d in collections_dir.iterdir() if d.is_dir()]
            )

    # Try to get collection info from Qdrant
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collection_info = client.get_collection(COLLECTION)

        storage_info["collection_info"] = {
            "name": COLLECTION,
            "status": collection_info.status.name,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "dimensions": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance.name,
            "dimension_match": collection_info.config.params.vectors.size
            == get_expected_dimension(),
        }
    except Exception as e:
        storage_info["collection_info"] = {
            "error": f"Could not retrieve collection info: {str(e)}",
            "collection_exists": False,
        }

    return storage_info


def ensure_qdrant_directories():
    """Ensure all required Qdrant directories exist."""
    collections_dir = QDRANT_DATA_DIR / "collections"
    snapshots_dir = QDRANT_DATA_DIR / "snapshots"

    try:
        collections_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Log the error but don't crash - Qdrant might create these itself
        print(f"Warning: Could not create Qdrant directories: {e}")
        pass
