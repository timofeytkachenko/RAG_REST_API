# utils.py
from __future__ import annotations

import hashlib
import json
import logging
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
# Logging
# -----------------------
logger = logging.getLogger("rag.utils")

# -----------------------
# Config
# -----------------------
API_TOKEN = os.getenv("API_TOKEN", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
QDRANT_DATA_DIR = Path(os.getenv("QDRANT_DATA_DIR", "./qdrant_data"))

STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DATA_DIR.mkdir(parents=True, exist_ok=True)

HASH_DB = STATE_DIR / "file_hashes.json"
COLLECTION = os.getenv("COLLECTION", "docs_rag")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")  # "openai" or "google"
HF_TOKEN = os.getenv("HF_TOKEN", "")

OPENAI_DIMENSION = 1536
GOOGLE_GEMMA_DIMENSION = 768

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))


def compute_hash_bytes(data: bytes) -> str:
    """Return SHA-256 for given bytes (stream-safe)."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_hash_file(path: Path) -> str:
    """Return SHA-256 for file at `path` using 1MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hash_db() -> dict:
    """Load file-hash DB from STATE_DIR; return empty dict if absent."""
    if HASH_DB.exists():
        try:
            data = json.loads(HASH_DB.read_text())
            logger.debug("Loaded hash DB with %d entries", len(data))
            return data
        except Exception:
            logger.exception("Failed to read HASH_DB; returning empty dict")
            return {}
    return {}


def write_hash_db(d: dict) -> None:
    """Persist file-hash DB to STATE_DIR with pretty formatting."""
    try:
        HASH_DB.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        logger.debug("Wrote hash DB with %d entries", len(d))
    except Exception:
        logger.exception("Failed to write HASH_DB")


def load_docs_from_path(path: Path) -> List[Document]:
    """Load Text/PDF into langchain Documents with metadata."""
    try:
        if path.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            logger.info("Loaded PDF: %s -> %d pages", path.name, len(docs))
            return docs
        if path.suffix.lower() == ".txt":
            docs = TextLoader(str(path), autodetect_encoding=True).load()
            logger.info("Loaded TXT: %s -> %d chunk(s)", path.name, len(docs))
            return docs
        logger.warning("Unsupported extension for load: %s", path.name)
        return []
    except Exception:
        logger.exception("Failed to load document: %s", path)
        return []


def new_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a robust recursive splitter for mixed prose/code."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    logger.debug("Initialized text splitter: chunk_size=500 overlap=50")
    return splitter


def get_expected_dimension() -> int:
    """Return expected embedding dimensionality based on provider."""
    provider = EMBED_PROVIDER.lower()
    if provider == "google":
        return GOOGLE_GEMMA_DIMENSION
    if provider == "openai":
        return OPENAI_DIMENSION
    raise ValueError(
        f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}. Use 'openai' or 'google'"
    )


def get_embeddings():
    """
    Instantiate embedding model by provider.
    """
    provider = EMBED_PROVIDER.lower()
    logger.info("Initializing embeddings: provider=%s", provider)
    if provider == "google":
        return GemmaEmbeddings()
    if provider == "openai":
        return OpenAIEmbeddings()
    raise ValueError(
        f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}. Use 'openai' or 'google'"
    )


def new_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for the configured URL/API key."""
    logger.info("Creating Qdrant client: url=%s", QDRANT_URL)
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def new_qdrant_vectorstore(
    client: QdrantClient, embeddings, expected_dim: int
) -> QdrantVectorStore:
    """
    Ensure collection exists with correct dimensionality and return a VectorStore.
    """
    from qdrant_client.models import Distance, VectorParams

    try:
        info = client.get_collection(COLLECTION)
        existing_dim = info.config.params.vectors.size
        logger.info("Found collection '%s' with dim=%d", COLLECTION, existing_dim)
        if existing_dim != expected_dim:
            logger.warning(
                "Dimension mismatch: existing=%d expected=%d; recreating",
                existing_dim,
                expected_dim,
            )
            try:
                client.delete_collection(COLLECTION)
                client.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=VectorParams(
                        size=expected_dim, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    "Recreated collection '%s' with dim=%d", COLLECTION, expected_dim
                )
            except Exception as e:
                logger.exception("Failed to recreate collection '%s'", COLLECTION)
                raise ValueError(
                    "Cannot recreate collection: dimension mismatch persists. "
                    "Use a different COLLECTION (e.g., COLLECTION=docs_openai) "
                    "or clear qdrant_data directory."
                ) from e
    except Exception:
        logger.info(
            "Creating new collection '%s' with dim=%d", COLLECTION, expected_dim
        )
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE),
        )

    logger.info("Initializing QdrantVectorStore on '%s'", COLLECTION)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        distance="Cosine",
    )


def new_chat_model():
    """Build the chat LLM from environment config."""
    logger.info(
        "Initializing Chat LLM: model=%s temp=%.2f", OPENAI_MODEL, OPENAI_TEMPERATURE
    )
    return ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("source_path") or "unknown"
        header = f"[{src}]"
        chunks.append(header + "\n" + d.page_content)
    logger.debug("Formatting %d docs into context", len(docs))
    return "\n\n---\n\n".join(chunks)


def get_qdrant_storage_info() -> dict:
    """Inspect Qdrant data dir size and remote collection state."""
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
        size = sum(f.stat().st_size for f in QDRANT_DATA_DIR.rglob("*") if f.is_file())
        storage_info["qdrant_data_size_bytes"] = size
        collections_dir = QDRANT_DATA_DIR / "collections"
        snapshots_dir = QDRANT_DATA_DIR / "snapshots"
        storage_info["collections_dir_exists"] = collections_dir.exists()
        storage_info["snapshots_dir_exists"] = snapshots_dir.exists()
        if collections_dir.exists():
            storage_info["collection_count"] = len(
                [d for d in collections_dir.iterdir() if d.is_dir()]
            )
        logger.debug(
            "Qdrant data: size=%dB collections_dir=%s snapshots_dir=%s",
            size,
            collections_dir.exists(),
            snapshots_dir.exists(),
        )

    try:
        client = new_qdrant_client()
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
        logger.info("Fetched collection info: %s", storage_info["collection_info"])
    except Exception as e:
        storage_info["collection_info"] = {
            "error": f"Could not retrieve collection info: {str(e)}",
            "collection_exists": False,
        }
        logger.warning("Could not retrieve collection info: %s", e)

    return storage_info


def ensure_qdrant_directories() -> None:
    """Ensure Qdrant data subdirs exist; suppress errors (Qdrant may create them)."""
    collections_dir = QDRANT_DATA_DIR / "collections"
    snapshots_dir = QDRANT_DATA_DIR / "snapshots"
    try:
        collections_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured qdrant directories exist")
    except OSError as e:
        logger.warning("Could not create Qdrant directories: %s", e)
