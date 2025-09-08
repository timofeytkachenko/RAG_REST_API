import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gemma import GemmaEmbeddings

load_dotenv()

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("rag.utils")

# -----------------------
# Config
# -----------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "./faiss_data"))

STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

HASH_DB = STATE_DIR / "file_hashes.json"
COLLECTION = os.getenv("COLLECTION", "docs_rag")

FAISS_INDEX_PATH = FAISS_INDEX_DIR / f"{COLLECTION}.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / f"{COLLECTION}.pkl"

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")  # "openai" or "google"
HF_TOKEN = os.getenv("HF_TOKEN", "")

OPENAI_DIMENSION = 1536
GOOGLE_GEMMA_DIMENSION = 768

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
PROJECT_SERVICE_TOKEN = os.getenv("PROJECT_SERVICE_TOKEN", "")


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
        chunk_size=200,
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


def get_faiss_index_path() -> tuple[Path, Path]:
    """Get FAISS index and metadata file paths."""
    return FAISS_INDEX_PATH, FAISS_METADATA_PATH


def new_faiss_vectorstore(embeddings) -> FAISS:
    """
    Create or load FAISS vectorstore from disk.

    Parameters
    ----------
    embeddings : Embeddings
        Embedding model (OpenAI or Google).

    Returns
    -------
    FAISS
        Ready-to-use FAISS vector store.
    """
    index_path, metadata_path = get_faiss_index_path()

    if index_path.exists() and metadata_path.exists():
        try:
            logger.info("Loading existing FAISS index from %s", index_path)
            vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings,
                COLLECTION,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "FAISS index loaded successfully with %d vectors",
                vectorstore.index.ntotal,
            )
            return vectorstore
        except Exception as e:
            logger.warning(
                "Failed to load existing FAISS index: %s. Creating new one.", e
            )

    # Create new empty FAISS index
    logger.info("Creating new FAISS index")
    # Create with a dummy document to initialize the index
    dummy_doc = Document(
        page_content="Initialize FAISS index", metadata={"source": "init"}
    )
    vectorstore = FAISS.from_documents([dummy_doc], embeddings)

    # Save the index (will be empty after first real document ingestion)
    vectorstore.save_local(str(FAISS_INDEX_DIR), COLLECTION)
    logger.info("New FAISS index created and saved")

    return vectorstore


def new_chat_model():
    """Build the chat LLM (Qwen via OpenAI-compatible endpoint) from env config.

    Environment variables used:
    - API_BASE_URL: Provider base URL (will be normalized to end with '/v1')
    - API_KEY: Bearer token for Authorization header
    - MODEL_NAME: Model identifier to use (falls back to OPENAI_MODEL)
    - PROJECT_SERVICE_TOKEN: Optional additional header 'Grace-Client-Secret'
    - OPENAI_TEMPERATURE: Generation temperature (float)
    """
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is not set. Configure MODEL_NAME")

    logger.info("Initializing Chat LLM (Qwen): model=%s", MODEL_NAME)

    headers = {"Grace-Client-Secret": PROJECT_SERVICE_TOKEN}

    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=OPENAI_TEMPERATURE,
        api_key=API_KEY or None,
        base_url=API_BASE_URL,
        default_headers=headers,
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("source_path") or "unknown"
        header = f"[{src}]"
        chunks.append(header + "\n" + d.page_content)
    logger.debug("Formatting %d docs into context", len(docs))
    return "\n\n---\n\n".join(chunks)


def get_faiss_storage_info() -> dict:
    """Get information about FAISS storage directory and index."""
    index_path, metadata_path = get_faiss_index_path()

    storage_info = {
        "faiss_index_dir": str(FAISS_INDEX_DIR),
        "faiss_index_exists": index_path.exists(),
        "faiss_metadata_exists": metadata_path.exists(),
        "faiss_index_size_bytes": 0,
        "current_embed_provider": EMBED_PROVIDER,
        "expected_dimensions": get_expected_dimension(),
        "collection_name": COLLECTION,
        "index_info": {},
    }

    if FAISS_INDEX_DIR.exists():
        # Calculate total size
        storage_info["faiss_index_size_bytes"] = sum(
            f.stat().st_size for f in FAISS_INDEX_DIR.rglob("*") if f.is_file()
        )

    # Try to get index info
    try:
        if index_path.exists() and metadata_path.exists():
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings,
                COLLECTION,
                allow_dangerous_deserialization=True,
            )

            storage_info["index_info"] = {
                "name": COLLECTION,
                "vectors_count": vectorstore.index.ntotal,
                "dimensions": vectorstore.index.d,
                "dimension_match": vectorstore.index.d == get_expected_dimension(),
                "index_type": type(vectorstore.index).__name__,
            }
        else:
            storage_info["index_info"] = {
                "error": "Index files not found",
                "index_exists": False,
            }
    except Exception as e:
        storage_info["index_info"] = {
            "error": f"Could not load index info: {str(e)}",
            "index_exists": False,
        }

    return storage_info


def ensure_faiss_directories() -> None:
    """Ensure FAISS index directory exists."""
    try:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("FAISS index directory ensured: %s", FAISS_INDEX_DIR)
    except OSError as e:
        logger.warning("Could not create FAISS directories: %s", e)
