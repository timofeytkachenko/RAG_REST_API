from __future__ import annotations
import os, io, hashlib, json, glob
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Embeddings & LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Vector stores
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

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

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "google/embeddinggemma-300m")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))


# -----------------------
# Utilities
# -----------------------
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
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )

def get_embeddings():
    if EMBED_PROVIDER.lower() == "hf":
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    # default OpenAI
    return OpenAIEmbeddings()

def get_vectorstore() -> QdrantVectorStore:
    # Ensure Qdrant storage directories exist
    ensure_qdrant_directories()
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Check if collection exists, create if not
    try:
        client.get_collection(COLLECTION)
    except Exception:
        # Collection doesn't exist, create it
        from qdrant_client.models import Distance, VectorParams
        embeddings = get_embeddings()
        # Get embedding dimension by creating a test embedding
        test_embedding = embeddings.embed_query("test")
        vector_size = len(test_embedding)
        
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=get_embeddings(),
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
    }
    
    if QDRANT_DATA_DIR.exists():
        # Calculate total size
        storage_info["qdrant_data_size_bytes"] = sum(
            f.stat().st_size for f in QDRANT_DATA_DIR.rglob('*') if f.is_file()
        )
        
        collections_dir = QDRANT_DATA_DIR / "collections"
        snapshots_dir = QDRANT_DATA_DIR / "snapshots"
        
        storage_info["collections_dir_exists"] = collections_dir.exists()
        storage_info["snapshots_dir_exists"] = snapshots_dir.exists()
        
        if collections_dir.exists():
            storage_info["collection_count"] = len([
                d for d in collections_dir.iterdir() if d.is_dir()
            ])
    
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

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="RAG Microservice (FastAPI + LangChain + Qdrant)")


# -----------------------
# Schemas
# -----------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="User query/question")
    search_type: Literal["mmr", "similarity", "similarity_score_threshold"] = "similarity"
    k: int = 6
    fetch_k: int = 20
    lambda_mult: float = 0.5
    # for similarity_score_threshold
    score_threshold: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    used_docs: List[str] = []

class IngestScanResponse(BaseModel):
    added_chunks: int
    processed_files: int

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}

@app.get("/storage/info")
def storage_info():
    """Get information about Qdrant storage and data persistence."""
    info = get_qdrant_storage_info()
    info.update({
        "data_dir": str(DATA_DIR),
        "state_dir": str(STATE_DIR),
        "collection_name": COLLECTION,
        "qdrant_url": QDRANT_URL,
    })
    return info

@app.post("/ingest/upload", response_model=IngestScanResponse)
async def ingest_upload(
    files: List[UploadFile] = File(..., description="PDF/TXT files"),
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splitter = build_splitter()
    vectorstore = get_vectorstore()
    hashes = read_hash_db()

    docs_to_add: List[Document] = []
    processed = 0

    for uf in files:
        suffix = Path(uf.filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            continue

        content = await uf.read()
        fhash = compute_hash_bytes(content)
        target = DATA_DIR / uf.filename

        # only if new or changed
        if hashes.get(str(target)) == fhash:
            continue

        # save file
        with open(target, "wb") as out:
            out.write(content)

        # load and split
        docs = load_docs_from_path(target)
        for d in docs:
            meta = (d.metadata or {})
            meta.update({"source_path": str(target)})
            d.metadata = meta
        chunks = splitter.split_documents(docs)
        docs_to_add.extend(chunks)

        hashes[str(target)] = fhash
        processed += 1

    if docs_to_add:
        vectorstore.add_documents(docs_to_add)
        write_hash_db(hashes)

    return IngestScanResponse(added_chunks=len(docs_to_add), processed_files=processed)

@app.post("/ingest/scan", response_model=IngestScanResponse)
def ingest_scan():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splitter = build_splitter()
    vectorstore = get_vectorstore()
    hashes = read_hash_db()

    candidates = [
        Path(p) for p in glob.glob(str(DATA_DIR / "**/*"), recursive=True)
        if p.lower().endswith((".pdf", ".txt"))
    ]

    docs_to_add: List[Document] = []
    processed = 0

    for p in candidates:
        h = compute_hash_file(p)
        if hashes.get(str(p)) == h:
            continue

        docs = load_docs_from_path(p)
        for d in docs:
            meta = (d.metadata or {})
            meta.update({"source_path": str(p)})
            d.metadata = meta
        chunks = splitter.split_documents(docs)
        docs_to_add.extend(chunks)

        hashes[str(p)] = h
        processed += 1

    if docs_to_add:
        vectorstore.add_documents(docs_to_add)
        write_hash_db(hashes)

    return IngestScanResponse(added_chunks=len(docs_to_add), processed_files=processed)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    vectorstore = get_vectorstore()

    # Configure retriever
    if req.search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": req.k, "fetch_k": req.fetch_k, "lambda_mult": req.lambda_mult},
        )
    elif req.search_type == "similarity":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": req.k},
        )
    else:  # "similarity_score_threshold"
        kwargs = {"k": req.k}
        if req.score_threshold is not None:
            kwargs["score_threshold"] = req.score_threshold
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=kwargs,
        )

    system_prompt = (
        "You are an assistant that responds strictly based on the context from documents. "
        "If there is no information, say that it's not available and suggest where to look for it. "
        "At the end of your response, indicate the sources (file name/path)."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    llm = build_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # (optional) list of source paths
    docs = retriever.invoke(req.question)
    used = []
    for d in docs:
        used.append(d.metadata.get("source_path") or d.metadata.get("source") or "unknown")

    ans = chain.invoke(req.question)
    text = ans.content if hasattr(ans, "content") else str(ans)
    return QueryResponse(answer=text, used_docs=used)