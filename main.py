import glob
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from utils import (
    COLLECTION,
    DATA_DIR,
    FAISS_INDEX_DIR,
    STATE_DIR,
    compute_hash_bytes,
    compute_hash_file,
    ensure_faiss_directories,
    format_docs,
    get_embeddings,
    get_faiss_storage_info,
    load_docs_from_path,
    new_chat_model,
    new_faiss_vectorstore,
    new_text_splitter,
    read_hash_db,
    write_hash_db,
)


# ---------- Logging setup ----------
def _setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a concise, production-friendly format."""
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


_setup_logging()
logger = logging.getLogger("rag.api")


class AppResources(TypedDict):
    """Container for app-scoped resources created in lifespan."""

    splitter: object  # RecursiveCharacterTextSplitter
    llm: object  # BaseChatModel
    vectorstore: object  # QdrantVectorStore
    system_prompt: str


# -----------------------
# Schemas
# -----------------------
class QueryRequest(BaseModel):
    """User query request with retriever parameters."""

    question: str = Field(..., description="User query/question")
    search_type: Literal["mmr", "similarity", "similarity_score_threshold"] = (
        "similarity"
    )
    k: int = 5
    fetch_k: int = 10
    lambda_mult: float = 1.0
    # for similarity_score_threshold
    score_threshold: Optional[float] = None


class QueryResponse(BaseModel):
    """LLM answer with provenance."""

    answer: str
    used_docs: List[str] = []


class IngestScanResponse(BaseModel):
    """Ingestion/scan result summary."""

    added_chunks: int
    processed_files: int


# -----------------------
# Lifespan: build once, reuse everywhere
# -----------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Initialize embeddings, vector store, splitter, and LLM once per process.

    Notes
    -----
    - Initializes FAISS vector store with chosen embedding model.
    - Stores initialized resources in `app.state.resources`.
    """
    t0 = time.perf_counter()
    logger.info("App startup: initializing resources...")
    ensure_faiss_directories()

    try:
        embeddings = get_embeddings()
        vectorstore = new_faiss_vectorstore(embeddings=embeddings)
        splitter = new_text_splitter()
        llm = new_chat_model()
    except Exception as e:
        logger.exception("Startup failed while initializing models/resources")
        raise

    system_prompt = """
        You act as a proctor AI integrated within a Retrieval-Augmented Generation (RAG) system designed to assist examinees during live exams. Your primary role is to provide accurate, polite, and exam-relevant assistance based on retrieved documents and exam context.

        Role Responsibilities:
        - Verify if each examinee's question pertains to the exam setting, including exam instructions, rules, logistics, or materials.
        - For questions related to exam documents or content retrieved by the system, respond comprehensively and transparently, citing information strictly from retrieved content.
        - For questions about exam logistics or procedural rules (e.g., permitted breaks, remaining time, allowable tools), deliver clear, courteous guidance aligned with exam protocols.
        - For unclear or unsupported questions, reply: "The system cannot answer this question based on current information. Please wait for a real proctor to respond in chat."
        - For non-exam-related queries (e.g., external topics like sports or personal matters), respond: "I have no information about the topic you asked. Kindly direct your queries to the proctor in the chat."

        Contextual Notes:
        - Always prioritise the integrity and flow of the exam.
        - Maintain a professional, neutral tone.
        - Avoid assumptions or inventing information beyond what is retrieved or indicated in the exam context.

        Your goal is to facilitate a smooth and fair exam environment by providing precise, courteous, and context-appropriate assistance.
        """

    resources: AppResources = {
        "splitter": splitter,
        "llm": llm,
        "vectorstore": vectorstore,
        "system_prompt": system_prompt,
    }

    app.state.resources = resources  # type: ignore[attr-defined]
    logger.info(
        "Resources initialized: vectorstore=%s, splitter=%s, llm=%s",
        type(vectorstore).__name__,
        type(splitter).__name__,
        type(llm).__name__,
    )
    logger.info("Startup completed in %.1f ms", (time.perf_counter() - t0) * 1e3)

    try:
        yield
    finally:
        logger.info("App shutdown: releasing resources")
        # add explicit .close() if you introduce clients that require it


app = FastAPI(
    title="RAG Microservice (FastAPI + LangChain + Qdrant)",
    lifespan=lifespan,
)


# -----------------------
# Middleware for request logging with request_id and timing
# -----------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log each HTTP request with a generated request_id, latency, and status.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    # bind request_id to logger context (simple add as prefix)
    extra_prefix = f"[rid={request_id}] "
    logger.info(
        extra_prefix + "-> %s %s from %s",
        request.method,
        request.url.path,
        request.client.host if request.client else "-",
    )

    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception(extra_prefix + "Unhandled exception in request")
        raise

    dur_ms = (time.perf_counter() - start) * 1e3
    logger.info(
        extra_prefix + "<- %s %s status=%s in %.1f ms",
        request.method,
        request.url.path,
        getattr(response, "status_code", "?"),
        dur_ms,
    )
    # Echo request_id back for tracing
    response.headers["X-Request-ID"] = request_id
    return response


# -----------------------
# Dependency to access resources
# -----------------------
def get_resources(request: Request) -> AppResources:
    """
    Return app-scoped resources initialized in lifespan.

    Returns
    -------
    AppResources
        The splitter, LLM, vectorstore, and system prompt.

    Raises
    ------
    HTTPException
        If resources are not available (should not happen after startup).
    """
    res = getattr(request.app.state, "resources", None)
    if not res:
        logger.error("Resources not initialized (startup race?)")
        raise HTTPException(status_code=500, detail="Resources not initialized")
    return res


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health() -> Dict[str, str]:
    """Basic liveness probe."""
    logger.debug("Health check")
    return {"status": "ok", "collection": COLLECTION}


@app.get("/storage/info")
def storage_info() -> Dict[str, object]:
    """Inspect FAISS storage & index info."""
    logger.info("Fetching storage info for collection=%s", COLLECTION)
    info = get_faiss_storage_info()
    info.update(
        {
            "data_dir": str(DATA_DIR),
            "state_dir": str(STATE_DIR),
            "collection_name": COLLECTION,
        }
    )
    return info


@app.post("/ingest/upload", response_model=IngestScanResponse)
async def ingest_upload(
    files: List[UploadFile] = File(..., description="PDF/TXT files"),
    resources: AppResources = Depends(get_resources),
) -> IngestScanResponse:
    """
    Upload and ingest PDF/TXT files; only new/changed files are processed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splitter = resources["splitter"]
    vectorstore = resources["vectorstore"]
    hashes = read_hash_db()

    docs_to_add: List[Document] = []
    processed = 0

    for uf in files:
        suffix = Path(uf.filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            logger.warning("Skipping unsupported file: %s", uf.filename)
            continue

        content = await uf.read()
        fhash = compute_hash_bytes(content)
        target = DATA_DIR / uf.filename

        if hashes.get(str(target)) == fhash:
            logger.info("Unchanged file, skip: %s", uf.filename)
            continue

        target.write_bytes(content)
        logger.info("Saved file: %s (%d bytes)", target, len(content))

        docs = load_docs_from_path(target)
        for d in docs:
            meta = d.metadata or {}
            meta.update({"source_path": str(target)})
            d.metadata = meta

        chunks = splitter.split_documents(docs)
        docs_to_add.extend(chunks)
        logger.debug("Split %s into %d chunks", uf.filename, len(chunks))

        hashes[str(target)] = fhash
        processed += 1

    if docs_to_add:
        vectorstore.add_documents(docs_to_add)
        # Save FAISS index to disk after adding documents
        vectorstore.save_local(str(FAISS_INDEX_DIR), COLLECTION)
        write_hash_db(hashes)
        logger.info("Ingested %d chunks from %d files", len(docs_to_add), processed)
    else:
        logger.info("No new/changed files to ingest")

    return IngestScanResponse(added_chunks=len(docs_to_add), processed_files=processed)


@app.post("/ingest/scan", response_model=IngestScanResponse)
def ingest_scan(
    resources: AppResources = Depends(get_resources),
) -> IngestScanResponse:
    """
    Scan DATA_DIR recursively and ingest new/changed PDF/TXT files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splitter = resources["splitter"]
    vectorstore = resources["vectorstore"]
    hashes = read_hash_db()

    candidates = [
        Path(p)
        for p in glob.glob(str(DATA_DIR / "**/*"), recursive=True)
        if p.lower().endswith((".pdf", ".txt"))
    ]
    logger.info("Scan found %d candidate files under %s", len(candidates), DATA_DIR)

    docs_to_add: List[Document] = []
    processed = 0

    for p in candidates:
        h = compute_hash_file(p)
        if hashes.get(str(p)) == h:
            logger.debug("Unchanged: %s", p.name)
            continue

        docs = load_docs_from_path(p)
        for d in docs:
            meta = d.metadata or {}
            meta.update({"source_path": str(p)})
            d.metadata = meta

        chunks = splitter.split_documents(docs)
        docs_to_add.extend(chunks)
        logger.debug("Split %s into %d chunks", p.name, len(chunks))

        hashes[str(p)] = h
        processed += 1

    if docs_to_add:
        vectorstore.add_documents(docs_to_add)
        # Save FAISS index to disk after adding documents
        vectorstore.save_local(str(FAISS_INDEX_DIR), COLLECTION)
        write_hash_db(hashes)
        logger.info(
            "Ingested %d chunks from %d updated files", len(docs_to_add), processed
        )
    else:
        logger.info("No updates detected during scan")

    return IngestScanResponse(added_chunks=len(docs_to_add), processed_files=processed)


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    resources: AppResources = Depends(get_resources),
) -> QueryResponse:
    """
    Run a RAG query using the initialized retriever/LLM.
    """
    vectorstore = resources["vectorstore"]
    llm = resources["llm"]
    system_prompt = resources["system_prompt"]

    logger.info(
        "Query received: search_type=%s k=%d fetch_k=%d lambda=%.2f thr=%s",
        req.search_type,
        req.k,
        req.fetch_k,
        req.lambda_mult,
        req.score_threshold,
    )

    # Configure retriever
    if req.search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": req.k,
                "fetch_k": req.fetch_k,
                "lambda_mult": req.lambda_mult,
            },
        )
    elif req.search_type == "similarity":
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": req.k}
        )
    else:  # similarity_score_threshold
        kwargs = {"k": req.k}
        if req.score_threshold is not None:
            kwargs["score_threshold"] = req.score_threshold
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=kwargs
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    t0 = time.perf_counter()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # provenance
    docs = retriever.invoke(req.question)
    used = [
        (d.metadata.get("source_path") or d.metadata.get("source") or "unknown")  # type: ignore[union-attr]
        for d in docs
    ]
    logger.info("Retriever returned %d docs", len(used))
    logger.debug("Provenance: %s", used)

    ans = chain.invoke(req.question)
    latency_ms = (time.perf_counter() - t0) * 1e3
    logger.info("LLM answered in %.1f ms (tokens depend on provider)", latency_ms)

    text = ans.content if hasattr(ans, "content") else str(ans)
    return QueryResponse(answer=text, used_docs=used)
