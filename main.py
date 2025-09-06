import glob
import os
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, File, UploadFile
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from utils import (
    COLLECTION,
    DATA_DIR,
    QDRANT_URL,
    STATE_DIR,
    build_llm,
    build_splitter,
    compute_hash_bytes,
    compute_hash_file,
    format_docs,
    get_qdrant_storage_info,
    get_vectorstore,
    load_docs_from_path,
    read_hash_db,
    write_hash_db,
)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="RAG Microservice (FastAPI + LangChain + Qdrant)")


# -----------------------
# Schemas
# -----------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="User query/question")
    search_type: Literal["mmr", "similarity", "similarity_score_threshold"] = (
        "similarity"
    )
    k: int = 5
    fetch_k: int = 10
    lambda_mult: float = 1
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
    info.update(
        {
            "data_dir": str(DATA_DIR),
            "state_dir": str(STATE_DIR),
            "collection_name": COLLECTION,
            "qdrant_url": QDRANT_URL,
        }
    )
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
            meta = d.metadata or {}
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
        Path(p)
        for p in glob.glob(str(DATA_DIR / "**/*"), recursive=True)
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
            meta = d.metadata or {}
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
            search_kwargs={
                "k": req.k,
                "fetch_k": req.fetch_k,
                "lambda_mult": req.lambda_mult,
            },
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

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
        used.append(
            d.metadata.get("source_path") or d.metadata.get("source") or "unknown"
        )

    ans = chain.invoke(req.question)
    text = ans.content if hasattr(ans, "content") else str(ans)
    return QueryResponse(answer=text, used_docs=used)
