# RAG Microservice

A production-ready RAG (Retrieval-Augmented Generation) microservice built with **FastAPI**, **LangChain**, and **Qdrant** vector database.

## ✨ Features

- **Document Ingestion**: Upload PDF/TXT files or scan directories
- **Vector Storage**: Persistent Qdrant vector database with automatic collection management
- **Multiple Search Types**: MMR, similarity, and similarity score threshold
- **Flexible Embeddings**: OpenAI or HuggingFace embedding models
- **Data Persistence**: Reliable vector data storage across service restarts
- **Health Monitoring**: Storage and service health endpoints

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (for embeddings/LLM)

### Setup

1. **Clone and setup environment**:
```bash
git clone <repository>
cd rag_api

# Copy environment template and configure
cp .env.example .env
# Edit .env with your OpenAI API key and other settings
```

2. **Start services**:
```bash
docker-compose up -d
```

3. **Verify services are running**:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/storage/info
```

## 📁 Data Persistence

The service is configured for **persistent vector storage** across restarts:

### Directory Structure
```
rag_api/
├── data/           # Document storage
├── state/          # Application state (file hashes)
├── qdrant_data/    # Qdrant vector database persistence
│   ├── collections/
│   └── snapshots/
└── ...
```

### Key Features
- **Automatic Collection Creation**: Collections are created automatically on first use
- **Persistent Storage**: Vector data survives container restarts
- **Storage Monitoring**: Track storage size and collection count
- **Directory Management**: Automatic directory structure creation

### Test Persistence
```bash
./test_persistence.sh
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Document storage directory |
| `STATE_DIR` | `./state` | Application state directory |
| `QDRANT_DATA_DIR` | `./qdrant_data` | Qdrant persistence directory |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL |
| `COLLECTION` | `docs_rag` | Default collection name |
| `EMBED_PROVIDER` | `openai` | Embedding provider (`openai` or `hf`) |
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for generation |

### Docker Volumes

The `docker-compose.yml` is configured with proper volume mappings:
- `./qdrant_data:/qdrant/storage:z` - Qdrant data persistence
- `./data:/app/data` - Document storage
- `./state:/app/state` - Application state

## 📡 API Endpoints

### Health & Monitoring
- `GET /health` - Service health check
- `GET /storage/info` - Storage and persistence information

### Document Ingestion
- `POST /ingest/upload` - Upload PDF/TXT files
- `POST /ingest/scan` - Scan data directory for new/changed files

### Query
- `POST /query` - Query documents with RAG

## 💡 Usage Examples

### Upload Documents
```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "files=@document.pdf" \
  -F "files=@document.txt"
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the documents?",
    "search_type": "mmr",
    "k": 5
  }'
```

### Check Storage Status
```bash
curl http://localhost:8000/storage/info
```

## 🔍 Storage Monitoring

The `/storage/info` endpoint provides detailed information about:
- Qdrant data directory status and size
- Collections and snapshots directories
- Collection count
- Data persistence status

Example response:
```json
{
  "qdrant_data_dir": "./qdrant_data",
  "qdrant_data_exists": true,
  "qdrant_data_size_bytes": 1048576,
  "collections_dir_exists": true,
  "snapshots_dir_exists": true,
  "collection_count": 1,
  "data_dir": "./data",
  "state_dir": "./state",
  "collection_name": "docs_rag",
  "qdrant_url": "http://localhost:6333"
}
```

## 🛠 Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant only
docker-compose up qdrant -d

# Run application locally
uvicorn main:app --reload
```

### Adding New Features
1. Update `main.py` with new functionality
2. Add corresponding tests
3. Update documentation
4. Test persistence behavior

## 📚 Architecture

- **FastAPI**: REST API framework
- **LangChain**: Document processing and RAG chain
- **Qdrant**: Vector database with cosine similarity
- **OpenAI/HuggingFace**: Embedding models and LLMs
- **Docker**: Containerized deployment

## 🔒 Production Considerations

- Configure proper API tokens and authentication
- Set up monitoring and logging
- Use production-grade Qdrant deployment
- Implement backup strategies for vector data
- Configure resource limits in docker-compose.yml
