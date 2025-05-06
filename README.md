
# 🧠 ragify-core-python

`ragify-core-python` is the backend intelligence of the RAGify platform. It performs PDF parsing, text chunking, embedding generation, and semantic question-answering using Retrieval-Augmented Generation (RAG).

---

## 📌 Responsibilities

- 🧾 Parse PDFs into clean text
- ✂️ Chunk text into semantic units
- 🧬 Generate vector embeddings for chunks
- 🔍 Store/retrieve from a vector database (Qdrant, Weaviate, etc.)
- 🤖 Run semantic search + LLM query answering
- 🔗 Serve gRPC interface for external orchestration (Node.js API)

---

## 📁 Folder Structure

```
ragify-core-python/
├── app/
│   ├── grpc_server.py         # gRPC server entrypoint
│   ├── main.py                # (Optional) FastAPI health/admin routes
│   ├── config.py              # Env + global settings loader
│   ├── services/
│   │   ├── parser.py          # PDF to raw text
│   │   ├── chunker.py         # Text to chunk list
│   │   ├── embedder.py        # Chunk to embeddings
│   │   └── rag_engine.py      # Retrieval + LLM inference
│   ├── vector_store/
│   │   └── qdrant_client.py   # Vector DB abstraction
│   ├── proto/                 # Generated stubs from rag.proto
│   └── utils/                 # Logging, helpers
├── Dockerfile
├── requirements.txt
└── entrypoint.sh              # Entrypoint for container run
```

---

## ⚙️ Configuration (`app/config.py`)

Uses `pydantic.BaseSettings` to manage environment variables.

```python
class Settings(BaseSettings):
    QDRANT_URL: str
    EMBEDDING_MODEL: str = "openai"
    USE_OPENAI: bool = True
    OPENAI_API_KEY: str = ""
    SERVICE_PORT: int = 50051
```

---

## 🛠️ Core Stack

| Function        | Library                 |
|----------------|--------------------------|
| PDF Parsing     | `PyMuPDF`, `pdfplumber` |
| Chunking        | `langchain` or custom   |
| Embedding       | `Instructor`, `OpenAI`, `sentence-transformers` |
| Vector DB       | `qdrant-client`, `weaviate-client` |
| gRPC Server     | `grpcio`, `protobuf`    |
| Health API      | `FastAPI` (optional)    |
| Config          | `pydantic`, `dotenv`    |
| Logging         | `loguru` or `structlog` |

---

## 🐳 Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app

CMD ["python", "-m", "app.grpc_server"]
```

---

## 🧪 Testing Strategy

- ✅ Unit tests for PDF parsing, chunking, and embedding
- ✅ gRPC endpoint tests (e.g. with `grpcurl`)
- ✅ Integration tests for full RAG pipeline
- ✅ Optional: use `pytest` and test PDFs

---

## 📊 Scalability Considerations

- Use Redis queue for long parsing tasks
- Add Prometheus metrics
- Deploy via K8s or Docker Swarm
- Add FastAPI admin panel for internal debugging

---

## 🔗 External Interface

All interaction is via gRPC as defined in the shared [`ragify-protos`](https://github.com/your-org/ragify-protos) repo.

