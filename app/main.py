"""
FastAPI server for health checks and admin routes.
Provides monitoring and administrative capabilities.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram
import time
from typing import Dict, Any
from loguru import logger
from starlette.responses import Response
import psutil
import os

from .services.rag_engine import RAGEngine
from .vector_store.qdrant_client import QdrantStore
from .config import settings

app = FastAPI(
    title="RAGify Admin API",
    description="Administrative API for RAGify service",
    version="1.0.0"
)

# Prometheus metrics
REQUESTS = Counter('ragify_requests_total', 'Total requests by endpoint')
LATENCY = Histogram('ragify_request_latency_seconds', 'Request latency by endpoint')

# Initialize services
rag_engine = RAGEngine()
vector_store = QdrantStore()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    """
    REQUESTS.labels(endpoint='/health').inc()
    
    with LATENCY.labels(endpoint='/health').time():
        try:
            # Check vector store connectivity
            collection_info = vector_store.collection_info
            
            return {
                "status": "healthy",
                "vector_store": "connected",
                "collection": collection_info.get("name"),
                "vector_count": collection_info.get("vectors_count", 0)
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Service unhealthy: {str(e)}"
            )

@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    """
    REQUESTS.labels(endpoint='/metrics').inc()
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get detailed system and service statistics.
    """
    REQUESTS.labels(endpoint='/stats').inc()
    
    with LATENCY.labels(endpoint='/stats').time():
        try:
            # System stats
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Vector store stats
            collection_info = vector_store.collection_info
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "pid": os.getpid()
                },
                "vector_store": {
                    "collection": collection_info.get("name"),
                    "vectors_count": collection_info.get("vectors_count", 0),
                    "segments_count": collection_info.get("segments_count", 0),
                    "status": collection_info.get("status", "unknown")
                },
                "service": {
                    "grpc_port": settings.SERVICE_PORT,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
                }
            }
        except Exception as e:
            logger.error(f"Stats collection failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect stats: {str(e)}"
            )

@app.post("/clear-collection")
async def clear_collection() -> Dict[str, Any]:
    """
    Clear all vectors from the collection.
    Warning: This will delete all stored documents!
    """
    REQUESTS.labels(endpoint='/clear-collection').inc()
    
    with LATENCY.labels(endpoint='/clear-collection').time():
        try:
            # Get current vector count
            collection_info = vector_store.collection_info
            initial_count = collection_info.get("vectors_count", 0)
            
            # Recreate collection
            vector_store._ensure_collection()
            
            return {
                "success": True,
                "message": f"Cleared {initial_count} vectors from collection",
                "vectors_removed": initial_count
            }
        except Exception as e:
            logger.error(f"Collection clear failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear collection: {str(e)}"
            )

@app.get("/collection-info")
async def get_collection_info() -> Dict[str, Any]:
    """
    Get detailed information about the vector collection.
    """
    REQUESTS.labels(endpoint='/collection-info').inc()
    
    with LATENCY.labels(endpoint='/collection-info').time():
        try:
            return vector_store.collection_info
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get collection info: {str(e)}"
            )

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {duration:.3f}s with status {response.status_code}"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add(
        "logs/fastapi.log",
        rotation="500 MB",
        retention="10 days",
        level=settings.LOG_LEVEL
    )
    
    # Start server
    uvicorn.run(
        app,
        host=settings.SERVICE_HOST,
        port=8000,  # Different port from gRPC
        log_level="info"
    )