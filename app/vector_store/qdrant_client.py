"""
Qdrant vector database client for storing and retrieving embeddings.
Implements connection pooling, retries, and error handling.
"""

from typing import List, Dict, Optional, Tuple, Any
import asyncio
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger
from ..config import settings

class QdrantError(Exception):
    """Custom exception for Qdrant operations."""
    pass

class QdrantStore:
    """
    Production-grade Qdrant vector store client with connection management
    and error handling.
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None,
        vector_size: int = 1536,  # OpenAI ada-002 default
        distance: str = "Cosine"
    ):
        """
        Initialize Qdrant client with connection parameters.
        
        Args:
            url (str): Qdrant server URL
            api_key (str): API key for authentication
            collection_name (str): Name of the collection to use
            vector_size (int): Dimension of vectors to store
            distance (str): Distance metric to use (Cosine or Dot)
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size
        self.distance = distance
        
        # Initialize client
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=10  # 10 second timeout
        )
        
        # Ensure collection exists
        self._ensure_collection()
        
    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists and has correct settings.
        """
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=rest.Distance[self.distance]
                    )
                )
                logger.info(f"Created collection {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
            raise QdrantError(f"Collection initialization failed: {str(e)}")
            
    async def upsert_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Store vectors and their metadata in the collection.
        
        Args:
            vectors (List[List[float]]): List of embedding vectors
            metadata (List[Dict]): List of metadata dicts for each vector
            ids (List[str]): Optional list of IDs for the vectors
            
        Returns:
            List[str]: List of assigned vector IDs
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in vectors]
                
            # Create points
            points = [
                models.PointStruct(
                    id=id_,
                    vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    payload=meta
                )
                for id_, vector, meta in zip(ids, vectors, metadata)
            ]
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            return ids
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            raise QdrantError(f"Vector upsert failed: {str(e)}")
            
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector (List[float]): Query embedding vector
            limit (int): Maximum number of results to return
            score_threshold (float): Minimum similarity score threshold
            filter_conditions (Dict): Optional filtering conditions
            
        Returns:
            List[Tuple[Dict, float]]: List of (metadata, score) tuples
        """
        try:
            search_result = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=rest.Filter(**filter_conditions) if filter_conditions else None
            )
            
            return [(hit.payload, hit.score) for hit in search_result]
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise QdrantError(f"Vector search failed: {str(e)}")
            
    async def delete_vectors(
        self,
        ids: List[str]
    ) -> None:
        """
        Delete vectors by their IDs.
        
        Args:
            ids (List[str]): List of vector IDs to delete
        """
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=ids
                )
            )
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            raise QdrantError(f"Vector deletion failed: {str(e)}")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._client.close()
        
    @property
    def collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information and statistics
        """
        try:
            return self._client.get_collection(
                collection_name=self.collection_name
            ).dict()
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            raise QdrantError(f"Failed to get collection info: {str(e)}")