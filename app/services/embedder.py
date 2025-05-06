"""
Embedder service that generates vector embeddings for text chunks.
Supports multiple embedding models with fallback options.
"""

from typing import List, Dict, Union, Optional
import numpy as np
from loguru import logger
import openai
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from ..config import settings

class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""
    pass

class Embedder:
    """
    Production-grade embedding service with support for multiple models
    and batched processing.
    """
    
    def __init__(
        self,
        model_name: str = None,
        batch_size: int = 8,
        max_workers: int = 4
    ):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name (str): Name of the embedding model to use
            batch_size (int): Number of texts to process in parallel
            max_workers (int): Maximum number of worker threads
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._model = None  # Lazy loading
        
        if settings.USE_OPENAI:
            openai.api_key = settings.OPENAI_API_KEY
            
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            model (str): Optional override for model name
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        model = model or self.model_name
        
        try:
            if model == "openai":
                return await self._generate_openai_embeddings(texts)
            else:
                return await self._generate_local_embeddings(texts, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's API.
        """
        async def process_batch(batch: List[str]) -> List[List[float]]:
            try:
                response = await openai.Embedding.acreate(
                    input=batch,
                    model="text-embedding-3-small"  # Using the latest model
                )
                return [item["embedding"] for item in response["data"]]
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {str(e)}")
                raise EmbeddingError(f"OpenAI embedding failed: {str(e)}")

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await process_batch(batch)
            embeddings.extend(batch_embeddings)
            
        return embeddings

    async def _generate_local_embeddings(
        self,
        texts: List[str],
        model_name: str
    ) -> List[List[float]]:
        """
        Generate embeddings using a local model.
        """
        loop = asyncio.get_running_loop()
        
        if self._model is None:
            def _load_model():
                return SentenceTransformer(model_name)
            self._model = await loop.run_in_executor(self.executor, _load_model)

        async def process_batch(batch: List[str]) -> np.ndarray:
            def _embed():
                return self._model.encode(
                    batch,
                    normalize_embeddings=True
                )
            return await loop.run_in_executor(self.executor, _embed)

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await process_batch(batch)
            embeddings.extend(batch_embeddings.tolist())
            
        return embeddings

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.executor.shutdown(wait=True)

    @staticmethod
    def normalize_embedding(embedding: List[float]) -> List[float]:
        """
        L2 normalize an embedding vector.
        
        Args:
            embedding (List[float]): Input embedding vector
            
        Returns:
            List[float]: Normalized embedding vector
        """
        array = np.array(embedding)
        norm = np.linalg.norm(array)
        if norm > 0:
            return (array / norm).tolist()
        return array.tolist()

    @staticmethod
    def cosine_similarity(
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        array1 = np.array(embedding1)
        array2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(array1, array2) / (norm1 * norm2))