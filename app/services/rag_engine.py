"""
RAG Engine that orchestrates the complete pipeline from document ingestion
to question answering using retrieved context.
"""

from typing import List, Dict, Optional, Any, Tuple
import asyncio
from loguru import logger
import openai
from .parser import PDFParser
from .chunker import TextChunker, Chunk
from .embedder import Embedder
from ..vector_store.qdrant_client import QdrantStore
from ..config import settings

class RAGEngine:
    """
    Production-grade RAG engine that coordinates document processing,
    retrieval, and generation.
    """
    
    def __init__(
        self,
        parser: Optional[PDFParser] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[QdrantStore] = None
    ):
        """
        Initialize the RAG engine with its component services.
        
        Args:
            parser (PDFParser): PDF parsing service
            chunker (TextChunker): Text chunking service
            embedder (Embedder): Text embedding service
            vector_store (QdrantStore): Vector storage service
        """
        self.parser = parser or PDFParser()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore()
        
        if settings.USE_OPENAI:
            openai.api_key = settings.OPENAI_API_KEY
            
    async def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path (str): Path to the PDF file
            metadata (Dict): Optional metadata to store with chunks
            
        Returns:
            List[str]: List of chunk IDs in the vector store
        """
        try:
            # Parse PDF
            logger.info(f"Parsing PDF: {file_path}")
            parsed_pages = await self.parser.parse_pdf(file_path)
            
            # Extract metadata if not provided
            if not metadata:
                metadata = await self.parser.extract_metadata(file_path)
            
            # Chunk text
            chunks = []
            for page_num, text in parsed_pages.items():
                page_chunks = self.chunker.chunk_document(
                    text,
                    {
                        **metadata,
                        "page_number": page_num,
                        "source": file_path
                    }
                )
                chunks.extend(page_chunks)
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            chunk_embeddings = await self.embedder.generate_embeddings(texts)
            
            # Store in vector database
            chunk_metadata = [
                {
                    **chunk.metadata,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx
                }
                for chunk in chunks
            ]
            
            chunk_ids = await self.vector_store.upsert_vectors(
                vectors=chunk_embeddings,
                metadata=chunk_metadata
            )
            
            logger.info(f"Stored {len(chunk_ids)} chunks in vector store")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    async def answer_question(
        self,
        question: str,
        num_chunks: int = 3,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question (str): The question to answer
            num_chunks (int): Number of chunks to retrieve
            temperature (float): Temperature for generation
            
        Returns:
            Dict[str, Any]: Answer and supporting information
        """
        try:
            # Generate embedding for question
            question_embedding = await self.embedder.generate_embeddings([question])
            
            # Retrieve relevant chunks
            relevant_chunks = await self.vector_store.search_vectors(
                query_vector=question_embedding[0],
                limit=num_chunks
            )
            
            # Extract text and metadata
            contexts = []
            sources = []
            for chunk_meta, score in relevant_chunks:
                contexts.append(chunk_meta.get("text", ""))
                sources.append({
                    "source": chunk_meta.get("source"),
                    "page": chunk_meta.get("page_number"),
                    "score": score
                })
            
            # Construct prompt with retrieved context
            prompt = self._construct_qa_prompt(question, contexts)
            
            # Generate answer
            response = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources,
                "model": settings.OPENAI_MODEL,
                "chunks_used": len(contexts)
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            raise

    def _construct_qa_prompt(
        self,
        question: str,
        contexts: List[str]
    ) -> str:
        """
        Construct a prompt for question answering.
        
        Args:
            question (str): The question to answer
            contexts (List[str]): Retrieved context passages
            
        Returns:
            str: Formatted prompt
        """
        context_str = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))
        
        return f"""Please answer the question based on the following context passages:

{context_str}

Question: {question}

Answer:"""

    async def update_document(
        self,
        file_path: str,
        chunk_ids: List[str],
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Update an existing document's chunks.
        
        Args:
            file_path (str): Path to the updated PDF
            chunk_ids (List[str]): IDs of chunks to update
            metadata (Dict): Optional metadata to update
            
        Returns:
            List[str]: List of new chunk IDs
        """
        # Delete old chunks
        await self.vector_store.delete_vectors(chunk_ids)
        
        # Process updated document
        return await self.process_document(file_path, metadata)

    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic search.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            filter_conditions (Dict): Optional filters
            
        Returns:
            List[Dict[str, Any]]: Relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = await self.embedder.generate_embeddings([query])
        
        # Search vector store
        results = await self.vector_store.search_vectors(
            query_vector=query_embedding[0],
            limit=limit,
            filter_conditions=filter_conditions
        )
        
        # Format results
        return [
            {
                "text": meta.get("text", ""),
                "source": meta.get("source"),
                "page": meta.get("page_number"),
                "score": score,
                "metadata": meta
            }
            for meta, score in results
        ]