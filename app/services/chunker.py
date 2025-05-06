"""
Text chunking service that splits documents into semantic units.
Implements multiple chunking strategies with configurable parameters.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from loguru import logger
from ..config import settings

@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    """
    text: str
    metadata: Dict
    start_idx: int
    end_idx: int
    page_number: Optional[int] = None

class TextChunker:
    """
    Production-grade text chunking service with multiple chunking strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "semantic"
    ):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            strategy (str): Chunking strategy ('semantic', 'fixed', or 'sentence')
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy

    def chunk_document(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Split document into chunks using the specified strategy.
        
        Args:
            text (str): Document text to chunk
            metadata (Dict): Optional metadata to attach to chunks
            
        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        if self.strategy == "semantic":
            return self._semantic_chunking(text, metadata)
        elif self.strategy == "fixed":
            return self._fixed_size_chunking(text, metadata)
        elif self.strategy == "sentence":
            return self._sentence_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _semantic_chunking(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Chunk text by trying to preserve semantic units (paragraphs, sections).
        """
        chunks = []
        metadata = metadata or {}
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph would exceed chunk size
            if current_length + para_length > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text)
                ))
                
                # Start new chunk, possibly including some overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
                start_idx = start_idx + len(chunk_text) - len(overlap_text) if overlap_text else start_idx + len(chunk_text)
            
            current_chunk.append(para)
            current_length += para_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                metadata=metadata.copy(),
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text)
            ))
            
        return chunks

    def _fixed_size_chunking(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces with overlap.
        """
        chunks = []
        metadata = metadata or {}
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space != -1:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunks.append(Chunk(
                text=chunk_text.strip(),
                metadata=metadata.copy(),
                start_idx=start,
                end_idx=end
            ))
            
            # Move start position accounting for overlap
            start = end - self.chunk_overlap
            
        return chunks

    def _sentence_chunking(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Chunk text by splitting on sentence boundaries.
        """
        chunks = []
        metadata = metadata or {}
        
        # Simple sentence detection (can be improved with nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text)
                ))
                
                # Start new chunk
                current_chunk = []
                current_length = 0
                start_idx = start_idx + len(chunk_text) + 1
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                metadata=metadata.copy(),
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text)
            ))
            
        return chunks