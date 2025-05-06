"""
PDF parsing service that handles text extraction from PDF documents.
Implements fallback mechanisms and robust error handling for production use.
"""

from typing import List, Dict, Optional
import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
import tempfile
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PDFParsingError(Exception):
    """Custom exception for PDF parsing errors."""
    pass

class PDFParser:
    """
    Production-grade PDF parser with fallback mechanisms and parallel processing capabilities.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the PDF parser.
        
        Args:
            max_workers (int): Maximum number of worker threads for parallel processing
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def parse_pdf(self, file_path: str, strategy: str = "auto") -> Dict[str, str]:
        """
        Parse a PDF file and extract its text content.
        
        Args:
            file_path (str): Path to the PDF file
            strategy (str): Parsing strategy ('auto', 'pymupdf', or 'pdfplumber')
            
        Returns:
            Dict[str, str]: Dictionary containing page numbers and their text content
        
        Raises:
            PDFParsingError: If parsing fails with both strategies
        """
        try:
            if strategy == "auto":
                # Try PyMuPDF first, fall back to pdfplumber if needed
                try:
                    return await self._parse_with_pymupdf(file_path)
                except Exception as e:
                    logger.warning(f"PyMuPDF parsing failed, falling back to pdfplumber: {e}")
                    return await self._parse_with_pdfplumber(file_path)
            elif strategy == "pymupdf":
                return await self._parse_with_pymupdf(file_path)
            elif strategy == "pdfplumber":
                return await self._parse_with_pdfplumber(file_path)
            else:
                raise ValueError(f"Unknown parsing strategy: {strategy}")
                
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF: {str(e)}")

    async def _parse_with_pymupdf(self, file_path: str) -> Dict[str, str]:
        """
        Parse PDF using PyMuPDF (faster but may have issues with some PDFs).
        """
        loop = asyncio.get_running_loop()
        
        def _process_pdf():
            result = {}
            doc = fitz.open(file_path)
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    result[str(page_num + 1)] = page.get_text("text")
                return result
            finally:
                doc.close()
                
        return await loop.run_in_executor(self.executor, _process_pdf)

    async def _parse_with_pdfplumber(self, file_path: str) -> Dict[str, str]:
        """
        Parse PDF using pdfplumber (slower but more reliable).
        """
        loop = asyncio.get_running_loop()
        
        def _process_pdf():
            result = {}
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    result[str(page_num)] = page.extract_text() or ""
            return result
            
        return await loop.run_in_executor(self.executor, _process_pdf)

    async def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract PDF metadata including title, author, creation date, etc.
        """
        loop = asyncio.get_running_loop()
        
        def _extract():
            with fitz.open(file_path) as doc:
                return dict(doc.metadata)
                
        return await loop.run_in_executor(self.executor, _extract)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.executor.shutdown(wait=True)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and normalizing newlines.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        import re
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()