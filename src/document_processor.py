"""
Simplified Document Processor using LlamaIndex SimpleDirectoryReader.

This module provides a streamlined interface that directly uses LlamaIndex's
SimpleDirectoryReader for optimal document processing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader


# Compatibility layer for legacy Document format
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    content_hash: Optional[str] = None


class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int, status: str = "") -> None: ...


@dataclass
class ProcessedDocument:
    """Container for processed document with metadata."""

    content: str
    metadata: Dict[str, Any]
    source_file: Path
    chunk_index: Optional[int] = None
    processing_time: float = 0.0


@dataclass
class ProcessingStats:
    """Statistics for document processing operations."""

    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    execution_time: float
    bytes_processed: int
    cache_hits: int = 0
    cache_misses: int = 0
    files_by_type: Dict[str, int] = field(default_factory=dict)


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""

    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when file format is not supported."""

    pass


class ContentValidationError(DocumentProcessingError):
    """Raised when content validation fails."""

    pass


class DocumentProcessor:
    """
    Simplified document processor using LlamaIndex SimpleDirectoryReader.

    This class provides a streamlined interface for document processing,
    delegating all heavy lifting to LlamaIndex's built-in capabilities.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        max_concurrency: int = 5,
        cache_dir: Optional[Path] = None,
        custom_tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize DocumentProcessor."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_concurrency = max_concurrency
        self.custom_tags = custom_tags or {}

        self.logger = logging.getLogger(f"{__name__}.DocumentProcessor")
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats (LlamaIndex SimpleDirectoryReader)."""
        return [".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv", ".json", ".xml"]

    def can_process(self, file_path: Path) -> bool:
        """Check if file can be processed."""
        return file_path.suffix.lower() in self.get_supported_formats()

    async def process_file(
        self,
        file_path: Path,
        preserve_paragraphs: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedDocument]:
        """Process single file using LlamaIndex SimpleDirectoryReader."""
        async with self.semaphore:
            try:
                if not self.can_process(file_path):
                    raise UnsupportedFormatError(
                        f"Unsupported format: {file_path.suffix}"
                    )

                return await self._process_with_llamaindex(file_path, custom_metadata)

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                raise DocumentProcessingError(f"Processing failed for {file_path}: {e}")

    async def _process_with_llamaindex(
        self,
        file_path: Path,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedDocument]:
        """Process file using LlamaIndex SimpleDirectoryReader."""
        try:
            # Use SimpleDirectoryReader directly
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                exclude_hidden=True,
            )
            documents = reader.load_data()

            # Convert to ProcessedDocument format
            processed_docs = []
            for i, doc in enumerate(documents):
                metadata = {
                    **doc.metadata,
                    **(custom_metadata or {}),
                    **self.custom_tags,
                    "chunk_index": i,
                    "total_chunks": len(documents),
                    "processed_at": datetime.utcnow().isoformat(),
                }

                processed_doc = ProcessedDocument(
                    content=doc.text,
                    metadata=metadata,
                    source_file=file_path,
                    chunk_index=i,
                    processing_time=0.0,
                )
                processed_docs.append(processed_doc)

            return processed_docs

        except Exception as e:
            self.logger.error(f"SimpleDirectoryReader processing failed: {e}")
            raise DocumentProcessingError(f"Failed to process {file_path}: {e}")


    async def process_files(
        self,
        file_paths: List[Path],
        preserve_paragraphs: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[ProcessedDocument], ProcessingStats]:
        """Process multiple files."""
        start_time = asyncio.get_event_loop().time()
        all_docs = []
        processed_count = 0
        failed_count = 0

        # Process files with concurrency control
        tasks = []
        for file_path in file_paths:
            if self.can_process(file_path):
                task = asyncio.create_task(
                    self.process_file(file_path, preserve_paragraphs, custom_metadata)
                )
                tasks.append((file_path, task))

        # Wait for all tasks to complete
        for i, (file_path, task) in enumerate(tasks):
            try:
                docs = await task
                all_docs.extend(docs)
                processed_count += 1

                if progress_callback:
                    progress_callback(i + 1, len(tasks), f"Processed {file_path.name}")

            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to process {file_path}: {e}")

                if progress_callback:
                    progress_callback(i + 1, len(tasks), f"Failed {file_path.name}")

        # Create statistics
        stats = ProcessingStats(
            total_files=len(file_paths),
            processed_files=processed_count,
            failed_files=failed_count,
            total_chunks=len(all_docs),
            execution_time=asyncio.get_event_loop().time() - start_time,
            bytes_processed=sum(len(doc.content) for doc in all_docs),
        )

        return all_docs, stats

    async def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "*",
        progress_callback: Optional[ProgressCallback] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[ProcessedDocument], ProcessingStats]:
        """Process all supported files in a directory using SimpleDirectoryReader."""
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use SimpleDirectoryReader for directory processing
            reader = SimpleDirectoryReader(
                input_dir=str(directory),
                recursive=recursive,
                exclude_hidden=True,
            )
            documents = reader.load_data()

            # Convert to ProcessedDocument format
            processed_docs = []
            for i, doc in enumerate(documents):
                metadata = {
                    **doc.metadata,
                    **(custom_metadata or {}),
                    **self.custom_tags,
                    "chunk_index": i,
                    "total_chunks": len(documents),
                    "processed_at": datetime.utcnow().isoformat(),
                }

                processed_doc = ProcessedDocument(
                    content=doc.text,
                    metadata=metadata,
                    source_file=Path(doc.metadata.get("file_path", "unknown")),
                    chunk_index=i,
                    processing_time=0.0,
                )
                processed_docs.append(processed_doc)

                if progress_callback:
                    progress_callback(i + 1, len(documents), f"Processed document {i+1}")

            # Create statistics
            stats = ProcessingStats(
                total_files=len(set(doc.metadata.get("file_path", "unknown") for doc in documents)),
                processed_files=len(set(doc.metadata.get("file_path", "unknown") for doc in documents)),
                failed_files=0,
                total_chunks=len(processed_docs),
                execution_time=asyncio.get_event_loop().time() - start_time,
                bytes_processed=sum(len(doc.content) for doc in processed_docs),
            )

            self.logger.info(f"Processed {len(processed_docs)} documents from {directory}")
            return processed_docs, stats

        except Exception as e:
            self.logger.error(f"Failed to process directory {directory}: {e}")
            raise DocumentProcessingError(f"Directory processing failed: {e}")

    def convert_to_documents(
        self, processed_docs: List[ProcessedDocument]
    ) -> List[Document]:
        """Convert ProcessedDocument list to Document list."""
        documents = []
        for proc_doc in processed_docs:
            doc = Document(
                page_content=proc_doc.content,
                metadata=proc_doc.metadata,
                id=f"{proc_doc.source_file.stem}_{proc_doc.chunk_index}",
            )
            documents.append(doc)
        return documents


# Convenience functions for backward compatibility using SimpleDirectoryReader
async def process_single_file(
    file_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    custom_tags: Optional[Dict[str, str]] = None,
) -> List[Document]:
    """Process a single file using SimpleDirectoryReader and return Documents."""
    try:
        # Use SimpleDirectoryReader directly
        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            exclude_hidden=True,
        )
        llama_docs = reader.load_data()
        
        # Convert to legacy Document format
        documents = []
        for i, doc in enumerate(llama_docs):
            metadata = {
                **doc.metadata,
                **(custom_tags or {}),
                "chunk_index": i,
                "total_chunks": len(llama_docs),
                "processed_at": datetime.utcnow().isoformat(),
            }
            
            document = Document(
                page_content=doc.text,
                metadata=metadata,
                id=f"{file_path.stem}_{i}",
            )
            documents.append(document)
            
        return documents
        
    except Exception as e:
        raise DocumentProcessingError(f"Failed to process {file_path}: {e}")


async def process_directory_simple(
    directory: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    recursive: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[List[Document], ProcessingStats]:
    """Process a directory using SimpleDirectoryReader and return Documents."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Use SimpleDirectoryReader directly
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            recursive=recursive,
            exclude_hidden=True,
        )
        llama_docs = reader.load_data()
        
        # Convert to legacy Document format
        documents = []
        for i, doc in enumerate(llama_docs):
            metadata = {
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(llama_docs),
                "processed_at": datetime.utcnow().isoformat(),
            }
            
            document = Document(
                page_content=doc.text,
                metadata=metadata,
                id=f"{Path(doc.metadata.get('file_path', 'unknown')).stem}_{i}",
            )
            documents.append(document)
            
            if progress_callback:
                progress_callback(i + 1, len(llama_docs), f"Processed document {i+1}")
        
        # Create statistics
        stats = ProcessingStats(
            total_files=len(set(doc.metadata.get("file_path", "unknown") for doc in llama_docs)),
            processed_files=len(set(doc.metadata.get("file_path", "unknown") for doc in llama_docs)),
            failed_files=0,
            total_chunks=len(documents),
            execution_time=asyncio.get_event_loop().time() - start_time,
            bytes_processed=sum(len(doc.page_content) for doc in documents),
        )
        
        return documents, stats
        
    except Exception as e:
        raise DocumentProcessingError(f"Failed to process directory {directory}: {e}")


# Export main classes and functions
__all__ = [
    "DocumentProcessor",
    "ProcessedDocument",
    "ProcessingStats",
    "DocumentProcessingError",
    "UnsupportedFormatError",
    "ContentValidationError",
    "process_single_file",
    "process_directory_simple",
    "ProgressCallback",
]
