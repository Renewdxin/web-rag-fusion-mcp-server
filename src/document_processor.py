"""
Simplified Document Processor for RAG applications.

This module provides backward compatibility while delegating heavy processing
to LlamaIndex for better performance and features.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# Local imports
try:
    from .vector_store import Document
except ImportError:

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
    Document processor that uses LlamaIndex for enhanced processing.

    This class provides backward compatibility while leveraging LlamaIndex
    for superior document processing capabilities.
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

        # Initialize LlamaIndex processor if available
        self._llamaindex_processor = None
        self._initialize_llamaindex()

    def _initialize_llamaindex(self):
        """Initialize LlamaIndex processor if available."""
        try:
            from .llamaindex_processor import LlamaIndexProcessor

            self._llamaindex_processor = LlamaIndexProcessor(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )
            self.logger.info("Using LlamaIndex for document processing")
        except ImportError:
            self.logger.warning("LlamaIndex not available, using fallback processing")

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        if self._llamaindex_processor:
            return [".pdf", ".txt", ".md", ".docx", ".html", ".htm"]
        return [".txt", ".md"]

    def can_process(self, file_path: Path) -> bool:
        """Check if file can be processed."""
        return file_path.suffix.lower() in self.get_supported_formats()

    async def process_file(
        self,
        file_path: Path,
        preserve_paragraphs: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedDocument]:
        """Process single file into chunks."""
        async with self.semaphore:
            try:
                if not self.can_process(file_path):
                    raise UnsupportedFormatError(
                        f"Unsupported format: {file_path.suffix}"
                    )

                if self._llamaindex_processor:
                    return await self._process_with_llamaindex(
                        file_path, custom_metadata
                    )
                else:
                    return await self._process_fallback(file_path, custom_metadata)

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                raise DocumentProcessingError(f"Processing failed for {file_path}: {e}")

    async def _process_with_llamaindex(
        self,
        file_path: Path,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedDocument]:
        """Process file using LlamaIndex."""
        try:
            # Load documents with LlamaIndex
            documents = await self._llamaindex_processor.load_documents(file_path)

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
            self.logger.error(f"LlamaIndex processing failed: {e}")
            return await self._process_fallback(file_path, custom_metadata)

    async def _process_fallback(
        self,
        file_path: Path,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedDocument]:
        """Fallback processing for basic text files."""
        if file_path.suffix.lower() not in [".txt", ".md"]:
            raise UnsupportedFormatError(f"Unsupported format: {file_path.suffix}")

        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Simple chunking
        chunks = []
        chunk_size = self.chunk_size
        overlap = self.overlap

        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        # Create ProcessedDocument objects
        processed_docs = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_at": datetime.utcnow().isoformat(),
                **(custom_metadata or {}),
                **self.custom_tags,
            }

            processed_doc = ProcessedDocument(
                content=chunk,
                metadata=metadata,
                source_file=file_path,
                chunk_index=i,
                processing_time=0.0,
            )
            processed_docs.append(processed_doc)

        return processed_docs

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
        """Process all supported files in a directory."""
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        # Find all files
        if recursive:
            file_paths = list(directory.rglob(file_pattern))
        else:
            file_paths = list(directory.glob(file_pattern))

        # Filter to only include files
        file_paths = [f for f in file_paths if f.is_file()]

        self.logger.info(f"Found {len(file_paths)} files in {directory}")

        return await self.process_files(
            file_paths,
            progress_callback=progress_callback,
            custom_metadata=custom_metadata,
        )

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


# Convenience functions for backward compatibility
async def process_single_file(
    file_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    custom_tags: Optional[Dict[str, str]] = None,
) -> List[Document]:
    """Process a single file and return Documents."""
    processor = DocumentProcessor(
        chunk_size=chunk_size, overlap=overlap, custom_tags=custom_tags
    )

    processed_docs = await processor.process_file(file_path)
    return processor.convert_to_documents(processed_docs)


async def process_directory_simple(
    directory: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    recursive: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[List[Document], ProcessingStats]:
    """Process a directory and return Documents."""
    processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)

    processed_docs, stats = await processor.process_directory(
        directory, recursive=recursive, progress_callback=progress_callback
    )

    documents = processor.convert_to_documents(processed_docs)
    return documents, stats


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
