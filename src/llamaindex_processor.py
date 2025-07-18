import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import config


class LlamaIndexProcessor:
    """
    Enhanced document processor using LlamaIndex for better RAG performance.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        similarity_top_k: int = 10,
        similarity_cutoff: float = 0.7,
    ):
        """
        Initialize the LlamaIndex processor.

        Args:
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model name
            similarity_top_k: Number of similar documents to retrieve
            similarity_cutoff: Minimum similarity score threshold
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff

        self.logger = logging.getLogger(f"{__name__}.LlamaIndexProcessor")

        # Initialize LlamaIndex settings
        self._setup_settings(embedding_model)

        # Initialize ChromaDB
        self._setup_chroma()

        # Initialize index
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None

        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
        )

    def _setup_settings(self, embedding_model: str):
        """Setup LlamaIndex global settings."""
        # Check if OpenAI API key is available
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for LlamaIndex integration")

        # Set up embedding model
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=config.OPENAI_API_KEY,
            embed_batch_size=100,  # Batch embeddings for efficiency
        )

        # Set up LLM for query processing
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=config.OPENAI_API_KEY,
            temperature=0.1,
        )

        # Set chunk size for node parser
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

    def _setup_chroma(self):
        """Setup ChromaDB client and collection."""
        try:
            # Initialize ChromaDB with persistent storage
            chroma_settings = ChromaSettings(
                persist_directory=str(config.VECTOR_STORE_PATH),
                anonymized_telemetry=False,
            )

            self.chroma_client = chromadb.PersistentClient(
                path=str(config.VECTOR_STORE_PATH),
                settings=chroma_settings,
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Create ChromaVectorStore
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

            self.logger.info(
                f"ChromaDB initialized with collection: {self.collection_name}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _create_index(self) -> VectorStoreIndex:
        """Create or load the vector index."""
        try:
            # Create storage context with vector store
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Try to load existing index
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                )
                self.logger.info("Loaded existing vector index")
                return index
            except Exception:
                # Create new index if none exists
                index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                )
                self.logger.info("Created new vector index")
                return index

        except Exception as e:
            self.logger.error(f"Failed to create/load index: {e}")
            raise

    def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create query engine with optimized retrieval."""
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.similarity_top_k,
            )

            # Create postprocessor for similarity filtering
            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=self.similarity_cutoff
            )

            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                streaming=False,
            )

            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[postprocessor],
            )

            self.logger.info("Created optimized query engine")
            return query_engine

        except Exception as e:
            self.logger.error(f"Failed to create query engine: {e}")
            raise

    async def load_documents(
        self,
        input_path: Union[str, Path],
        file_extractor: Optional[Dict[str, Any]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load documents from file or directory using LlamaIndex readers.

        Args:
            input_path: Path to file or directory
            file_extractor: Custom file extractors for specific formats
            recursive: Whether to recursively load from subdirectories

        Returns:
            List of LlamaIndex Document objects
        """
        try:
            input_path = Path(input_path)

            if not input_path.exists():
                raise FileNotFoundError(f"Path does not exist: {input_path}")

            # Use SimpleDirectoryReader for comprehensive file loading
            if input_path.is_file():
                # Load single file
                reader = SimpleDirectoryReader(
                    input_files=[str(input_path)],
                    file_extractor=file_extractor,
                )
            else:
                # Load directory
                reader = SimpleDirectoryReader(
                    input_dir=str(input_path),
                    recursive=recursive,
                    file_extractor=file_extractor,
                    exclude_hidden=True,
                )

            # Load documents
            documents = reader.load_data()

            # Add custom metadata
            for doc in documents:
                doc.metadata.update(
                    {
                        "processor": "llamaindex",
                        "source": str(input_path),
                    }
                )

            self.logger.info(f"Loaded {len(documents)} documents from {input_path}")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load documents from {input_path}: {e}")
            raise

    async def process_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """
        Process documents and create/update vector index.

        Args:
            documents: List of documents to process
            show_progress: Whether to show processing progress

        Returns:
            Updated vector index
        """
        try:
            if not documents:
                self.logger.warning("No documents to process")
                return self.index or self._create_index()

            # Create index if it doesn't exist
            if self.index is None:
                self.index = self._create_index()

            # Parse documents into nodes with custom chunking
            nodes = self.node_parser.get_nodes_from_documents(
                documents, show_progress=show_progress
            )

            # Add nodes to index
            self.index.insert_nodes(nodes)

            # Create/update query engine
            self.query_engine = self._create_query_engine()

            self.logger.info(
                f"Processed {len(documents)} documents into {len(nodes)} nodes"
            )
            return self.index

        except Exception as e:
            self.logger.error(f"Failed to process documents: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_cutoff: Optional[float] = None,
    ) -> List[NodeWithScore]:
        """
        Search the knowledge base using optimized retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_cutoff: Minimum similarity score

        Returns:
            List of nodes with similarity scores
        """
        try:
            if self.index is None:
                raise RuntimeError("Index not initialized. Process documents first.")

            # Use custom parameters if provided
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k or self.similarity_top_k,
            )

            # Retrieve nodes
            nodes = retriever.retrieve(query)

            # Apply similarity cutoff if specified
            if similarity_cutoff is not None:
                nodes = [node for node in nodes if node.score >= similarity_cutoff]
            elif self.similarity_cutoff > 0:
                nodes = [node for node in nodes if node.score >= self.similarity_cutoff]

            self.logger.debug(f"Retrieved {len(nodes)} nodes for query: {query}")
            return nodes

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    async def query(
        self,
        query: str,
        response_mode: str = "compact",
    ) -> str:
        """
        Query the knowledge base with response synthesis.

        Args:
            query: Query string
            response_mode: Response synthesis mode

        Returns:
            Synthesized response
        """
        try:
            if self.query_engine is None:
                raise RuntimeError(
                    "Query engine not initialized. Process documents first."
                )

            # Execute query
            response = self.query_engine.query(query)

            self.logger.debug(f"Generated response for query: {query}")
            return str(response)

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            stats = {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "similarity_top_k": self.similarity_top_k,
                "similarity_cutoff": self.similarity_cutoff,
                "index_initialized": self.index is not None,
                "query_engine_initialized": self.query_engine is not None,
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    async def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.index = None
            self.query_engine = None

            # Recreate collection
            self._setup_chroma()

            self.logger.info(
                f"Deleted and recreated collection: {self.collection_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False

    async def add_documents_from_path(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        file_extractor: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, List[str]]:
        """
        Convenience method to load and process documents from a path.

        Args:
            path: Path to file or directory
            recursive: Whether to recursively process subdirectories
            file_extractor: Custom file extractors

        Returns:
            Tuple of (document_count, error_messages)
        """
        try:
            # Load documents
            documents = await self.load_documents(
                path, file_extractor=file_extractor, recursive=recursive
            )

            if not documents:
                return 0, ["No documents found"]

            # Process documents
            await self.process_documents(documents)

            return len(documents), []

        except Exception as e:
            error_msg = f"Failed to add documents from {path}: {e}"
            self.logger.error(error_msg)
            return 0, [error_msg]


# Convenience functions for backward compatibility
async def create_llamaindex_processor(
    collection_name: str = "documents", **kwargs
) -> LlamaIndexProcessor:
    """Create and initialize a LlamaIndex processor."""
    processor = LlamaIndexProcessor(collection_name=collection_name, **kwargs)
    return processor


async def process_documents_with_llamaindex(
    input_path: Union[str, Path], collection_name: str = "documents", **kwargs
) -> LlamaIndexProcessor:
    """Process documents using LlamaIndex and return the processor."""
    processor = await create_llamaindex_processor(
        collection_name=collection_name, **kwargs
    )

    await processor.add_documents_from_path(input_path)
    return processor


__all__ = [
    "LlamaIndexProcessor",
    "create_llamaindex_processor",
    "process_documents_with_llamaindex",
]
