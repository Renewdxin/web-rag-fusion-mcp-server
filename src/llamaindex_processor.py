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
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    BaseRetriever,
)

# BM25Retriever import
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except ImportError:
    try:
        from llama_index.core.retrievers.bm25 import BM25Retriever
    except ImportError:
        # Fallback: use base retriever
        BM25Retriever = BaseRetriever

# FusionRetriever import
try:
    from llama_index.core.retrievers import QueryFusionRetriever as FusionRetriever
except ImportError:
    try:
        from llama_index.retrievers.fusion import QueryFusionRetriever as FusionRetriever
    except ImportError:
        # Fallback: create a simple fusion retriever class
        from llama_index.core.retrievers import BaseRetriever
        FusionRetriever = BaseRetriever

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import config
from src.embedding_provider import get_embed_model_from_env, EmbeddingProviderError


class RAGEngine:
    """
    Advanced RAG Engine with hybrid search capabilities.
    
    Features:
    - Vector similarity search using ChromaDB
    - BM25 text-based search
    - Query fusion for enhanced retrieval
    - Metadata filtering and personalization
    - Efficient query engine caching
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
        similarity_top_k: int = 10,
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            similarity_top_k: Number of similar documents to retrieve
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        
        self.logger = logging.getLogger(f"{__name__}.RAGEngine")
        
        # Initialize LlamaIndex settings with global config
        self._setup_settings(embedding_model, llm_model)
        
        # Initialize ChromaDB and vector store
        self._setup_chroma()
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Load or create vector index
        self.index = self._load_or_create_index()
        
        # Load all documents for BM25 retriever
        self.documents = self._load_all_documents()
        
        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
        )
        
        # Cache for query engines to avoid rebuilding
        self._query_engine_cache = {}
        
        self.logger.info(f"RAGEngine initialized with {len(self.documents)} documents")
    
    def _setup_settings(self, embedding_model: str, llm_model: str):
        """Setup LlamaIndex global settings using dynamic embedding provider."""
        # Configure embedding model using the new provider system
        try:
            # Use the provider specified in config (with fallback to openai)
            Settings.embed_model = get_embed_model_from_env(
                provider_env_var="EMBED_PROVIDER",
                fallback_provider="openai"
            )
            
            # Override model if specified in constructor
            if embedding_model != "text-embedding-3-small":  # If not default
                # Get the current provider from config
                provider = config.EMBED_PROVIDER
                if provider == "dashscope":
                    # Re-create with custom model for DashScope
                    from src.embedding_provider import get_embed_model
                    Settings.embed_model = get_embed_model(
                        provider="dashscope",
                        model=embedding_model,
                        api_key=config.DASHSCOPE_API_KEY or config.OPENAI_API_KEY
                    )
                else:
                    # Re-create with custom model for OpenAI
                    from src.embedding_provider import get_embed_model
                    Settings.embed_model = get_embed_model(
                        provider="openai", 
                        model=embedding_model,
                        api_key=config.OPENAI_API_KEY
                    )
            
            self.logger.info(f"Configured embedding provider: {config.EMBED_PROVIDER} with model: {embedding_model}")
            
        except EmbeddingProviderError as e:
            raise ValueError(f"Failed to initialize embedding provider: {e}")
        
        # Configure LLM based on provider or explicit configuration
        try:
            # Check if using DashScope LLM (inferred from base URL or provider)
            is_dashscope_llm = (
                config.OPENAI_BASE_URL and 'dashscope' in config.OPENAI_BASE_URL.lower()
            ) or config.EMBED_PROVIDER == "dashscope"
            
            if is_dashscope_llm:
                # Use DashScope LLM
                api_key = config.DASHSCOPE_API_KEY or config.OPENAI_API_KEY
                if not api_key:
                    raise ValueError("DashScope API key is required for DashScope LLM")
                
                Settings.llm = DashScope(
                    model_name=llm_model,
                    api_key=api_key,
                    temperature=0.1
                )
                self.logger.info(f"Configured DashScope LLM: {llm_model}")
                
            else:
                # Use OpenAI LLM
                if not config.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key is required for OpenAI LLM")
                
                # Prepare OpenAI client arguments
                openai_kwargs = {
                    "api_key": config.OPENAI_API_KEY,
                    "model": llm_model,
                    "temperature": 0.1,
                }
                
                # Add base_url if configured (for proxy support)
                if config.OPENAI_BASE_URL:
                    openai_kwargs["base_url"] = config.OPENAI_BASE_URL
                
                Settings.llm = OpenAI(**openai_kwargs)
                self.logger.info(f"Configured OpenAI LLM: {llm_model}")
                
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
        
        # Set chunk settings
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
        self.logger.info(f"RAG settings configured successfully")
    
    def _setup_chroma(self):
        """Setup ChromaDB client and vector store."""
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
            
            # Create ChromaVectorStore for LlamaIndex
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            self.logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            error_msg = f"Failed to setup ChromaDB: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index or create new one."""
        try:
            # Try to load existing index from vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            doc_count = self.collection.count()
            self.logger.info(f"Loaded existing index with {doc_count} documents")
            return index
            
        except Exception:
            # Create new empty index
            index = VectorStoreIndex([], storage_context=self.storage_context)
            self.logger.info("Created new empty index")
            return index
    
    def _load_all_documents(self) -> List[Document]:
        """Load all documents from the index for BM25 retrieval."""
        try:
            # Get all documents from the vector store
            all_nodes = self.index.docstore.docs
            documents = []
            
            for node_id, node in all_nodes.items():
                doc = Document(
                    text=node.text,
                    metadata=node.metadata or {},
                    doc_id=node_id
                )
                documents.append(doc)
            
            self.logger.info(f"Loaded {len(documents)} documents for BM25")
            return documents
            
        except Exception as e:
            self.logger.warning(f"Failed to load documents for BM25: {e}")
            return []
    
    def _get_query_engine(self, search_type: str = "vector", **kwargs) -> RetrieverQueryEngine:
        """Get cached query engine or create new one."""
        # Simplified cache key using only search type and top_k
        top_k = kwargs.get("top_k", self.similarity_top_k)
        cache_key = f"{search_type}_{top_k}"
        
        if cache_key in self._query_engine_cache:
            return self._query_engine_cache[cache_key]
        
        # Create retriever based on search type
        retriever = self._create_retriever(search_type, top_k)
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(
                response_mode=kwargs.get("response_mode", ResponseMode.COMPACT)
            ),
            node_postprocessors=[
                SimilarityPostprocessor(
                    similarity_cutoff=kwargs.get("similarity_cutoff", 0.7)
                )
            ],
        )
        
        # Cache the query engine
        self._query_engine_cache[cache_key] = query_engine
        return query_engine
    
    def _create_retriever(self, search_type: str, top_k: int):
        """Create retriever with fallback handling."""
        if search_type == "vector":
            return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        
        elif search_type == "bm25" and self.documents:
            try:
                return BM25Retriever.from_defaults(
                    docstore=self.index.docstore, similarity_top_k=top_k
                )
            except Exception:
                # Fallback to vector retriever
                return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        
        elif search_type == "hybrid" and self.documents:
            try:
                vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=self.index.docstore, similarity_top_k=top_k
                )
                return FusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever], similarity_top_k=top_k
                )
            except Exception:
                # Fallback to vector retriever
                return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        
        else:
            # Default to vector retriever
            return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
    
    async def search(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using specified search type.
        
        Args:
            query: Search query
            search_type: "vector", "bm25", or "hybrid"
            top_k: Number of results to return
            filters: Metadata filters (basic support)
            **kwargs: Additional parameters for query engine
            
        Returns:
            List of search results with content, metadata, and scores
        """
        try:
            top_k = top_k or self.similarity_top_k
            query_engine = self._get_query_engine(search_type, top_k=top_k, **kwargs)
            
            # Execute query
            response = await asyncio.to_thread(query_engine.query, query)
            
            # Extract results
            results = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    # Apply basic metadata filtering if specified
                    if filters and not self._matches_filters(node.node.metadata or {}, filters):
                        continue
                    
                    result = {
                        "content": node.node.text,
                        "metadata": node.node.metadata or {},
                        "score": getattr(node, "score", 0.0),
                        "node_id": node.node.node_id,
                    }
                    results.append(result)
            
            self.logger.info(f"Search ({search_type}) completed: {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Search failed: {e}"
            self.logger.error(error_msg)
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Basic metadata filtering."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    async def query(
        self,
        query_text: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        **kwargs
    ) -> Any:
        """
        Query the knowledge base and return a response object with source nodes.
        
        This method matches the interface expected by mcp_server.py.
        
        Args:
            query_text: The query string
            user_preferences: User preferences for filtering (converted to filters)
            search_type: Type of search to use ("vector", "bm25", or "hybrid")
            **kwargs: Additional parameters
            
        Returns:
            Response object with 'response' and 'source_nodes' attributes
        """
        try:
            # Convert user_preferences to filters for backward compatibility
            filters = None
            if user_preferences:
                filters = {k: v for k, v in user_preferences.items() if k != "tags"}
                if "tags" in user_preferences:
                    # For now, we ignore tags filtering as it requires more complex implementation
                    pass
            
            # Get query engine and execute query
            query_engine = self._get_query_engine(search_type, **kwargs)
            response = await asyncio.to_thread(query_engine.query, query_text)
            
            # Return response object that matches expected interface
            return response
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            # Return a mock response object with empty results
            class EmptyResponse:
                def __init__(self):
                    self.response = ""
                    self.source_nodes = []
                    
            return EmptyResponse()
    
    async def query_with_response(
        self,
        query: str,
        search_type: str = "hybrid",
        **kwargs
    ) -> str:
        """
        Query the knowledge base and get a synthesized response.
        
        Args:
            query: Question to ask
            search_type: Type of search to use
            **kwargs: Additional parameters
            
        Returns:
            Synthesized response text
        """
        try:
            query_engine = self._get_query_engine(search_type, **kwargs)
            response = await asyncio.to_thread(query_engine.query, query)
            return str(response)
            
        except Exception as e:
            error_msg = f"Query failed: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
    
    async def add_documents_from_path(
        self, path: Union[str, Path]
    ) -> Tuple[int, List[str]]:
        """
        Add documents from a file or directory path.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Tuple of (number of documents added, list of error messages)
        """
        try:
            path = Path(path)
            if not path.exists():
                return 0, [f"Path does not exist: {path}"]
            
            # Load documents using SimpleDirectoryReader
            if path.is_file():
                loader = SimpleDirectoryReader(
                    input_files=[str(path)],
                    filename_as_id=True
                )
            else:
                loader = SimpleDirectoryReader(
                    input_dir=str(path),
                    filename_as_id=True,
                    recursive=True
                )
            
            documents = loader.load_data()
            
            if not documents:
                return 0, ["No documents found"]
            
            # Parse nodes and add to index
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Add nodes to index
            for node in nodes:
                self.index.insert(node)
            
            # Refresh documents for BM25
            self.documents = self._load_all_documents()
            
            # Clear query engine cache since index changed
            self._query_engine_cache.clear()
            
            self.logger.info(f"Added {len(documents)} documents from {path}")
            return len(documents), []
            
        except Exception as e:
            error_msg = f"Failed to add documents from {path}: {e}"
            self.logger.error(error_msg)
            return 0, [error_msg]
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add a list of documents to the index."""
        try:
            # Parse nodes and add to index
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Add nodes to index
            for node in nodes:
                self.index.insert(node)
            
            # Refresh documents for BM25
            self.documents = self._load_all_documents()
            
            # Clear query engine cache since index changed
            self._query_engine_cache.clear()
            
            self.logger.info(f"Added {len(documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG engine."""
        try:
            return {
                "collection_name": self.collection_name,
                "document_count": len(self.documents),
                "chroma_document_count": self.collection.count(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "similarity_top_k": self.similarity_top_k,
                "index_initialized": self.index is not None,
                "documents_loaded": len(self.documents) > 0,
                "cached_query_engines": len(self._query_engine_cache),
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


__all__ = ["RAGEngine"]