#!/usr/bin/env python3
"""
Test script for dynamic embedding providers

This script tests the embedding provider system with both OpenAI and DashScope,
verifying that embeddings can be generated and that provider switching works correctly.

Usage:
    python test_embedding_providers.py

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI tests
    DASHSCOPE_API_KEY: Required for DashScope tests
    EMBED_PROVIDER: Provider to use by default (openai/dashscope)
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.embedding_provider import (
    get_embed_model,
    get_embed_model_from_env,
    create_index_with_provider,
    list_providers,
    validate_provider_config,
    EmbeddingProviderError,
    PROVIDER_INFO
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_provider_info():
    """Test provider information functions."""
    print("🔍 Testing provider information functions...")
    
    # Test list_providers
    providers = list_providers()
    print(f"✅ Available providers: {list(providers.keys())}")
    
    for provider_name, info in providers.items():
        print(f"  📋 {provider_name}: {info['description']}")
        print(f"     Default model: {info['default_model']}")
        print(f"     API key env: {info['api_key_env']}")
    
    print()


def test_provider_validation():
    """Test provider configuration validation."""
    print("🔧 Testing provider configuration validation...")
    
    for provider in ["openai", "dashscope", "invalid_provider"]:
        validation = validate_provider_config(provider)
        status = "✅" if validation["valid"] else "❌"
        print(f"{status} Provider '{provider}': {validation}")
    
    print()


async def test_embedding_generation(provider: str, test_texts: List[str]):
    """Test embedding generation for a specific provider."""
    print(f"🧠 Testing {provider} embedding generation...")
    
    try:
        # Get embedding model
        embed_model = get_embed_model(provider)
        print(f"✅ Successfully initialized {provider} embedding model")
        
        # Test embedding generation
        for i, text in enumerate(test_texts, 1):
            try:
                # Generate embedding
                embedding = await asyncio.to_thread(embed_model.get_text_embedding, text)
                
                # Verify embedding properties
                if isinstance(embedding, list) and len(embedding) > 0:
                    print(f"  ✅ Text {i}: Generated embedding of dimension {len(embedding)}")
                    print(f"     First few values: {embedding[:3]}...")
                else:
                    print(f"  ❌ Text {i}: Invalid embedding format")
                    
            except Exception as e:
                print(f"  ❌ Text {i}: Failed to generate embedding - {e}")
        
        return True
        
    except EmbeddingProviderError as e:
        print(f"❌ {provider} provider error: {e}")
        return False
    except Exception as e:
        print(f"❌ {provider} unexpected error: {e}")
        return False


async def test_index_creation(provider: str):
    """Test VectorStoreIndex creation with specific provider."""
    print(f"📚 Testing {provider} index creation...")
    
    try:
        # Sample documents
        from llama_index.core import Document
        
        documents = [
            Document(text="The quick brown fox jumps over the lazy dog."),
            Document(text="LlamaIndex is a powerful framework for building RAG applications."),
            Document(text="Python is a versatile programming language.")
        ]
        
        # Create index with specific provider
        index = create_index_with_provider(
            provider=provider,
            documents=documents
        )
        
        print(f"✅ Successfully created index with {provider} provider")
        print(f"   Index type: {type(index)}")
        
        # Test querying (simple check)
        query_engine = index.as_query_engine()
        response = await asyncio.to_thread(query_engine.query, "What is LlamaIndex?")
        
        if response and str(response).strip():
            print(f"✅ Index query successful")
            print(f"   Response preview: {str(response)[:100]}...")
        else:
            print(f"❌ Index query returned empty response")
        
        return True
        
    except Exception as e:
        print(f"❌ {provider} index creation failed: {e}")
        return False


async def test_provider_switching():
    """Test dynamic provider switching via environment variables."""
    print("🔄 Testing dynamic provider switching...")
    
    # Save original environment
    original_provider = os.getenv("EMBED_PROVIDER")
    
    try:
        # Test switching to each provider
        for provider in ["openai", "dashscope"]:
            print(f"  🔀 Switching to {provider}...")
            
            # Set environment variable
            os.environ["EMBED_PROVIDER"] = provider
            
            try:
                # Get embedding model from environment
                embed_model = get_embed_model_from_env()
                print(f"    ✅ Successfully loaded {provider} from environment")
                
                # Test a simple embedding
                embedding = await asyncio.to_thread(
                    embed_model.get_text_embedding, 
                    "Test embedding for provider switching"
                )
                
                if isinstance(embedding, list) and len(embedding) > 0:
                    print(f"    ✅ Embedding generation successful (dim: {len(embedding)})")
                else:
                    print(f"    ❌ Invalid embedding generated")
                    
            except EmbeddingProviderError as e:
                print(f"    ⚠️ {provider} not available: {e}")
            except Exception as e:
                print(f"    ❌ {provider} error: {e}")
    
    finally:
        # Restore original environment
        if original_provider is not None:
            os.environ["EMBED_PROVIDER"] = original_provider
        elif "EMBED_PROVIDER" in os.environ:
            del os.environ["EMBED_PROVIDER"]
    
    print()


async def test_error_handling():
    """Test error handling scenarios."""
    print("⚠️ Testing error handling...")
    
    # Test invalid provider
    try:
        get_embed_model("invalid_provider")
        print("❌ Should have raised ValueError for invalid provider")
    except ValueError as e:
        print(f"✅ Correctly caught invalid provider error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
    
    # Test missing API key (temporarily unset)
    original_openai_key = os.getenv("OPENAI_API_KEY")
    original_dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    
    try:
        # Temporarily remove API keys
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "DASHSCOPE_API_KEY" in os.environ:
            del os.environ["DASHSCOPE_API_KEY"]
        
        # Test OpenAI without API key
        try:
            get_embed_model("openai")
            print("❌ Should have raised EmbeddingProviderError for missing OpenAI key")
        except EmbeddingProviderError as e:
            print(f"✅ Correctly caught missing OpenAI API key: {e}")
        
        # Test DashScope without API key
        try:
            get_embed_model("dashscope")
            print("❌ Should have raised EmbeddingProviderError for missing DashScope key")
        except EmbeddingProviderError as e:
            print(f"✅ Correctly caught missing DashScope API key: {e}")
    
    finally:
        # Restore API keys
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        if original_dashscope_key:
            os.environ["DASHSCOPE_API_KEY"] = original_dashscope_key
    
    print()


async def main():
    """Run comprehensive embedding provider tests."""
    print("🚀 Starting Embedding Provider Tests\n")
    print("=" * 60)
    
    # Test texts for embedding generation
    test_texts = [
        "Hello, world!",
        "This is a test of the embedding system.",
        "Artificial intelligence and machine learning are transforming the world."
    ]
    
    # 1. Test provider info functions
    test_provider_info()
    
    # 2. Test provider validation
    test_provider_validation()
    
    # 3. Test embedding generation for each provider
    providers_tested = []
    
    for provider in ["openai", "dashscope"]:
        # Check if provider is configured
        validation = validate_provider_config(provider)
        
        if validation["valid"]:
            success = await test_embedding_generation(provider, test_texts)
            if success:
                providers_tested.append(provider)
                # Also test index creation
                await test_index_creation(provider)
        else:
            print(f"⚠️ Skipping {provider} tests: {validation.get('error', 'Not configured')}")
        
        print()
    
    # 4. Test provider switching (only if multiple providers available)
    if len(providers_tested) > 1:
        await test_provider_switching()
    else:
        print("⚠️ Skipping provider switching tests (need multiple configured providers)\n")
    
    # 5. Test error handling
    await test_error_handling()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"✅ Providers successfully tested: {', '.join(providers_tested) if providers_tested else 'None'}")
    
    if not providers_tested:
        print("❌ No providers were successfully tested. Please check your API key configuration.")
        print("\nRequired environment variables:")
        for provider, info in PROVIDER_INFO.items():
            print(f"  - {info['api_key_env']}: For {provider} provider")
    else:
        print("🎉 Embedding provider system is working correctly!")
    
    print("\n🏁 Tests completed!")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())