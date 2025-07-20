#!/usr/bin/env python3
"""
Example usage of the dynamic embedding provider system

This script demonstrates how to use the embedding provider system
for both OpenAI and DashScope providers in real-world scenarios.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.embedding_provider import (
    get_embed_model,
    get_embed_model_from_env,
    create_index_with_provider,
    list_providers,
    validate_provider_config
)


async def example_basic_usage():
    """Example 1: Basic usage with different providers."""
    print("📚 Example 1: Basic Embedding Generation")
    print("-" * 40)
    
    text = "LlamaIndex is a powerful framework for building RAG applications."
    
    # Method 1: Explicit provider selection
    try:
        # OpenAI embeddings
        openai_model = get_embed_model("openai")
        openai_embedding = await asyncio.to_thread(openai_model.get_text_embedding, text)
        print(f"✅ OpenAI embedding dimension: {len(openai_embedding)}")
    except Exception as e:
        print(f"❌ OpenAI embedding failed: {e}")
    
    try:
        # DashScope embeddings
        dashscope_model = get_embed_model("dashscope")
        dashscope_embedding = await asyncio.to_thread(dashscope_model.get_text_embedding, text)
        print(f"✅ DashScope embedding dimension: {len(dashscope_embedding)}")
    except Exception as e:
        print(f"❌ DashScope embedding failed: {e}")
    
    print()


async def example_environment_based():
    """Example 2: Environment-based provider selection."""
    print("🌍 Example 2: Environment-Based Provider Selection")
    print("-" * 50)
    
    # Method 2: Use environment variable
    try:
        embed_model = get_embed_model_from_env()
        current_provider = os.getenv("EMBED_PROVIDER", "openai")
        
        text = "Testing environment-based provider selection."
        embedding = await asyncio.to_thread(embed_model.get_text_embedding, text)
        
        print(f"✅ Current provider: {current_provider}")
        print(f"✅ Embedding dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ Environment-based embedding failed: {e}")
    
    print()


async def example_multiple_indexes():
    """Example 3: Creating multiple indexes with different providers."""
    print("📖 Example 3: Multiple Indexes with Different Providers")
    print("-" * 55)
    
    from llama_index.core import Document
    
    # Sample documents
    documents = [
        Document(text="The quick brown fox jumps over the lazy dog."),
        Document(text="LlamaIndex provides tools for building LLM applications."),
        Document(text="Vector databases enable semantic search capabilities.")
    ]
    
    # Create indexes with different providers
    indexes = {}
    
    for provider in ["openai", "dashscope"]:
        # Check if provider is available
        validation = validate_provider_config(provider)
        
        if validation["valid"]:
            try:
                index = create_index_with_provider(
                    provider=provider,
                    documents=documents
                )
                indexes[provider] = index
                print(f"✅ Created index with {provider} provider")
                
                # Test querying
                query_engine = index.as_query_engine()
                response = await asyncio.to_thread(
                    query_engine.query, 
                    "What is LlamaIndex?"
                )
                print(f"   Query response: {str(response)[:100]}...")
                
            except Exception as e:
                print(f"❌ Failed to create {provider} index: {e}")
        else:
            print(f"⚠️ Skipping {provider}: {validation.get('error', 'Not configured')}")
    
    print(f"\n✅ Successfully created {len(indexes)} indexes")
    print()


async def example_custom_models():
    """Example 4: Using custom models with providers."""
    print("⚙️ Example 4: Custom Models with Providers")
    print("-" * 42)
    
    # Test different models for each provider
    provider_models = {
        "openai": [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        "dashscope": [
            "text-embedding-v1",
            "text-embedding-v2"
        ]
    }
    
    text = "Testing custom embedding models."
    
    for provider, models in provider_models.items():
        # Check if provider is available
        validation = validate_provider_config(provider)
        
        if not validation["valid"]:
            print(f"⚠️ Skipping {provider}: {validation.get('error', 'Not configured')}")
            continue
        
        print(f"🔧 Testing {provider} models:")
        
        for model in models:
            try:
                embed_model = get_embed_model(provider, model=model)
                embedding = await asyncio.to_thread(embed_model.get_text_embedding, text)
                print(f"  ✅ {model}: dimension {len(embedding)}")
                
            except Exception as e:
                print(f"  ❌ {model}: {e}")
    
    print()


def example_configuration_check():
    """Example 5: Configuration validation and provider info."""
    print("🔍 Example 5: Configuration Check and Provider Info")
    print("-" * 52)
    
    # List available providers
    providers = list_providers()
    print("Available providers:")
    for name, info in providers.items():
        print(f"  📋 {name}: {info['description']}")
        print(f"     Default model: {info['default_model']}")
        print(f"     API key env: {info['api_key_env']}")
    
    print("\nProvider configuration status:")
    for provider in providers.keys():
        validation = validate_provider_config(provider)
        status = "✅ Ready" if validation["valid"] else "❌ Not configured"
        print(f"  {status} {provider}")
        if not validation["valid"]:
            print(f"    Error: {validation.get('error', 'Unknown error')}")
    
    print()


async def example_provider_switching():
    """Example 6: Runtime provider switching."""
    print("🔄 Example 6: Runtime Provider Switching")
    print("-" * 38)
    
    # Save original provider setting
    original_provider = os.getenv("EMBED_PROVIDER")
    
    text = "Testing runtime provider switching."
    
    for provider in ["openai", "dashscope"]:
        # Check if provider is available
        validation = validate_provider_config(provider)
        
        if not validation["valid"]:
            print(f"⚠️ Skipping {provider}: {validation.get('error', 'Not configured')}")
            continue
        
        # Switch provider via environment
        os.environ["EMBED_PROVIDER"] = provider
        
        try:
            embed_model = get_embed_model_from_env()
            embedding = await asyncio.to_thread(embed_model.get_text_embedding, text)
            print(f"✅ {provider}: Successfully generated embedding (dim: {len(embedding)})")
            
        except Exception as e:
            print(f"❌ {provider}: {e}")
    
    # Restore original setting
    if original_provider is not None:
        os.environ["EMBED_PROVIDER"] = original_provider
    elif "EMBED_PROVIDER" in os.environ:
        del os.environ["EMBED_PROVIDER"]
    
    print()


async def main():
    """Run all examples."""
    print("🚀 Dynamic Embedding Provider Examples")
    print("=" * 60)
    print()
    
    # Check what providers are available
    print("🔧 Checking provider availability...")
    available_providers = []
    for provider in ["openai", "dashscope"]:
        validation = validate_provider_config(provider)
        if validation["valid"]:
            available_providers.append(provider)
    
    print(f"Available providers: {', '.join(available_providers) if available_providers else 'None'}")
    
    if not available_providers:
        print("\n❌ No providers are configured. Please set up API keys:")
        print("  - OPENAI_API_KEY for OpenAI provider")
        print("  - DASHSCOPE_API_KEY for DashScope provider")
        return
    
    print(f"✅ Found {len(available_providers)} configured provider(s)\n")
    
    # Run examples
    await example_basic_usage()
    await example_environment_based()
    await example_multiple_indexes()
    await example_custom_models()
    example_configuration_check()
    await example_provider_switching()
    
    print("=" * 60)
    print("🎉 All examples completed successfully!")
    
    print("\n💡 Key takeaways:")
    print("  1. Use get_embed_model(provider) for explicit provider selection")
    print("  2. Use get_embed_model_from_env() for environment-based selection")
    print("  3. Use create_index_with_provider() for per-index provider selection")
    print("  4. Use validate_provider_config() to check configuration")
    print("  5. Set EMBED_PROVIDER environment variable for global defaults")


if __name__ == "__main__":
    asyncio.run(main())