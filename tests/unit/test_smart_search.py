#!/usr/bin/env python3
"""
Test script to verify the smart_search tool implementation.
"""

import asyncio
import sys
import os

# Add project root to path so we can import config and src modules
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from mcp_server import RAGMCPServer


async def test_smart_search():
    """Test the smart_search tool with various parameters."""
    
    try:
        # Initialize server
        server = RAGMCPServer()
        
        # Test parameters for smart search
        test_cases = [
            {
                "name": "Basic smart search",
                "args": {
                    "query": "machine learning algorithms"
                }
            },
            {
                "name": "Smart search with custom threshold", 
                "args": {
                    "query": "artificial intelligence trends",
                    "similarity_threshold": 0.8,
                    "local_top_k": 3,
                    "web_max_results": 3
                }
            },
            {
                "name": "Smart search with low threshold",
                "args": {
                    "query": "latest technology news",
                    "similarity_threshold": 0.5,
                    "local_top_k": 2,
                    "web_max_results": 5
                }
            }
        ]
        
        print("🧠 Testing Smart Search Implementation")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print("-" * 30)
            
            try:
                # Call the smart search internal method directly
                result = await server._smart_search_internal(
                    request_id=f"test_{i}",
                    **test_case['args']
                )
                
                print(f"✅ Test passed - Response length: {len(result[0].text) if result else 0} characters")
                print(f"📊 Parameters used: {test_case['args']}")
                
                # Show first 200 characters of response
                if result and result[0].text:
                    preview = result[0].text[:200] + "..." if len(result[0].text) > 200 else result[0].text
                    print(f"📝 Response preview: {preview}")
                
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("🎯 Smart search testing completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_smart_search())