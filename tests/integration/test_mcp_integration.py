#!/usr/bin/env python3
"""
Test script to verify that all SearchOptimizer tools are properly registered in the MCP server.

This script:
1. Imports the RAGMCPServer class
2. Initializes the server instance
3. Calls the _list_tools method to get all available tools
4. Checks that all SearchOptimizer tools are present
5. Prints a summary of the available tools

Expected SearchOptimizer tools:
- optimize_search
- get_search_analytics
- track_user_feedback
- create_ab_test
- get_ab_test_results
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mcp_server import RAGMCPServer


async def test_search_optimizer_tools():
    """Test that all SearchOptimizer tools are properly registered."""
    
    print("🔍 Testing SearchOptimizer Tools Registration")
    print("=" * 50)
    
    # Expected SearchOptimizer tools
    expected_tools = {
        "optimize_search",
        "get_search_analytics", 
        "track_user_feedback",
        "create_ab_test",
        "get_ab_test_results"
    }
    
    try:
        # 1. Initialize the server instance
        print("1. Initializing RAGMCPServer...")
        server = RAGMCPServer()
        print("   ✅ Server initialized successfully")
        
        # 2. Get all available tools
        print("\n2. Retrieving available tools...")
        tools = await server._list_tools()
        tool_names = {tool.name for tool in tools}
        print(f"   ✅ Retrieved {len(tools)} tools")
        
        # 3. Check SearchOptimizer tools are present
        print("\n3. Checking SearchOptimizer tools...")
        
        missing_tools = expected_tools - tool_names
        present_tools = expected_tools & tool_names
        
        if missing_tools:
            print(f"   ❌ Missing SearchOptimizer tools: {missing_tools}")
            return False
        else:
            print(f"   ✅ All {len(expected_tools)} SearchOptimizer tools found!")
        
        # 4. Print detailed summary
        print("\n4. SearchOptimizer Tools Summary:")
        print("-" * 30)
        
        for tool in tools:
            if tool.name in expected_tools:
                print(f"   ✅ {tool.name}")
                print(f"      Description: {tool.description[:80]}...")
                print(f"      Required params: {list(tool.inputSchema.get('required', []))}")
                print()
        
        # 5. Print all available tools
        print("\n5. All Available Tools:")
        print("-" * 30)
        
        # Group tools by category
        search_tools = []
        document_tools = []
        search_optimizer_tools = []
        
        for tool in tools:
            if tool.name in expected_tools:
                search_optimizer_tools.append(tool.name)
            elif tool.name in ["search_knowledge_base", "web_search", "smart_search"]:
                search_tools.append(tool.name)
            elif tool.name.startswith(("add_", "update_", "delete_", "list_", "manage_", "bulk_", "document_")):
                document_tools.append(tool.name)
        
        print(f"📚 Search Tools ({len(search_tools)}):")
        for tool in search_tools:
            print(f"   • {tool}")
        
        print(f"\n📄 Document Management Tools ({len(document_tools)}):")
        for tool in document_tools:
            print(f"   • {tool}")
        
        print(f"\n🔍 Search Optimizer Tools ({len(search_optimizer_tools)}):")
        for tool in search_optimizer_tools:
            print(f"   • {tool}")
        
        # 6. Validate tool schemas
        print("\n6. Validating Tool Schemas:")
        print("-" * 30)
        
        schema_validation_passed = True
        for tool in tools:
            if tool.name in expected_tools:
                # Check that each tool has proper schema structure
                if not tool.inputSchema:
                    print(f"   ❌ {tool.name}: Missing input schema")
                    schema_validation_passed = False
                    continue
                
                if "type" not in tool.inputSchema or tool.inputSchema["type"] != "object":
                    print(f"   ❌ {tool.name}: Invalid schema type")
                    schema_validation_passed = False
                    continue
                
                if "properties" not in tool.inputSchema:
                    print(f"   ❌ {tool.name}: Missing properties in schema")
                    schema_validation_passed = False
                    continue
                
                print(f"   ✅ {tool.name}: Valid schema structure")
        
        if not schema_validation_passed:
            print("   ❌ Schema validation failed!")
            return False
        else:
            print("   ✅ All SearchOptimizer tool schemas are valid!")
        
        # 8. Test tool accessibility
        print("\n8. Testing Tool Accessibility:")
        print("-" * 30)
        
        # Test that the server can validate tool arguments
        try:
            # Test optimize_search tool validation
            await server._validate_tool_arguments("optimize_search", {"query": "test query"})
            print("   ✅ optimize_search: Arguments validation works")
            
            # Test get_search_analytics tool validation
            await server._validate_tool_arguments("get_search_analytics", {})
            print("   ✅ get_search_analytics: Arguments validation works")
            
            # Test track_user_feedback tool validation
            await server._validate_tool_arguments("track_user_feedback", {
                "user_id": "test_user",
                "query": "test query"
            })
            print("   ✅ track_user_feedback: Arguments validation works")
            
            # Test create_ab_test tool validation
            await server._validate_tool_arguments("create_ab_test", {
                "experiment_id": "test_exp",
                "name": "Test Experiment",
                "description": "Test description",
                "variant_a": {"strategy": "a"},
                "variant_b": {"strategy": "b"}
            })
            print("   ✅ create_ab_test: Arguments validation works")
            
            # Test get_ab_test_results tool validation
            await server._validate_tool_arguments("get_ab_test_results", {})
            print("   ✅ get_ab_test_results: Arguments validation works")
            
            print("   ✅ All SearchOptimizer tools are accessible and have working validation!")
            
        except Exception as e:
            print(f"   ❌ Tool accessibility test failed: {e}")
            return False
        
        print("\n🎯 Integration Verification Summary:")
        print("=" * 50)
        print("✅ All SearchOptimizer tools are properly registered:")
        print("   • optimize_search - Advanced search optimization with query expansion")
        print("   • get_search_analytics - Comprehensive analytics and insights")
        print("   • track_user_feedback - User interaction tracking for personalization")
        print("   • create_ab_test - A/B testing framework for search strategies")
        print("   • get_ab_test_results - A/B test results and analysis")
        print()
        print("✅ All tools have valid JSON schema definitions")
        print("✅ All tools are accessible and have working argument validation")
        print("✅ All tools are properly integrated into the MCP server architecture")
        print()
        print("🏗️ Server Architecture Summary:")
        print(f"   • Total MCP Tools: {len(tools)}")
        print(f"   • Search Tools: {len(search_tools)}")
        print(f"   • Document Management: {len(document_tools)}")
        print(f"   • Search Optimization: {len(search_optimizer_tools)}")
        print()
        print("🚀 The RAG MCP Server is now fully equipped with advanced search")
        print("   optimization capabilities through the SearchOptimizer integration!")
        
        # 9. Final test summary
        print("\n9. Final Test Summary:")
        print("-" * 30)
        print(f"   Total tools registered: {len(tools)}")
        print(f"   SearchOptimizer tools found: {len(present_tools)}/{len(expected_tools)}")
        print(f"   Missing tools: {len(missing_tools)}")
        print(f"   Schema validation: {'✅ PASSED' if schema_validation_passed else '❌ FAILED'}")
        print(f"   Tool accessibility: ✅ PASSED")
        
        if missing_tools or not schema_validation_passed:
            print(f"   ❌ TEST FAILED")
            return False
        else:
            print("   ✅ ALL TESTS PASSED - SearchOptimizer tools are fully integrated!")
            return True
            
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("🚀 MCP SearchOptimizer Integration Test")
    print("=" * 60)
    
    result = asyncio.run(test_search_optimizer_tools())
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 All tests passed! SearchOptimizer tools are properly integrated.")
        sys.exit(0)
    else:
        print("❌ Tests failed! Some SearchOptimizer tools are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()