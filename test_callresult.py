#!/usr/bin/env python3
"""
Test CallToolResult iteration behavior
"""

from mcp.types import CallToolResult, TextContent

def test_callresult_behavior():
    """Test how CallToolResult behaves when iterated."""
    
    # Create test content
    content = [TextContent(type="text", text="Test response")]
    result = CallToolResult(content=content)
    
    print("CallToolResult created successfully")
    print(f"Content: {result.content}")
    print(f"Type: {type(result)}")
    
    # Test iteration behavior
    print("\nTesting iteration:")
    try:
        for item in result:
            print(f"Iteration item: {item} (type: {type(item)})")
    except Exception as e:
        print(f"Iteration error: {e}")
    
    # Test model_dump
    print(f"\nModel dump: {result.model_dump()}")
    
    # Test conversion back to expected format
    print("\nTesting conversion:")
    if hasattr(result, 'content'):
        print(f"Content attribute exists: {result.content}")
        if isinstance(result.content, list):
            print("Content is a list")
            if result.content and hasattr(result.content[0], 'type'):
                print(f"First content item type: {result.content[0].type}")
    
    # Test what happens if we accidentally iterate over CallToolResult
    print("\nTesting problematic iteration (what causes the bug):")
    items = list(result)
    print(f"Items from list(result): {items}")
    
    print("\nThis demonstrates the bug - CallToolResult.__iter__ returns attribute tuples!")

if __name__ == "__main__":
    test_callresult_behavior()