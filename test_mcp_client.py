#!/usr/bin/env python3
"""
Simple MCP Client for testing the RAG MCP Server
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict

class MCPClient:
    def __init__(self, command: list):
        self.command = command
        self.process = None
        self.request_id = 0

    async def start(self):
        """Start the MCP server process."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"Started MCP server with command: {' '.join(self.command)}")

    async def send_message(self, message: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Send a JSON-RPC message to the server and get response."""
        if not self.process:
            raise RuntimeError("Server not started")

        try:
            # Send message
            message_str = json.dumps(message) + '\n'
            self.process.stdin.write(message_str.encode())
            await self.process.stdin.drain()
            
            print(f"→ Sent: {message['method']} (ID: {message.get('id', 'N/A')})")

            # Read response with timeout
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
            
            if not response_line:
                stderr_output = await self.process.stderr.read()
                print(f"No response received. Stderr: {stderr_output.decode()}")
                return {}
            
            response = json.loads(response_line.decode().strip())
            print(f"← Received: {response.get('result', response.get('error', 'Unknown'))}")
            return response
            
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout waiting for response to {message['method']}")
            return {"error": "timeout"}
        except json.JSONDecodeError as e:
            print(f"Failed to decode response: {response_line.decode()}")
            print(f"JSON error: {e}")
            return {"error": "json_decode_error"}

    async def initialize(self):
        """Initialize the MCP connection."""
        self.request_id += 1
        init_message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = await self.send_message(init_message)
        
        # Send initialized notification
        initialized_message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self.send_message(initialized_message)
        
        return response

    async def list_tools(self):
        """List available tools."""
        self.request_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/list",
            "params": {}
        }
        return await self.send_message(message)

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """Call a tool with given arguments."""
        self.request_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        return await self.send_message(message)

    async def close(self):
        """Close the connection."""
        if self.process:
            self.process.stdin.close()
            await self.process.wait()

async def test_rag_server():
    """Test the RAG MCP server."""
    
    # Server command (using Docker)
    server_command = [
        "docker", "run", "--rm", "-i", 
        "--env-file", "/Users/renxin/project/mcp/.env",
        "rag-mcp-server:production"
    ]
    
    client = MCPClient(server_command)
    
    try:
        print("🚀 Starting RAG MCP Server test...")
        await client.start()
        
        print("\n📡 Initializing connection...")
        init_response = await client.initialize()
        
        if "result" not in init_response:
            print("❌ Failed to initialize")
            return
        
        print("✅ Connection initialized successfully")
        
        print("\n🛠️ Listing available tools...")
        tools_response = await client.list_tools()
        
        if "result" not in tools_response:
            print("❌ Failed to list tools")
            return
        
        tools = tools_response["result"]["tools"]
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description'][:60]}...")
        
        # Test 1: Search knowledge base (should return empty since no docs)
        print("\n🔍 Testing knowledge base search...")
        search_response = await client.call_tool("search_knowledge_base", {
            "query": "test query",
            "top_k": 3
        })
        
        if "result" in search_response:
            print("✅ Knowledge base search completed")
            content = search_response["result"]["content"]
            if content and len(content) > 0:
                print(f"Response: {content[0]['text'][:200]}...")
        else:
            print("❌ Knowledge base search failed")
            if "error" in search_response:
                print(f"Error: {search_response['error']}")
        
        # Test 2: Web search
        print("\n🌐 Testing web search...")
        web_search_response = await client.call_tool("web_search", {
            "query": "Python programming",
            "max_results": 2
        })
        
        if "result" in web_search_response:
            print("✅ Web search completed")
            content = web_search_response["result"]["content"]
            if content and len(content) > 0:
                print(f"Response: {content[0]['text'][:200]}...")
        else:
            print("❌ Web search failed")
            if "error" in web_search_response:
                print(f"Error: {web_search_response['error']}")
        
        # Test 3: Add document
        print("\n📄 Testing document addition...")
        doc_response = await client.call_tool("add_document", {
            "content": "This is a test document about Python programming. Python is a versatile programming language.",
            "metadata": {"source": "test", "type": "text"},
            "tags": ["python", "programming", "test"]
        })
        
        if "result" in doc_response:
            print("✅ Document addition completed")
            content = doc_response["result"]["content"]
            if content and len(content) > 0:
                print(f"Response: {content[0]['text'][:200]}...")
        else:
            print("❌ Document addition failed")
            if "error" in doc_response:
                print(f"Error: {doc_response['error']}")
        
        # Test 4: Search again after adding document
        print("\n🔍 Testing knowledge base search after adding document...")
        search_response2 = await client.call_tool("search_knowledge_base", {
            "query": "Python programming",
            "top_k": 3
        })
        
        if "result" in search_response2:
            print("✅ Second knowledge base search completed")
            content = search_response2["result"]["content"]
            if content and len(content) > 0:
                print(f"Response: {content[0]['text'][:200]}...")
        else:
            print("❌ Second knowledge base search failed")
            if "error" in search_response2:
                print(f"Error: {search_response2['error']}")
        
        print("\n✅ All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 Closing connection...")
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_rag_server())