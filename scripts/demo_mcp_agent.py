#!/usr/bin/env python3
"""
Demo script for MCP Agent with Neo4j Knowledge Graph.

This script demonstrates the full pipeline:
1. Connect to Neo4j and verify data
2. Create MCP Agent with Qwen2.5-7B
3. Run agentic queries that use MCP tools to query the knowledge graph
4. Show tool call traces and final answers

Prerequisites:
- Neo4j running at bolt://localhost:7687
- Knowledge graph data loaded (run migrate_faiss_to_neo4j.py first)
- GPU with ~6GB VRAM for Qwen2.5-7B with 4-bit quantization

Usage:
    # Source the local LLM config first
    source export_local_llm.sh
    
    # Run demo
    python scripts/demo_mcp_agent.py
    
    # Run with custom Neo4j connection
    python scripts/demo_mcp_agent.py --neo4j-password mypassword
    
    # Run with verbose output to see tool calls
    python scripts/demo_mcp_agent.py --verbose
    
    # Test without loading the full LLM (uses API backend if configured)
    python scripts/demo_mcp_agent.py --lite
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_neo4j_connection(uri: str, user: str, password: str) -> bool:
    """Test Neo4j connection and show stats."""
    try:
        from rag.neo4j_store import Neo4jStore
        
        print("\n" + "=" * 60)
        print("Step 1: Testing Neo4j Connection")
        print("=" * 60)
        
        store = Neo4jStore(
            uri=uri,
            user=user,
            password=password,
            create_indexes=False,
        )
        
        print(f"  Connected to: {uri}")
        print(f"  Total triplets: {store.size}")
        
        # Show sample triplets
        if store.size > 0:
            print("\n  Sample triplets:")
            with store.driver.session() as session:
                result = session.run("""
                    MATCH (t:Triplet)
                    RETURN t.subject AS s, t.predicate AS p, t.object AS o
                    LIMIT 5
                """)
                for record in result:
                    print(f"    ({record['s']}, {record['p']}, {record['o']})")
        
        store.close()
        return True
        
    except Exception as e:
        print(f"  ERROR: Failed to connect to Neo4j: {e}")
        print("  Make sure Neo4j is running and accessible.")
        return False


def run_mcp_agent_demo(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    verbose: bool = False,
    lite_mode: bool = False,
):
    """Run the MCP Agent demo."""
    
    print("\n" + "=" * 60)
    print("Step 2: Creating MCP Agent")
    print("=" * 60)
    
    # Get model configuration from environment
    llm_model = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    embedder_model = os.environ.get("LOCAL_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    quantize = os.environ.get("LOCAL_LLM_QUANTIZE", "4bit")
    
    print(f"  LLM Model: {llm_model}")
    print(f"  Embedder: {embedder_model}")
    print(f"  Quantization: {quantize}")
    print(f"  Mode: {'Lite (API)' if lite_mode else 'Full (Local GPU)'}")
    
    if lite_mode:
        # Use lightweight agent with API backend
        from rag.neo4j_store import Neo4jStore
        from rag.embedder import get_embedder
        from rag.mcp_agent import MCPAgentLite
        
        neo4j_store = Neo4jStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )
        embedder = get_embedder(model_name=embedder_model)
        agent = MCPAgentLite(
            neo4j_store=neo4j_store,
            embedder=embedder,
            max_iterations=3,
        )
    else:
        # Use full agent with local LLM
        from rag.mcp_agent import create_mcp_agent
        
        agent = create_mcp_agent(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            llm_model=llm_model,
            quantize=quantize,
            max_iterations=3,
        )
    
    print("  Agent created successfully!")
    
    # Demo queries
    demo_queries = [
        "What is in the knowledge graph?",
        "Tell me about the main entities in the database.",
        "What relationships exist between entities?",
    ]
    
    print("\n" + "=" * 60)
    print("Step 3: Running Demo Queries")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Query {i}: {query} ---\n")
        
        try:
            result = agent.run(query, verbose=verbose)
            
            print(f"Answer: {result.answer}")
            print(f"\nIterations: {result.iterations}")
            print(f"Tool calls: {len(result.tool_calls)}")
            
            if result.tool_calls:
                print("\nTool call trace:")
                for tc in result.tool_calls:
                    print(f"  [{tc['iteration']}] {tc['tool']}({tc['arguments']})")
                    if 'result' in tc and 'num_facts' in tc['result']:
                        print(f"      -> Found {tc['result']['num_facts']} facts")
            
            if result.warning:
                print(f"\nWarning: {result.warning}")
                
        except Exception as e:
            print(f"ERROR: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def run_simple_search_demo(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
):
    """Run a simple search demo without loading the full LLM."""
    
    print("\n" + "=" * 60)
    print("Simple Search Demo (No LLM Required)")
    print("=" * 60)
    
    from rag.neo4j_store import Neo4jStore
    from rag.embedder import get_embedder
    from rag.mcp_neo4j_server import Neo4jMCPToolHandler
    
    # Create components
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    embedder = get_embedder()
    handler = Neo4jMCPToolHandler(neo4j_store, embedder)
    
    print(f"\nConnected to Neo4j with {neo4j_store.size} triplets")
    print("\nAvailable MCP tools:")
    for tool in handler.get_tools():
        print(f"  - {tool['function']['name']}: {tool['function']['description'][:60]}...")
    
    # Test semantic search
    print("\n--- Testing search_knowledge_graph tool ---")
    result = handler.handle_tool_call(
        "search_knowledge_graph",
        {"query": "Tell me about the entities", "top_k": 5}
    )
    import json
    data = json.loads(result)
    print(f"Query: 'Tell me about the entities'")
    print(f"Found {data.get('num_results', 0)} results:")
    for fact in data.get('facts', [])[:3]:
        print(f"  [{fact.get('score', 0):.3f}] {fact.get('fact', '')}")
    
    neo4j_store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demo MCP Agent with Neo4j Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (default: from env or 'password123')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including tool call details"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use lite mode (API backend instead of local GPU)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple search demo without LLM"
    )
    
    args = parser.parse_args()
    
    # Get password
    neo4j_password = args.neo4j_password
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    print("=" * 60)
    print("MCP Agent Demo - Qwen2.5 + Neo4j Knowledge Graph")
    print("=" * 60)
    
    # Test connection first
    if not test_neo4j_connection(args.neo4j_uri, args.neo4j_user, neo4j_password):
        print("\nCannot proceed without Neo4j connection.")
        print("Please ensure Neo4j is running and try again.")
        sys.exit(1)
    
    if args.simple:
        # Run simple demo without LLM
        run_simple_search_demo(
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
        )
    else:
        # Run full MCP agent demo
        run_mcp_agent_demo(
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
            verbose=args.verbose,
            lite_mode=args.lite,
        )


if __name__ == "__main__":
    main()
