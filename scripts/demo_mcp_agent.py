#!/usr/bin/env python3
"""
Demo script for MCP Agent with Neo4j Knowledge Graph.

This script demonstrates the full pipeline:
1. Connect to Neo4j and verify data
2. Create MCP Agent with Qwen2.5-7B
3. Run agentic queries that use MCP tools to query the knowledge graph
4. Show tool call traces and final answers
5. Demonstrate triplet expansion to enrich sparse knowledge graphs
6. Demonstrate dual-LLM validation workflow (local proposal + remote validation)

Prerequisites:
- Neo4j running at bolt://localhost:7687
- Knowledge graph data loaded (run migrate_faiss_to_neo4j.py first)
- GPU with ~6GB VRAM for Qwen2.5-7B with 4-bit quantization

Usage:
    # Source the local LLM config first
    source export_local_llm.sh
    
    # Run full MCP agent demo
    python scripts/demo_mcp_agent.py
    
    # Run with custom Neo4j connection
    python scripts/demo_mcp_agent.py --neo4j-password mypassword
    
    # Run with verbose output to see tool calls
    python scripts/demo_mcp_agent.py --verbose
    
    # Test without loading the full LLM (uses API backend if configured)
    python scripts/demo_mcp_agent.py --lite
    
    # Run triplet expansion demo (shows how LLM generates new facts)
    python scripts/demo_mcp_agent.py --expand
    
    # Run triplet expansion and persist to Neo4j
    python scripts/demo_mcp_agent.py --expand --persist
    
    # Run simple search demo (no LLM required)
    python scripts/demo_mcp_agent.py --simple
    
    # Run dual-LLM validated expansion demo (requires remote LLM config)
    # First configure: source export_dual_llm.sh
    python scripts/demo_mcp_agent.py --validated-expand
    
    # Run validated expansion with verbose output
    python scripts/demo_mcp_agent.py --validated-expand --verbose
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
    handler = Neo4jMCPToolHandler(neo4j_store, embedder, enable_expansion=True)
    
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


def run_triplet_expansion_demo(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    persist: bool = False,
):
    """
    Demonstrate triplet expansion with MCP tools.
    
    Shows how the LLM can generate additional knowledge graph triplets
    based on existing facts and a user query.
    """
    import json
    
    print("\n" + "=" * 60)
    print("Triplet Expansion Demo")
    print("=" * 60)
    
    from rag.neo4j_store import Neo4jStore
    from rag.embedder import get_embedder
    from rag.mcp_neo4j_server import Neo4jMCPToolHandler
    
    # Create components
    print("\n[1] Setting up components...")
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    embedder = get_embedder()
    handler = Neo4jMCPToolHandler(
        neo4j_store, 
        embedder, 
        enable_expansion=True,
        allow_cypher=True,
    )
    
    print(f"  Neo4j connected with {neo4j_store.size} triplets")
    print(f"  Triplet expansion: {'enabled' if handler.enable_expansion else 'disabled'}")
    
    # Step 1: Search for existing facts
    print("\n[2] Searching knowledge graph for relevant facts...")
    search_query = "Tell me about the main topic"
    search_result = handler.handle_tool_call(
        "search_knowledge_graph",
        {"query": search_query, "top_k": 5}
    )
    search_data = json.loads(search_result)
    
    existing_facts = []
    print(f"\n  Found {search_data.get('num_results', 0)} existing facts:")
    for fact in search_data.get('facts', []):
        print(f"    [{fact.get('score', 0):.3f}] {fact.get('fact', '')}")
        existing_facts.append(fact.get('fact', ''))
    
    if not existing_facts:
        print("\n  No existing facts found. Using sample facts for demo...")
        existing_facts = [
            "(Einstein, born_in, Germany)",
            "(Einstein, field, Physics)",
            "(Einstein, known_for, Relativity)",
        ]
        print("  Sample facts:")
        for f in existing_facts:
            print(f"    {f}")
    
    # Step 2: Expand triplets
    print("\n[3] Expanding triplets using LLM...")
    expansion_query = "What else do we know about this topic and related facts?"
    
    expand_result = handler.handle_tool_call(
        "expand_triplets",
        {
            "query": expansion_query,
            "existing_facts": existing_facts,
            "max_new_triplets": 5,
            "persist_to_graph": persist,
        }
    )
    expand_data = json.loads(expand_result)
    
    if "error" in expand_data:
        print(f"\n  Error: {expand_data['error']}")
    else:
        print(f"\n  Existing facts used: {expand_data.get('existing_facts_count', 0)}")
        print(f"  New facts generated: {expand_data.get('expanded_facts_count', 0)}")
        
        if expand_data.get('expanded_facts'):
            print("\n  Generated triplets:")
            for fact in expand_data['expanded_facts']:
                print(f"    + {fact.get('fact', '')}")
                print(f"      (source: {fact.get('source', 'unknown')})")
        
        if persist and expand_data.get('persisted_count', 0) > 0:
            print(f"\n  Persisted {expand_data['persisted_count']} triplets to Neo4j!")
            print(f"  New total: {neo4j_store.size} triplets")
    
    # Step 3: Show how this helps answering questions
    print("\n[4] How expansion helps answer questions:")
    print("  " + "-" * 50)
    print("  Without expansion: Only the original retrieved facts are available")
    print("  With expansion: LLM generates additional relevant facts that can")
    print("                  fill gaps in the knowledge graph and improve answers")
    print("  " + "-" * 50)
    
    neo4j_store.close()
    
    print("\n" + "=" * 60)
    print("Triplet Expansion Demo Complete!")
    print("=" * 60)


def run_validated_expansion_demo(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    verbose: bool = False,
    query: str = None,
):
    """
    Demonstrate the dual-LLM validated expansion workflow with detailed intermediate messages.
    
    This demo shows the complete workflow:
    1. Query the knowledge graph for relevant facts
    2. LLM assesses if facts are sufficient (with complete triplet proposals)
    3. If insufficient, shows detailed gap detection message with proposed triplets
    4. Validates with remote LLM (with retry mechanism up to 3 attempts)
    5. Shows validation attempt results for each attempt
    6. Persists validated triplets and shows confirmation
    7. Handles graceful failure if all attempts fail
    
    Prerequisites:
        - Configure dual-LLM mode: source export_dual_llm.sh
        - Or configure REMOTE_LLM_* environment variables
    """
    import json
    
    print("\n" + "=" * 60)
    print("Dual-LLM Validated Expansion Demo (with LLM-based Assessment)")
    print("=" * 60)
    
    # Check if remote LLM is configured
    from rag.edc.edc.utils.llm_utils import is_remote_llm_configured, get_remote_model_name
    
    if not is_remote_llm_configured():
        print("\n  ERROR: Remote LLM not configured!")
        print("\n  To use validated expansion, configure a remote LLM:")
        print("    1. Run: source export_dual_llm.sh")
        print("    2. Edit the script to add your API key")
        print("    3. Or set REMOTE_LLM_API_KEY and REMOTE_LLM_MODEL environment variables")
        print("\n  Alternatively, you can use:")
        print("    - source export_google_ai.sh  (uses Google AI Studio)")
        print("    - source export_sambanova.sh  (uses SambaNova)")
        return
    
    print(f"\n[1] Configuration")
    print("-" * 40)
    local_model = os.environ.get("LOCAL_LLM_MODEL", "API-based")
    remote_model = get_remote_model_name()
    max_validation_retries = 3
    max_knowledge_iterations = 3
    print(f"  Local LLM (assessment + proposals): {local_model}")
    print(f"  Remote LLM (validation):            {remote_model}")
    print(f"  Max validation retries:             {max_validation_retries}")
    print(f"  Max knowledge iterations:           {max_knowledge_iterations}")
    
    # Create the agent with validation
    print(f"\n[2] Creating MCPAgentWithValidation")
    print("-" * 40)
    
    from rag.mcp_agent import create_mcp_agent_with_validation
    
    agent = create_mcp_agent_with_validation(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        enable_validation=True,
        auto_expand=True,
        max_validation_retries=max_validation_retries,
        max_knowledge_iterations=max_knowledge_iterations,
    )
    
    print("  Agent created successfully!")
    
    # Demo queries that are likely to trigger expansion
    demo_queries = [
        query if query else "What awards and achievements is Einstein known for?",
    ]
    
    print(f"\n[3] Running Demo Queries with LLM-Based Assessment & Validated Expansion")
    print("-" * 40)
    
    for i, q in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {q}")
        print("="*60)
        
        try:
            # Run with verbose=True to see intermediate messages in real-time
            # When verbose=False, we'll display them from the result
            result = agent.run(q, verbose=verbose, force_expand=True)
            
            # If not verbose, display intermediate messages from result
            if not verbose and result.intermediate_messages:
                print("\n--- Intermediate Messages (Process Log) ---")
                for msg in result.intermediate_messages:
                    print(msg)
            
            # Display LLM Assessment details
            if result.llm_assessment:
                print(f"\n--- LLM Assessment Details ---")
                assessment = result.llm_assessment
                print(f"  Assessment: {assessment.get('assessment', 'N/A')}")
                print(f"  Confidence: {assessment.get('confidence', 0):.2f}")
                
                if assessment.get('missing_information'):
                    print(f"  Missing information ({len(assessment['missing_information'])} items):")
                    for item in assessment['missing_information']:
                        print(f"    - {item}")
                
                if assessment.get('answer'):
                    partial = assessment['answer'][:100] + "..." if len(assessment.get('answer', '')) > 100 else assessment['answer']
                    print(f"  Partial answer: {partial}")
            
            # Display validation statistics
            print(f"\n--- Validation Summary ---")
            print(f"  Knowledge iterations: {result.knowledge_iterations}")
            print(f"  Knowledge gap detected: {result.knowledge_gap_detected}")
            print(f"  Validation attempts: {result.validation_attempts}")
            print(f"  Validation failed: {result.validation_failed}")
            
            if result.proposed_triplets:
                print(f"\n--- Proposed Triplets (Local LLM) ---")
                print(f"  Count: {len(result.proposed_triplets)}")
                for t in result.proposed_triplets:
                    print(f"    + {t}")
            
            if result.validated_triplets:
                print(f"\n--- Validated Triplets (Remote LLM) ---")
                print(f"  Count: {len(result.validated_triplets)}")
                for vt in result.validated_triplets:
                    status = vt.get('status', 'unknown')
                    triplet = vt.get('triplet', '')
                    reason = vt.get('reason', '')
                    print(f"    [{status}] {triplet}")
                    if reason:
                        print(f"            Reason: {reason}")
            
            if result.rejected_triplets:
                print(f"\n--- Rejected Triplets (All Attempts) ---")
                print(f"  Count: {len(result.rejected_triplets)}")
                for rt in result.rejected_triplets:
                    triplet = rt.get('triplet', '')
                    reason = rt.get('reason', '')
                    print(f"    [rejected] {triplet}")
                    if reason:
                        print(f"               Reason: {reason}")
            
            # Display validation history if available
            if result.validation_history:
                print(f"\n--- Validation History ---")
                for vh in result.validation_history:
                    attempt = vh.get('attempt', '?')
                    all_rejected = vh.get('all_rejected', False)
                    validated = vh.get('validated', [])
                    rejected = vh.get('rejected', [])
                    status = "ALL REJECTED" if all_rejected else f"{len(validated)} accepted"
                    print(f"  Attempt {attempt}: {status}, {len(rejected)} rejected")
            
            if result.persisted_count > 0:
                print(f"\n--- Persistence ---")
                print(f"  Persisted {result.persisted_count} new triplets to Neo4j")
            
            print(f"\n--- Final Answer ---")
            print(result.answer)
            
            print(f"\n--- Tool Calls Summary ---")
            print(f"  Total iterations: {result.iterations}")
            for tc in result.tool_calls:
                tool = tc.get('tool', '')
                args_preview = str(tc.get('arguments', {}))[:50]
                print(f"    [{tc.get('iteration')}] {tool}({args_preview}...)")
            
            if result.warning:
                print(f"\n  Warning: {result.warning}")
                
        except Exception as e:
            print(f"ERROR: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Dual-LLM Validated Expansion Demo Complete!")
    print("=" * 60)
    print("\nSummary of Features:")
    print("  - LLM-based knowledge gap assessment (instead of heuristics)")
    print("  - Complete triplet coverage: LLM proposes facts for ALL missing info")
    print("  - ITERATIVE knowledge expansion:")
    print("      * Search KB -> Assess -> Propose -> Validate -> Persist")
    print("      * Re-search with new facts and re-assess")
    print("      * Repeat until knowledge is sufficient or max iterations")
    print("  - Detailed intermediate messages showing:")
    print("      * Gap detection with proposed triplets")
    print("      * Each validation attempt with remote LLM response")
    print("      * Iteration summaries with progress")
    print("      * Persistence confirmation")
    print("  - Up to 3 retry attempts per iteration with feedback loop")
    print("  - Graceful failure handling if validation cannot succeed")


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
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Run triplet expansion demo"
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist expanded triplets to Neo4j (use with --expand)"
    )
    parser.add_argument(
        "--validated-expand",
        action="store_true",
        help="Run dual-LLM validated expansion demo (requires remote LLM config)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom query for demos (optional)"
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
    
    if args.validated_expand:
        # Run dual-LLM validated expansion demo
        run_validated_expansion_demo(
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
            verbose=args.verbose,
            query=args.query,
        )
    elif args.expand:
        # Run triplet expansion demo
        run_triplet_expansion_demo(
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
            persist=args.persist,
        )
    elif args.simple:
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
