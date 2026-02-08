#!/usr/bin/env python3
"""
Interactive MCP Agent - Chat with your Knowledge Graph.

A REPL-style interface for conversing with the Neo4j knowledge graph
using the MCP Agent. Ask questions, explore entities, and see tool calls
in real-time.

Supports dual-LLM mode for validated knowledge expansion:
- Local LLM: Proposes new triplets when knowledge gaps are detected
- Remote LLM: Validates proposed triplets for factual accuracy

Prerequisites:
- Neo4j running at bolt://localhost:7687
- Knowledge graph data loaded (run migrate_faiss_to_neo4j.py first)
- For full mode: GPU with ~6GB VRAM for Qwen2.5-7B with 4-bit quantization
- For lite mode: API backend configured (export_google_ai.sh or similar)
- For validation mode: Remote LLM configured (export_dual_llm.sh)

Usage:
    # Source the local LLM config first
    source export_local_llm.sh
    
    # Start interactive session (full local LLM)
    python scripts/interactive_agent.py
    
    # Use lite mode with API backend (lower memory usage)
    python scripts/interactive_agent.py --lite
    
    # Enable dual-LLM validation mode (local proposes, remote validates)
    source export_dual_llm.sh  # Configure remote LLM first
    python scripts/interactive_agent.py --validate
    
    # Custom Neo4j connection
    python scripts/interactive_agent.py --neo4j-password mypassword
    
    # Start with verbose mode enabled
    python scripts/interactive_agent.py --verbose

Interactive Commands:
    /help              - Show available commands
    /quit or /exit     - Exit the session
    /tools             - List available MCP tools
    /stats             - Show knowledge graph statistics  
    /verbose           - Toggle verbose mode (show tool calls)
    /validate          - Toggle dual-LLM validation mode
    /expand            - Toggle triplet expansion
    /history           - Show query history
    /clear             - Clear screen
    /search <query>    - Direct semantic search (no agent reasoning)
    /entity <name>     - Query all facts about an entity
    /cypher <query>    - Execute raw Cypher query
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.WARNING,  # Quieter by default for interactive use
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ANSI color codes
class Colors:
    CYAN = '\033[1;36m'
    YELLOW = '\033[1;33m'
    GREEN = '\033[1;32m'
    RED = '\033[1;31m'
    MAGENTA = '\033[1;35m'
    BLUE = '\033[1;34m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    # Background colors
    BG_YELLOW = '\033[43m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'


def print_colored(text: str, color: str = Colors.RESET):
    """Print text with ANSI color codes."""
    print(f"{color}{text}{Colors.RESET}")


def print_section(title: str, color: str = Colors.YELLOW):
    """Print a section header."""
    print(f"\n{color}{'─' * 50}{Colors.RESET}")
    print(f"{color}{title}{Colors.RESET}")
    print(f"{color}{'─' * 50}{Colors.RESET}")


def display_validation_details(result, verbose: bool = True):
    """
    Display detailed information about the dual-LLM validation process.
    
    Shows the interaction between local LLM (proposer) and remote LLM (validator),
    including validation results with the remote model name, and persistence
    justification decisions.
    """
    # Check if this is a validated result with validation info
    if not hasattr(result, 'knowledge_gap_detected'):
        return
    
    if not result.knowledge_gap_detected and not result.proposed_triplets:
        return
    
    # Get the remote model name
    remote_model = getattr(result, 'remote_model_name', '') or 'unknown'
    
    print_section("Dual-LLM Validation Details", Colors.BLUE)
    
    # LLM Assessment from local model
    if result.llm_assessment:
        print(f"\n  {Colors.CYAN}[Local LLM Assessment]{Colors.RESET}")
        assessment = result.llm_assessment
        status = assessment.get('assessment', 'N/A')
        confidence = assessment.get('confidence', 0)
        
        status_color = Colors.GREEN if status == 'SUFFICIENT' else Colors.YELLOW
        print(f"    Assessment: {status_color}{status}{Colors.RESET}")
        print(f"    Confidence: {confidence:.0%}")
        
        if assessment.get('missing_information'):
            print(f"    Missing information ({len(assessment['missing_information'])} items):")
            for item in assessment['missing_information'][:5]:
                print(f"      {Colors.GRAY}• {item}{Colors.RESET}")
            if len(assessment['missing_information']) > 5:
                print(f"      {Colors.GRAY}... and {len(assessment['missing_information']) - 5} more{Colors.RESET}")
    
    # Knowledge gap detection
    print(f"\n  {Colors.CYAN}[Knowledge Gap Detection]{Colors.RESET}")
    gap_status = f"{Colors.YELLOW}YES{Colors.RESET}" if result.knowledge_gap_detected else f"{Colors.GREEN}NO{Colors.RESET}"
    print(f"    Gap detected: {gap_status}")
    print(f"    Knowledge iterations: {result.knowledge_iterations}")
    
    # Proposed triplets from local LLM
    if result.proposed_triplets:
        print(f"\n  {Colors.CYAN}[Local LLM → Proposed Triplets]{Colors.RESET}")
        print(f"    Count: {len(result.proposed_triplets)}")
        for i, triplet in enumerate(result.proposed_triplets[:8], 1):
            print(f"    {Colors.GRAY}{i}.{Colors.RESET} {triplet}")
        if len(result.proposed_triplets) > 8:
            print(f"    {Colors.GRAY}... and {len(result.proposed_triplets) - 8} more{Colors.RESET}")
    
    # Validation process details (with remote model name)
    if result.validation_attempts > 0:
        print(f"\n  {Colors.CYAN}[Remote LLM Validation ({Colors.MAGENTA}{remote_model}{Colors.CYAN})]{Colors.RESET}")
        print(f"    Validation attempts: {result.validation_attempts}")
        print(f"    Validation failed: {Colors.RED if result.validation_failed else Colors.GREEN}{'YES' if result.validation_failed else 'NO'}{Colors.RESET}")
    
    # Validation history (shows back-and-forth)
    if result.validation_history and verbose:
        print(f"\n  {Colors.CYAN}[Validation History]{Colors.RESET}")
        for vh in result.validation_history:
            attempt = vh.get('attempt', '?')
            validated = vh.get('validated', [])
            rejected = vh.get('rejected', [])
            all_rejected = vh.get('all_rejected', False)
            vh_model = vh.get('remote_model', remote_model)
            
            ki = vh.get('knowledge_iteration', '')
            ki_label = f"Iter {ki}, " if ki else ""
            status_icon = "x" if all_rejected else "✓"
            print(f"\n    {Colors.BOLD}{ki_label}Attempt {attempt} ({vh_model}):{Colors.RESET} {status_icon}")
            
            if validated:
                print(f"      {Colors.GREEN}Validated ({len(validated)}):{Colors.RESET}")
                for v in validated[:5]:
                    triplet = v.get('triplet', v) if isinstance(v, dict) else v
                    reason = v.get('reason', '') if isinstance(v, dict) else ''
                    status = v.get('status', 'validated') if isinstance(v, dict) else 'validated'
                    status_note = " (corrected)" if status == "corrected" else ""
                    print(f"        ✓ {triplet}{status_note}")
                    if reason and verbose:
                        reason_display = f"{reason[:60]}..." if len(reason) > 60 else reason
                        print(f"          {Colors.GRAY}Reason: {reason_display}{Colors.RESET}")
            
            if rejected:
                print(f"      {Colors.RED}Rejected ({len(rejected)}):{Colors.RESET}")
                for r in rejected[:5]:
                    triplet = r.get('triplet', r) if isinstance(r, dict) else r
                    reason = r.get('reason', '') if isinstance(r, dict) else ''
                    print(f"        x {triplet}")
                    if reason and verbose:
                        reason_display = f"{reason[:60]}..." if len(reason) > 60 else reason
                        print(f"          {Colors.GRAY}Reason: {reason_display}{Colors.RESET}")


def display_persistence_details(result, verbose: bool = True):
    """
    Display persistence justification and results after the answer.
    
    Shows which validated triplets the local LLM decided to persist vs skip,
    with reasons for each decision.
    """
    if not hasattr(result, 'persistence_justification'):
        return
    
    justification = result.persistence_justification
    if not justification:
        return
    
    persist_list = justification.get("persist", [])
    skip_list = justification.get("skip", [])
    skipped_triplets = getattr(result, 'skipped_triplets', [])
    
    # Only show if there was something to decide about
    if not persist_list and not skip_list and not skipped_triplets:
        return
    
    print_section("Persistence Justification (Local LLM)", Colors.CYAN)
    
    if persist_list:
        print(f"\n  {Colors.GREEN}Approved for persistence ({len(persist_list)}):{Colors.RESET}")
        for item in persist_list:
            triplet = item.get('triplet', '')
            reason = item.get('reason', '')
            print(f"    + {triplet}")
            if reason and verbose:
                print(f"      {Colors.GRAY}{reason}{Colors.RESET}")
    
    if skip_list or skipped_triplets:
        skip_display = skip_list or skipped_triplets
        print(f"\n  {Colors.RED}Skipped (not persisted) ({len(skip_display)}):{Colors.RESET}")
        for item in skip_display:
            triplet = item.get('triplet', '')
            reason = item.get('reason', '')
            print(f"    - {triplet}")
            if reason and verbose:
                print(f"      {Colors.GRAY}{reason}{Colors.RESET}")
    
    # Final persistence count
    persisted_count = getattr(result, 'persisted_count', 0)
    if persisted_count > 0:
        print(f"\n  {Colors.GREEN}Persisted to Neo4j: {persisted_count} triplet(s){Colors.RESET}")
    elif persist_list:
        print(f"\n  {Colors.YELLOW}Persistence pending...{Colors.RESET}")
    else:
        print(f"\n  {Colors.GRAY}No triplets persisted (all skipped by local LLM){Colors.RESET}")


def test_neo4j_connection(uri: str, user: str, password: str) -> bool:
    """Test Neo4j connection before starting interactive session."""
    try:
        from rag.neo4j_store import Neo4jStore
        
        store = Neo4jStore(
            uri=uri,
            user=user,
            password=password,
            create_indexes=False,
        )
        size = store.size
        store.close()
        return True
        
    except Exception as e:
        print_colored(f"ERROR: Failed to connect to Neo4j: {e}", Colors.RED)
        print("Make sure Neo4j is running and accessible.")
        return False


def run_interactive_session(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    lite_mode: bool = False,
    enable_expansion: bool = True,
    start_verbose: bool = False,
    enable_validation: bool = False,
):
    """
    Run an interactive session with the MCP Agent.
    
    Args:
        neo4j_uri: Neo4j Bolt URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        lite_mode: Use API backend instead of local GPU
        enable_expansion: Enable triplet expansion by default
        start_verbose: Start with verbose mode enabled
        enable_validation: Enable dual-LLM validation mode
    """
    try:
        import readline  # Enable arrow key history in input
    except ImportError:
        pass  # readline not available on all platforms
    
    print("\n" + "=" * 60)
    print_colored("  Interactive MCP Agent Session", Colors.BOLD)
    print("=" * 60)
    
    # Get model configuration from environment
    llm_model = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    embedder_model = os.environ.get("LOCAL_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    quantize = os.environ.get("LOCAL_LLM_QUANTIZE", "4bit")
    
    # Check for remote LLM configuration
    from rag.edc.edc.utils.llm_utils import is_remote_llm_configured, get_remote_model_name
    remote_configured = is_remote_llm_configured()
    remote_model = get_remote_model_name() if remote_configured else "Not configured"
    
    print(f"\n  {Colors.CYAN}Local LLM:{Colors.RESET} {llm_model}")
    print(f"  {Colors.CYAN}Embedder:{Colors.RESET} {embedder_model}")
    print(f"  {Colors.CYAN}Mode:{Colors.RESET} {'Lite (API)' if lite_mode else 'Full (Local GPU)'}")
    print(f"  {Colors.CYAN}Remote LLM:{Colors.RESET} {remote_model}")
    
    if enable_validation and not remote_configured:
        print_colored("\n  Warning: Validation mode requested but remote LLM not configured.", Colors.YELLOW)
        print("  Run 'source export_dual_llm.sh' to configure.")
        enable_validation = False
    
    print("\n  Loading agent... (this may take a moment)")
    
    # Create components
    from rag.neo4j_store import Neo4jStore
    from rag.embedder import get_embedder
    from rag.mcp_neo4j_server import Neo4jMCPToolHandler
    
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    embedder = get_embedder(model_name=embedder_model)
    handler = Neo4jMCPToolHandler(
        neo4j_store, 
        embedder, 
        enable_expansion=enable_expansion,
        allow_cypher=True,
    )
    
    # Create agent based on mode
    validation_mode = enable_validation and remote_configured
    agent = None
    validated_agent = None
    
    if lite_mode:
        from rag.mcp_agent import MCPAgentLite
        agent = MCPAgentLite(
            neo4j_store=neo4j_store,
            embedder=embedder,
            max_iterations=5,
        )
    else:
        from rag.mcp_agent import create_mcp_agent
        agent = create_mcp_agent(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            llm_model=llm_model,
            quantize=quantize,
            max_iterations=5,
        )
    
    # Create validated agent if remote LLM is configured
    if remote_configured:
        from rag.mcp_agent import create_mcp_agent_with_validation
        validated_agent = create_mcp_agent_with_validation(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            enable_validation=True,
            auto_expand=True,
            max_validation_retries=3,
            max_knowledge_iterations=3,
        )
    
    print_colored("  Agent ready!", Colors.GREEN)
    print(f"  Knowledge Graph: {neo4j_store.size} triplets")
    
    if validation_mode:
        print_colored(f"  Validation Mode: ENABLED (Local: {llm_model.split('/')[-1]} ↔ Remote: {remote_model})", Colors.GREEN)
    
    # Session state
    verbose_mode = start_verbose
    expand_mode = enable_expansion
    query_history = []
    
    # Help text
    help_text = f"""
{Colors.BOLD}Available Commands:{Colors.RESET}
  /help              - Show this help message
  /quit or /exit     - Exit the interactive session
  /tools             - List available MCP tools
  /stats             - Show knowledge graph statistics  
  /verbose           - Toggle verbose mode (show tool calls + validation details)
  /validate          - Toggle dual-LLM validation mode (local↔remote)
  /expand            - Toggle triplet expansion
  /history           - Show query history
  /clear             - Clear screen
  /search <query>    - Direct semantic search (no agent reasoning)
  /entity <name>     - Query all facts about an entity
  /cypher <query>    - Execute raw Cypher query

{Colors.BOLD}Dual-LLM Validation:{Colors.RESET}
  When validation mode is ON, the agent uses two LLMs:
  • {Colors.CYAN}Local LLM{Colors.RESET}  - Detects knowledge gaps, proposes new triplets
  • {Colors.MAGENTA}Remote LLM{Colors.RESET} - Validates proposed triplets for factual accuracy
  
  Use /verbose to see detailed interaction between the LLMs.
  
{Colors.BOLD}Tips:{Colors.RESET}
  - Use arrow keys to navigate command history
  - Press Ctrl+C to cancel current operation
  - Press Ctrl+D or type /quit to exit
"""
    
    print(help_text)
    print("-" * 60)
    
    # Show initial status
    status_parts = []
    if verbose_mode:
        status_parts.append(f"Verbose: {Colors.GREEN}ON{Colors.RESET}")
    if validation_mode:
        status_parts.append(f"Validation: {Colors.GREEN}ON{Colors.RESET}")
    if status_parts:
        print("  " + " | ".join(status_parts))
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Colors.CYAN}You:{Colors.RESET} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                
                if cmd in ("/quit", "/exit", "/q"):
                    print("\nGoodbye! Closing connection...")
                    break
                
                elif cmd == "/help":
                    print(help_text)
                
                elif cmd == "/tools":
                    print_colored("\nAvailable MCP Tools:", Colors.YELLOW)
                    for tool in handler.get_tools():
                        name = tool['function']['name']
                        desc = tool['function']['description'][:70]
                        print(f"  {Colors.BOLD}{name}{Colors.RESET}")
                        print(f"    {desc}...")
                
                elif cmd == "/stats":
                    print_colored("\nKnowledge Graph Statistics:", Colors.YELLOW)
                    print(f"  Total triplets: {neo4j_store.size}")
                    
                    # Get predicate distribution
                    with neo4j_store.driver.session() as session:
                        result = session.run("""
                            MATCH (t:Triplet)
                            RETURN t.predicate AS predicate, count(*) AS count
                            ORDER BY count DESC
                            LIMIT 10
                        """)
                        print("\n  Top predicates:")
                        for record in result:
                            print(f"    {record['predicate']}: {record['count']}")
                        
                        result = session.run("""
                            MATCH (t:Triplet)
                            RETURN count(DISTINCT t.subject) AS subjects,
                                   count(DISTINCT t.object) AS objects
                        """)
                        record = result.single()
                        if record:
                            print(f"\n  Unique subjects: {record['subjects']}")
                            print(f"  Unique objects: {record['objects']}")
                
                elif cmd == "/verbose":
                    verbose_mode = not verbose_mode
                    status = f"{Colors.GREEN}ON{Colors.RESET}" if verbose_mode else f"{Colors.RED}OFF{Colors.RESET}"
                    print(f"\n  Verbose mode: {status}")
                
                elif cmd == "/expand":
                    expand_mode = not expand_mode
                    handler.enable_expansion = expand_mode
                    status = f"{Colors.GREEN}ON{Colors.RESET}" if expand_mode else f"{Colors.RED}OFF{Colors.RESET}"
                    print(f"\n  Triplet expansion: {status}")
                
                elif cmd == "/validate":
                    if not remote_configured:
                        print_colored("\n  Remote LLM not configured!", Colors.RED)
                        print("  Run 'source export_dual_llm.sh' to configure dual-LLM mode.")
                    else:
                        validation_mode = not validation_mode
                        status = f"{Colors.GREEN}ON{Colors.RESET}" if validation_mode else f"{Colors.RED}OFF{Colors.RESET}"
                        print(f"\n  Dual-LLM Validation: {status}")
                        if validation_mode:
                            print(f"    {Colors.CYAN}Local:{Colors.RESET}  {llm_model.split('/')[-1]} (proposes triplets)")
                            print(f"    {Colors.MAGENTA}Remote:{Colors.RESET} {remote_model} (validates triplets)")
                            print(f"    Use /verbose to see detailed LLM interactions.")
                
                elif cmd == "/history":
                    if not query_history:
                        print("\n  No query history yet.")
                    else:
                        print_colored("\nQuery History:", Colors.YELLOW)
                        for i, (q, a) in enumerate(query_history[-10:], 1):
                            q_preview = q[:50] + "..." if len(q) > 50 else q
                            a_preview = a[:50] + "..." if len(a) > 50 else a
                            print(f"  {i}. Q: {q_preview}")
                            print(f"     A: {a_preview}")
                
                elif cmd == "/clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print("=" * 60)
                    print_colored("  Interactive MCP Agent Session", Colors.BOLD)
                    print("=" * 60)
                
                elif cmd == "/search":
                    if not cmd_arg:
                        print("  Usage: /search <query>")
                    else:
                        print_colored(f"\nSearching: {cmd_arg}", Colors.YELLOW)
                        result = handler.handle_tool_call(
                            "search_knowledge_graph",
                            {"query": cmd_arg, "top_k": 10}
                        )
                        data = json.loads(result)
                        print(f"\n  Found {data.get('num_results', 0)} results:")
                        for fact in data.get('facts', []):
                            score = fact.get('score', 0)
                            f = fact.get('fact', '')
                            print(f"    [{score:.3f}] {f}")
                
                elif cmd == "/entity":
                    if not cmd_arg:
                        print("  Usage: /entity <entity_name>")
                    else:
                        print_colored(f"\nQuerying entity: {cmd_arg}", Colors.YELLOW)
                        result = handler.handle_tool_call(
                            "query_entity",
                            {"entity": cmd_arg}
                        )
                        data = json.loads(result)
                        print(f"\n  Found {data.get('num_facts', 0)} facts:")
                        for fact in data.get('facts', []):
                            print(f"    {fact.get('fact', '')}")
                
                elif cmd == "/cypher":
                    if not cmd_arg:
                        print("  Usage: /cypher <cypher_query>")
                    else:
                        print_colored("\nExecuting Cypher:", Colors.YELLOW)
                        result = handler.handle_tool_call(
                            "run_cypher",
                            {"cypher": cmd_arg}
                        )
                        data = json.loads(result)
                        if "error" in data:
                            print_colored(f"  Error: {data['error']}", Colors.RED)
                        else:
                            print(f"\n  Results ({data.get('num_results', 0)}):")
                            for r in data.get('results', [])[:10]:
                                print(f"    {r}")
                
                else:
                    print(f"  Unknown command: {cmd}")
                    print("  Type /help for available commands")
                
                continue
            
            # Regular query - run through agent
            query_history.append((user_input, ""))
            
            # Choose agent based on validation mode
            if validation_mode and validated_agent:
                print_colored("\nThinking... (dual-LLM validation enabled)", Colors.YELLOW)
                print(f"  {Colors.GRAY}Local LLM will propose triplets if knowledge gaps detected{Colors.RESET}")
                print(f"  {Colors.GRAY}Remote LLM will validate proposed triplets{Colors.RESET}")
            else:
                print_colored("\nThinking...", Colors.YELLOW)
            
            try:
                # Run the appropriate agent
                if validation_mode and validated_agent:
                    result = validated_agent.run(user_input, verbose=verbose_mode, force_expand=True)
                else:
                    result = agent.run(user_input, verbose=verbose_mode)
                
                # Display tool calls if verbose
                if verbose_mode and result.tool_calls:
                    print_section("Tool Calls", Colors.MAGENTA)
                    for tc in result.tool_calls:
                        print(f"  [{tc['iteration']}] {Colors.BOLD}{tc['tool']}{Colors.RESET}")
                        args_str = json.dumps(tc['arguments'], indent=4)
                        for line in args_str.split('\n'):
                            print(f"      {line}")
                        if 'result' in tc:
                            res = tc['result']
                            if isinstance(res, dict):
                                if 'num_facts' in res:
                                    print(f"      {Colors.GREEN}→ Found {res['num_facts']} facts{Colors.RESET}")
                                elif 'num_results' in res:
                                    print(f"      {Colors.GREEN}→ Found {res['num_results']} results{Colors.RESET}")
                
                # Display validation details (before answer)
                if validation_mode and hasattr(result, 'knowledge_gap_detected'):
                    display_validation_details(result, verbose=verbose_mode)
                
                # Display answer
                print(f"\n{Colors.GREEN}{'─' * 50}{Colors.RESET}")
                print(f"{Colors.GREEN}Agent:{Colors.RESET} {result.answer}")
                
                # Display persistence justification (after answer)
                if validation_mode and hasattr(result, 'persistence_justification'):
                    display_persistence_details(result, verbose=verbose_mode)
                
                # Update history with answer
                query_history[-1] = (user_input, result.answer)
                
                # Show stats
                stats_parts = [f"iterations: {result.iterations}", f"tools: {len(result.tool_calls)}"]
                if hasattr(result, 'knowledge_iterations'):
                    stats_parts.append(f"knowledge_cycles: {result.knowledge_iterations}")
                if hasattr(result, 'persisted_count') and result.persisted_count > 0:
                    stats_parts.append(f"persisted: {result.persisted_count}")
                skipped_count = len(getattr(result, 'skipped_triplets', []))
                if skipped_count > 0:
                    stats_parts.append(f"skipped: {skipped_count}")
                
                print(f"\n{Colors.GRAY}  ({', '.join(stats_parts)}){Colors.RESET}")
                
                if result.warning:
                    print_colored(f"  Warning: {result.warning}", Colors.YELLOW)
                    
            except Exception as e:
                print_colored(f"\nError: {e}", Colors.RED)
                if verbose_mode:
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\n  (Interrupted)")
            continue
        
        except EOFError:
            print("\n\nGoodbye!")
            break
    
    # Cleanup
    neo4j_store.close()
    print("Connection closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MCP Agent - Chat with your Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/interactive_agent.py                    # Full local LLM
  python scripts/interactive_agent.py --lite             # API backend
  python scripts/interactive_agent.py --verbose          # Start with verbose mode
  python scripts/interactive_agent.py --validate         # Enable dual-LLM validation
  python scripts/interactive_agent.py --validate -v      # Validation + verbose output
  python scripts/interactive_agent.py --neo4j-password pass123

Dual-LLM Validation Mode:
  To use validation mode, first configure the remote LLM:
    source export_dual_llm.sh
  
  Then run with --validate flag to see the local LLM propose triplets
  and the remote LLM validate them for factual accuracy.
        """
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
        help="Neo4j password (default: from NEO4J_PASSWORD env or 'password123')"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use lite mode (API backend instead of local GPU)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Start with verbose mode enabled"
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable triplet expansion by default"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable dual-LLM validation mode (requires remote LLM config)"
    )
    
    args = parser.parse_args()
    
    # Get password
    neo4j_password = args.neo4j_password
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    # Test connection first
    print("Connecting to Neo4j...")
    if not test_neo4j_connection(args.neo4j_uri, args.neo4j_user, neo4j_password):
        print("\nCannot proceed without Neo4j connection.")
        print("Please ensure Neo4j is running and try again.")
        sys.exit(1)
    
    # Run interactive session
    run_interactive_session(
        args.neo4j_uri,
        args.neo4j_user,
        neo4j_password,
        lite_mode=args.lite,
        enable_expansion=not args.no_expansion,
        start_verbose=args.verbose,
        enable_validation=args.validate,
    )


if __name__ == "__main__":
    main()
