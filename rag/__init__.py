"""
RAG (Retrieval-Augmented Generation) module for Knowledge Graph triplets.

This module provides functionality to:
1. Load and parse KG triplets from EDC pipeline output (canon_kg.txt)
2. Convert triplets to embeddable text representations
3. Generate embeddings using sentence-transformers
4. Store and retrieve from FAISS index or Neo4j database
5. Retrieve relevant triplets for a query
5.5. Expand triplets using LLM (optional)
5.6. Validate triplets with remote LLM (optional, dual-LLM mode)
6. Build augmented prompts with KG context
7. Generate answers using local LLM
8. MCP-enabled agentic queries with Neo4j (optional)
9. Dual-LLM knowledge expansion with validation (optional)

Usage:
    from rag import KGRagIndexer, KGRagGenerator
    
    # Index from EDC output (FAISS backend - default)
    indexer = KGRagIndexer()
    indexer.index_from_path("./output/tmp", mode="triplet_text")
    indexer.save("./output/rag")
    
    # Index with Neo4j backend
    indexer = KGRagIndexer(
        store_type="neo4j",
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="password123",
    )
    indexer.index_from_path("./output/tmp", mode="triplet_text")
    
    # Search only
    indexer = KGRagIndexer.load("./output/rag")
    results = indexer.search("Where is Trane located?")
    
    # Full RAG with LLM generation
    generator = KGRagGenerator(indexer)
    result = generator.generate("Where is Trane located?")
    print(result.answer)
    
    # RAG with triplet expansion (enriches sparse facts)
    result = generator.generate("Where is Trane located?", expand_triplets=True)
    print(result.answer)
    print(f"Expanded triplets: {result.expanded_triplets}")
    
    # MCP Agent with Neo4j (agentic tool-calling)
    from rag import create_mcp_agent
    agent = create_mcp_agent(neo4j_password="password123")
    result = agent.run("What did Einstein discover?")
    print(result.answer)
"""

from .triplet_loader import TripletLoader, Triplet
from .representation import TripletRepresenter, RepresentationMode, EmbeddableItem, get_representer
from .embedder import Embedder, get_embedder
from .faiss_store import FaissStore, SearchResult
from .kg_rag_indexer import KGRagIndexer
from .retriever import KGRetriever, RetrievalResult, ContextFormat
from .prompt_builder import KGPromptBuilder, get_prompt_builder
from .triplet_expander import TripletExpander, get_triplet_expander
from .duplicate_detector import DuplicateDetector, DuplicateCheckResult, get_duplicate_detector
from .generator import KGRagGenerator, GenerationResult, create_generator

# Knowledge gap detection and validation (for dual-LLM mode)
from .triplet_validator import TripletValidator, ValidationResult, ValidatedTriplet, get_triplet_validator

# Prompt loading utilities
from .prompts import load_prompt, get_prompt_path, list_prompts, clear_cache as clear_prompt_cache

# Neo4j and MCP components (lazy imports to avoid hard dependency)
def _get_neo4j_store():
    from .neo4j_store import Neo4jStore
    return Neo4jStore

def _get_mcp_agent():
    from .mcp_agent import MCPAgent, MCPAgentLite, AgentResult, create_mcp_agent
    return MCPAgent, MCPAgentLite, AgentResult, create_mcp_agent

def _get_mcp_agent_with_validation():
    from .mcp_agent import MCPAgentWithValidation, ValidatedAgentResult, create_mcp_agent_with_validation
    return MCPAgentWithValidation, ValidatedAgentResult, create_mcp_agent_with_validation

def _get_mcp_tools():
    from .mcp_neo4j_server import Neo4jMCPToolHandler, NEO4J_TOOLS, format_tools_for_prompt
    return Neo4jMCPToolHandler, NEO4J_TOOLS, format_tools_for_prompt

# Convenience function for MCP agent creation
def create_mcp_agent(*args, **kwargs):
    """Create an MCP-enabled agent for Neo4j knowledge graph queries."""
    from .mcp_agent import create_mcp_agent as _create
    return _create(*args, **kwargs)

def create_mcp_agent_with_validation(*args, **kwargs):
    """Create an MCP agent with dual-LLM validation for knowledge expansion."""
    from .mcp_agent import create_mcp_agent_with_validation as _create
    return _create(*args, **kwargs)

__all__ = [
    # Main interfaces
    "KGRagIndexer",
    "KGRagGenerator",
    "create_generator",
    
    # Data loading
    "TripletLoader",
    "Triplet",
    
    # Representation
    "TripletRepresenter",
    "RepresentationMode",
    "EmbeddableItem",
    "get_representer",
    
    # Embeddings
    "Embedder",
    "get_embedder",
    
    # Storage - FAISS
    "FaissStore",
    "SearchResult",
    
    # Storage - Neo4j (use _get_neo4j_store() for import)
    # Neo4jStore is available via: from rag.neo4j_store import Neo4jStore
    
    # Retrieval (Step 5)
    "KGRetriever",
    "RetrievalResult",
    "ContextFormat",
    
    # Triplet Expansion (Step 5.5)
    "TripletExpander",
    "get_triplet_expander",
    
    # Knowledge Gap Detection (Step 5.6)
    "KnowledgeGapDetector",
    "GapAnalysis",
    "get_knowledge_gap_detector",
    
    # Triplet Validation (Step 5.7 - dual-LLM mode)
    "TripletValidator",
    "ValidationResult",
    "ValidatedTriplet",
    "get_triplet_validator",
    
    # Duplicate Detection (for offline persistence)
    "DuplicateDetector",
    "DuplicateCheckResult",
    "get_duplicate_detector",
    
    # Prompt Building (Step 6)
    "KGPromptBuilder",
    "get_prompt_builder",
    
    # Generation (Step 7)
    "GenerationResult",
    
    # MCP Agent (Step 8 - agentic tool-calling)
    "create_mcp_agent",
    # MCPAgent, MCPAgentLite available via: from rag.mcp_agent import MCPAgent
    # Neo4jMCPToolHandler available via: from rag.mcp_neo4j_server import Neo4jMCPToolHandler
    
    # MCP Agent with Validation (Step 9 - dual-LLM knowledge expansion)
    "create_mcp_agent_with_validation",
    # MCPAgentWithValidation available via: from rag.mcp_agent import MCPAgentWithValidation
    
    # Prompt utilities
    "load_prompt",
    "get_prompt_path",
    "list_prompts",
    "clear_prompt_cache",
]
