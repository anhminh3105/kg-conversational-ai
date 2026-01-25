"""
MCP Server exposing Neo4j Knowledge Graph tools.

This module provides tool definitions in OpenAI function calling format
and a handler class that bridges LLM tool calls to actual Neo4j operations.

Tools:
- search_knowledge_graph: Semantic search for triplets using embeddings
- query_entity: Find all facts about a specific entity via graph traversal
- expand_entity: Explore connected entities within N hops
- run_cypher: Execute Cypher queries directly (for advanced users)

Usage:
    from rag.mcp_neo4j_server import Neo4jMCPToolHandler, NEO4J_TOOLS
    
    # Initialize handler with Neo4j store and embedder
    handler = Neo4jMCPToolHandler(neo4j_store, embedder)
    
    # Get tool definitions for LLM
    tools = handler.get_tools()
    
    # Execute a tool call from the LLM
    result = handler.handle_tool_call("search_knowledge_graph", {"query": "Where is Einstein from?"})
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Tool definitions in OpenAI/MCP function calling format
NEO4J_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": "Search the knowledge graph for facts related to a query using semantic similarity. Use this when you need to find information about a topic but don't know the exact entities involved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what information you're looking for"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_entity",
            "description": "Find all facts (triplets) about a specific entity. Use this when you know the exact entity name and want to find everything known about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The exact entity name to query (e.g., 'Albert Einstein', 'Paris', 'quantum mechanics')"
                    },
                    "relationship": {
                        "type": "string",
                        "description": "Optional: filter results to only show facts with this relationship type (e.g., 'born_in', 'discovered', 'located_in')"
                    }
                },
                "required": ["entity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "expand_entity",
            "description": "Explore the knowledge graph starting from an entity, finding connected entities and their relationships within N hops. Use this to discover related information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The starting entity name"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Number of hops to explore (default: 2, max: 3)",
                        "default": 2
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of facts to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["entity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_cypher",
            "description": "Execute a Cypher query against the Neo4j knowledge graph. Only use this for complex queries that cannot be handled by other tools. The graph has Triplet nodes with properties: subject, predicate, object, document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "The Cypher query to execute. Example: MATCH (t:Triplet) WHERE t.subject = 'Einstein' RETURN t.subject, t.predicate, t.object LIMIT 10"
                    }
                },
                "required": ["cypher"]
            }
        }
    }
]


# Simplified tool list for smaller models (without Cypher)
NEO4J_TOOLS_SIMPLE = [
    NEO4J_TOOLS[0],  # search_knowledge_graph
    NEO4J_TOOLS[1],  # query_entity
    NEO4J_TOOLS[2],  # expand_entity
]


class Neo4jMCPToolHandler:
    """
    Handles MCP tool calls for Neo4j knowledge graph operations.
    
    This bridges between the LLM's tool calls and the actual Neo4j store,
    formatting results appropriately for the LLM to consume.
    
    Attributes:
        store: Neo4jStore instance for database operations
        embedder: Embedder instance for semantic search
        allow_cypher: Whether to allow raw Cypher queries (security consideration)
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        allow_cypher: bool = True,
    ):
        """
        Initialize the tool handler.
        
        Args:
            neo4j_store: Neo4jStore instance
            embedder: Embedder instance for generating query embeddings
            allow_cypher: Whether to allow raw Cypher queries
        """
        self.store = neo4j_store
        self.embedder = embedder
        self.allow_cypher = allow_cypher
        
        logger.info(f"Initialized Neo4jMCPToolHandler (cypher={allow_cypher})")
    
    def get_tools(self, include_cypher: bool = None) -> List[Dict]:
        """
        Return tool definitions for the LLM.
        
        Args:
            include_cypher: Override for whether to include Cypher tool
            
        Returns:
            List of tool definitions in OpenAI function calling format
        """
        if include_cypher is None:
            include_cypher = self.allow_cypher
        
        if include_cypher:
            return NEO4J_TOOLS
        else:
            return NEO4J_TOOLS_SIMPLE
    
    def get_tool_names(self) -> List[str]:
        """Return list of available tool names."""
        return [t["function"]["name"] for t in self.get_tools()]
    
    def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """
        Execute a tool call and return the result as a JSON string.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments from the LLM
            
        Returns:
            JSON string containing the tool result
        """
        logger.info(f"Handling tool call: {tool_name}({arguments})")
        
        try:
            if tool_name == "search_knowledge_graph":
                return self._search_kg(
                    arguments["query"],
                    arguments.get("top_k", 5)
                )
            
            elif tool_name == "query_entity":
                return self._query_entity(
                    arguments["entity"],
                    arguments.get("relationship")
                )
            
            elif tool_name == "expand_entity":
                return self._expand_entity(
                    arguments["entity"],
                    arguments.get("depth", 2),
                    arguments.get("max_results", 20)
                )
            
            elif tool_name == "run_cypher":
                if not self.allow_cypher:
                    return json.dumps({
                        "error": "Cypher queries are disabled for security reasons"
                    })
                return self._run_cypher(arguments["cypher"])
            
            else:
                return json.dumps({
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": self.get_tool_names()
                })
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return json.dumps({
                "error": str(e),
                "tool": tool_name,
            })
    
    def _search_kg(self, query: str, top_k: int) -> str:
        """
        Semantic search in the knowledge graph.
        
        Args:
            query: Natural language query
            top_k: Number of results
            
        Returns:
            JSON string with search results
        """
        # Validate top_k
        top_k = min(max(1, top_k), 20)
        
        # Embed the query
        query_embedding = self.embedder.embed_query(query, normalize=True)
        
        # Search
        results = self.store.search(query_embedding, top_k=top_k)
        
        # Format results
        facts = []
        for r in results:
            m = r.metadata
            facts.append({
                "fact": f"({m['subject']}, {m['predicate']}, {m['object']})",
                "subject": m["subject"],
                "predicate": m["predicate"],
                "object": m["object"],
                "score": round(r.score, 3),
            })
        
        return json.dumps({
            "query": query,
            "num_results": len(facts),
            "facts": facts,
        }, ensure_ascii=False)
    
    def _query_entity(
        self,
        entity: str,
        relationship: Optional[str] = None,
    ) -> str:
        """
        Query all facts about an entity.
        
        Args:
            entity: Entity name
            relationship: Optional relationship filter
            
        Returns:
            JSON string with entity facts
        """
        results = self.store.graph_search(entity, relationship)
        
        facts = []
        for r in results:
            facts.append({
                "fact": f"({r['subject']}, {r['predicate']}, {r['object']})",
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
            })
        
        response = {
            "entity": entity,
            "num_facts": len(facts),
            "facts": facts,
        }
        
        if relationship:
            response["filtered_by_relationship"] = relationship
        
        return json.dumps(response, ensure_ascii=False)
    
    def _expand_entity(
        self,
        entity: str,
        depth: int,
        max_results: int,
    ) -> str:
        """
        Expand from an entity to find connected facts.
        
        Args:
            entity: Starting entity
            depth: Number of hops
            max_results: Maximum results
            
        Returns:
            JSON string with expanded graph
        """
        # Validate parameters
        depth = min(max(1, depth), 3)
        max_results = min(max(1, max_results), 100)
        
        results = self.store.graph_expand(entity, depth=depth, max_results=max_results)
        
        facts = []
        entities_found = set()
        
        for r in results:
            facts.append({
                "fact": f"({r['subject']}, {r['predicate']}, {r['object']})",
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
            })
            entities_found.add(r["subject"])
            entities_found.add(r["object"])
        
        return json.dumps({
            "starting_entity": entity,
            "depth": depth,
            "num_facts": len(facts),
            "num_entities": len(entities_found),
            "entities": list(entities_found),
            "facts": facts,
        }, ensure_ascii=False)
    
    def _run_cypher(self, cypher: str) -> str:
        """
        Execute a Cypher query.
        
        Args:
            cypher: Cypher query string
            
        Returns:
            JSON string with query results
        """
        # Basic security check - disallow destructive operations
        cypher_upper = cypher.upper()
        dangerous_keywords = ["DELETE", "DETACH", "DROP", "CREATE", "SET", "REMOVE", "MERGE"]
        
        for keyword in dangerous_keywords:
            if keyword in cypher_upper and "RETURN" not in cypher_upper:
                return json.dumps({
                    "error": f"Query contains potentially destructive operation: {keyword}",
                    "hint": "Only read queries (MATCH...RETURN) are allowed"
                })
        
        try:
            results = self.store.cypher_search(cypher)
            
            # Limit results to prevent overwhelming the LLM
            results = results[:50]
            
            return json.dumps({
                "query": cypher,
                "num_results": len(results),
                "results": results,
            }, ensure_ascii=False, default=str)
        
        except Exception as e:
            return json.dumps({
                "error": f"Cypher execution failed: {str(e)}",
                "query": cypher,
            })


def format_tools_for_prompt(tools: List[Dict] = None) -> str:
    """
    Format tool definitions as a string for inclusion in prompts.
    
    Useful for models that don't support native function calling.
    
    Args:
        tools: Tool definitions (defaults to NEO4J_TOOLS)
        
    Returns:
        Formatted string describing available tools
    """
    if tools is None:
        tools = NEO4J_TOOLS
    
    lines = ["Available tools:\n"]
    
    for tool in tools:
        func = tool["function"]
        lines.append(f"### {func['name']}")
        lines.append(f"{func['description']}\n")
        lines.append("Parameters:")
        
        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])
        
        for param_name, param_info in params.items():
            req_marker = " (required)" if param_name in required else ""
            lines.append(f"  - {param_name}{req_marker}: {param_info['description']}")
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Print tool definitions for reference
    print("=== Neo4j MCP Tool Definitions ===\n")
    print(format_tools_for_prompt())
    
    print("\n=== JSON Format ===\n")
    print(json.dumps(NEO4J_TOOLS, indent=2))
