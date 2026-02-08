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
    },
    {
        "type": "function",
        "function": {
            "name": "expand_triplets",
            "description": "Generate additional knowledge graph triplets using LLM. Given a query and existing facts, uses the LLM to infer and generate new related triplets that help answer the question. Useful when retrieved facts are sparse or incomplete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question that needs more context"
                    },
                    "existing_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of existing facts in format '(subject, predicate, object)'"
                    },
                    "max_new_triplets": {
                        "type": "integer",
                        "description": "Maximum number of new triplets to generate (default: 5)",
                        "default": 5
                    },
                    "persist_to_graph": {
                        "type": "boolean",
                        "description": "Whether to save generated triplets to the knowledge graph (default: false)",
                        "default": False
                    }
                },
                "required": ["query", "existing_facts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_and_persist_triplets",
            "description": "Validate proposed triplets using a remote LLM and persist validated ones to the knowledge graph. Use this when you want to expand the knowledge base with verified facts. The remote LLM checks factual accuracy and can correct minor errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question that prompted the triplet generation"
                    },
                    "proposed_triplets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of proposed triplets in format '(subject, predicate, object)'"
                    },
                    "existing_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of existing facts from the knowledge graph for context"
                    },
                    "persist_validated": {
                        "type": "boolean",
                        "description": "Whether to persist validated triplets to the graph (default: true)",
                        "default": True
                    },
                    "skip_validation": {
                        "type": "boolean",
                        "description": "Skip remote validation and directly persist triplets (use for already-validated triplets)",
                        "default": False
                    }
                },
                "required": ["query", "proposed_triplets"]
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
        triplet_expander: Optional TripletExpander for generating new triplets
        triplet_validator: Optional TripletValidator for remote LLM validation
        enable_validation: Whether remote validation is enabled
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        allow_cypher: bool = True,
        enable_expansion: bool = True,
        enable_validation: bool = True,
        schema_path: Optional[str] = None,
        local_model_name: str = "local_llm",
    ):
        """
        Initialize the tool handler.
        
        Args:
            neo4j_store: Neo4jStore instance
            embedder: Embedder instance for generating query embeddings
            allow_cypher: Whether to allow raw Cypher queries
            enable_expansion: Whether to enable triplet expansion tool
            enable_validation: Whether to enable remote LLM validation
            schema_path: Optional path to schema CSV for triplet expansion
            local_model_name: Name of the local LLM (for metadata tracking)
        """
        self.store = neo4j_store
        self.embedder = embedder
        self.allow_cypher = allow_cypher
        self.enable_expansion = enable_expansion
        self.enable_validation = enable_validation
        self.local_model_name = local_model_name
        
        # Initialize triplet expander if enabled
        self.triplet_expander = None
        if enable_expansion:
            try:
                from .triplet_expander import get_triplet_expander
                self.triplet_expander = get_triplet_expander(schema_path=schema_path)
                logger.info("Triplet expansion enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize triplet expander: {e}")
                self.enable_expansion = False
        
        # Initialize triplet validator if enabled
        self.triplet_validator = None
        if enable_validation:
            try:
                from .triplet_validator import get_triplet_validator
                from .edc.edc.utils.llm_utils import is_remote_llm_configured
                
                if is_remote_llm_configured():
                    self.triplet_validator = get_triplet_validator(
                        local_model_name=local_model_name
                    )
                    logger.info("Triplet validation enabled (remote LLM configured)")
                else:
                    logger.warning("Remote LLM not configured - validation disabled")
                    self.enable_validation = False
            except Exception as e:
                logger.warning(f"Failed to initialize triplet validator: {e}")
                self.enable_validation = False
        
        logger.info(
            f"Initialized Neo4jMCPToolHandler "
            f"(cypher={allow_cypher}, expansion={enable_expansion}, validation={enable_validation})"
        )
    
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
            
            elif tool_name == "expand_triplets":
                if not self.enable_expansion:
                    return json.dumps({
                        "error": "Triplet expansion is not enabled"
                    })
                return self._expand_triplets(
                    arguments["query"],
                    arguments["existing_facts"],
                    arguments.get("max_new_triplets", 5),
                    arguments.get("persist_to_graph", False),
                )
            
            elif tool_name == "validate_and_persist_triplets":
                skip_validation = arguments.get("skip_validation", False)
                if not self.enable_validation and not skip_validation:
                    return json.dumps({
                        "error": "Triplet validation is not enabled. Configure remote LLM first."
                    })
                return self._validate_and_persist_triplets(
                    arguments["query"],
                    arguments["proposed_triplets"],
                    arguments.get("existing_facts", []),
                    arguments.get("persist_validated", True),
                    skip_validation=skip_validation,
                )
            
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
    
    def _expand_triplets(
        self,
        query: str,
        existing_facts: List[str],
        max_new_triplets: int,
        persist_to_graph: bool,
    ) -> str:
        """
        Expand triplets using LLM to generate additional related facts.
        
        Args:
            query: User's question
            existing_facts: List of fact strings in format "(subject, predicate, object)"
            max_new_triplets: Maximum new triplets to generate
            persist_to_graph: Whether to save to Neo4j
            
        Returns:
            JSON string with expanded triplets
        """
        import re
        
        # Parse existing facts into triplet tuples
        triplet_pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        existing_triplets = []
        
        for fact in existing_facts:
            match = re.search(triplet_pattern, fact)
            if match:
                s = match.group(1).strip().replace(" ", "_")
                p = match.group(2).strip().replace(" ", "_")
                o = match.group(3).strip().replace(" ", "_")
                existing_triplets.append((s, p, o))
        
        if not existing_triplets:
            return json.dumps({
                "error": "No valid triplets provided. Format: (subject, predicate, object)",
                "query": query,
            })
        
        # Validate max_new_triplets
        max_new_triplets = min(max(1, max_new_triplets), 15)
        
        # Call triplet expander
        try:
            expanded_triplets = self.triplet_expander.expand(
                query=query,
                retrieved_triplets=existing_triplets,
                max_new_triplets=max_new_triplets,
                temperature=0.3,
            )
        except Exception as e:
            return json.dumps({
                "error": f"Triplet expansion failed: {str(e)}",
                "query": query,
            })
        
        # Format expanded triplets
        expanded_facts = []
        for s, p, o in expanded_triplets:
            s_h = s.replace("_", " ")
            p_h = p.replace("_", " ")
            o_h = o.replace("_", " ")
            expanded_facts.append({
                "fact": f"({s_h}, {p_h}, {o_h})",
                "subject": s,
                "predicate": p,
                "object": o,
                "source": "llm_expanded",
            })
        
        # Persist to graph if requested
        persisted_count = 0
        if persist_to_graph and expanded_triplets:
            try:
                from .representation import EmbeddableItem
                import numpy as np
                
                # Generate embeddings for new triplets
                texts = [f"{s.replace('_', ' ')} {p.replace('_', ' ')} {o.replace('_', ' ')}" 
                         for s, p, o in expanded_triplets]
                embeddings = self.embedder.embed_texts(texts, normalize=True)
                
                # Create items with metadata
                items = []
                for (s, p, o), text in zip(expanded_triplets, texts):
                    items.append(EmbeddableItem(
                        text=text,
                        metadata={
                            "subject": s,
                            "predicate": p,
                            "object": o,
                            "document": f"({s}, {p}, {o})",
                            "source": "llm_expanded",
                            "source_text": "",
                            "representation_mode": "triplet_text",
                        }
                    ))
                
                # Add to store
                self.store.add(embeddings, items)
                persisted_count = len(items)
                logger.info(f"Persisted {persisted_count} expanded triplets to Neo4j")
                
            except Exception as e:
                logger.warning(f"Failed to persist expanded triplets: {e}")
        
        return json.dumps({
            "query": query,
            "existing_facts_count": len(existing_triplets),
            "expanded_facts_count": len(expanded_facts),
            "expanded_facts": expanded_facts,
            "persisted_to_graph": persist_to_graph,
            "persisted_count": persisted_count,
        }, ensure_ascii=False)
    
    def _validate_and_persist_triplets(
        self,
        query: str,
        proposed_triplets: List[str],
        existing_facts: List[str],
        persist_validated: bool,
        skip_validation: bool = False,
    ) -> str:
        """
        Validate proposed triplets using remote LLM and persist validated ones.
        
        This implements the dual-LLM validation workflow:
        1. Parse proposed triplets from local LLM
        2. Send to remote LLM for factual verification (unless skip_validation=True)
        3. Persist validated/corrected triplets to Neo4j
        4. Return validation report
        
        Args:
            query: User's original question
            proposed_triplets: List of triplet strings from local LLM
            existing_facts: Existing facts from the knowledge graph
            persist_validated: Whether to persist validated triplets
            skip_validation: Skip remote validation (for already-validated triplets)
            
        Returns:
            JSON string with validation report
        """
        import re
        from datetime import datetime
        from .triplet_validator import ValidatedTriplet, ValidationResult
        
        # Parse proposed triplets
        triplet_pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        parsed_triplets = []
        
        for fact in proposed_triplets:
            match = re.search(triplet_pattern, fact)
            if match:
                s = match.group(1).strip().replace(" ", "_")
                p = match.group(2).strip().replace(" ", "_")
                o = match.group(3).strip().replace(" ", "_")
                parsed_triplets.append((s, p, o))
        
        if not parsed_triplets:
            return json.dumps({
                "error": "No valid triplets provided. Format: (subject, predicate, object)",
                "query": query,
            })
        
        # Skip validation if requested (for already-validated triplets)
        if skip_validation:
            # Create a mock validation result with all triplets accepted
            validated_list = [
                ValidatedTriplet(
                    subject=s,
                    predicate=p,
                    object=o,
                    status="validated",
                    reason="Pre-validated triplet (skipped re-validation)",
                )
                for s, p, o in parsed_triplets
            ]
            validation_result = ValidationResult(
                validated=validated_list,
                rejected=[],
                query=query,
                timestamp=datetime.now().isoformat(),
                remote_model="skipped",
                local_model=self.local_model_name,
            )
            logger.info(f"Skipped validation for {len(parsed_triplets)} pre-validated triplets")
        else:
            # Validate with remote LLM
            try:
                validation_result = self.triplet_validator.validate(
                    proposed_triplets=parsed_triplets,
                    query=query,
                    existing_facts=existing_facts,
                )
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                return json.dumps({
                    "error": f"Validation failed: {str(e)}",
                    "query": query,
                })
        
        # Persist validated triplets if requested
        persisted_count = 0
        if persist_validated and validation_result.accepted_triplets:
            try:
                from .representation import EmbeddableItem
                import numpy as np
                
                # Generate embeddings for validated triplets
                texts = [
                    f"{s.replace('_', ' ')} {p.replace('_', ' ')} {o.replace('_', ' ')}"
                    for s, p, o in validation_result.accepted_triplets
                ]
                embeddings = self.embedder.embed_texts(texts, normalize=True)
                
                # Create items with metadata (including validation info)
                items = []
                for validated, text in zip(validation_result.validated, texts):
                    metadata = {
                        "subject": validated.subject,
                        "predicate": validated.predicate,
                        "object": validated.object,
                        "document": f"({validated.subject}, {validated.predicate}, {validated.object})",
                        "source": "remote_validated",
                        "source_text": "",
                        "representation_mode": "triplet_text",
                        "validation_status": validated.status,
                        "validation_reason": validated.reason,
                        "validated_at": datetime.now().isoformat(),
                        "local_llm": self.local_model_name,
                        "remote_llm": validation_result.remote_model,
                    }
                    
                    # Add original if this was corrected
                    if validated.original:
                        metadata["original_proposal"] = f"({validated.original[0]}, {validated.original[1]}, {validated.original[2]})"
                    
                    items.append(EmbeddableItem(text=text, metadata=metadata))
                
                # Add to store
                self.store.add(embeddings, items)
                persisted_count = len(items)
                logger.info(f"Persisted {persisted_count} validated triplets to Neo4j")
                
            except Exception as e:
                logger.warning(f"Failed to persist validated triplets: {e}")
        
        # Build response
        response = {
            "query": query,
            "proposed_count": len(parsed_triplets),
            "validated_count": len(validation_result.validated),
            "rejected_count": len(validation_result.rejected),
            "persisted_count": persisted_count,
            "remote_model": validation_result.remote_model,
            "validated_triplets": [
                {
                    "triplet": f"({v.subject}, {v.predicate}, {v.object})",
                    "status": v.status,
                    "reason": v.reason,
                    "original": f"({v.original[0]}, {v.original[1]}, {v.original[2]})" if v.original else None,
                }
                for v in validation_result.validated
            ],
            "rejected_triplets": [
                {
                    "triplet": f"({r.subject}, {r.predicate}, {r.object})",
                    "reason": r.reason,
                }
                for r in validation_result.rejected
            ],
            "user_notification": validation_result.get_user_notification(),
        }
        
        return json.dumps(response, ensure_ascii=False)


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
