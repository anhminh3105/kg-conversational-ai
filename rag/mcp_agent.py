"""
MCP-enabled Agent using Qwen2.5-7B-Instruct.

Implements an agentic loop for knowledge graph Q&A:
1. Receive user query
2. LLM decides which tools to call (if any)
3. Execute tools via MCP handler
4. LLM generates final answer with tool results

This module provides a self-contained agent that can autonomously
query the knowledge graph using MCP tools to answer questions.

Usage:
    from rag.mcp_agent import MCPAgent, create_mcp_agent
    
    # Quick setup
    agent = create_mcp_agent(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="password123",
    )
    
    # Ask a question
    result = agent.run("What did Einstein discover?")
    print(result["answer"])
    print(result["tool_calls"])  # Tools used to answer
"""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent run."""
    answer: str
    query: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    reasoning: Optional[str] = None
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "reasoning": self.reasoning,
            "warning": self.warning,
        }


class MCPAgent:
    """
    Agentic RAG system using Qwen2.5-7B with MCP tools.
    
    Qwen2.5-7B-Instruct has native tool/function calling support via its
    chat template, making it suitable for MCP-style interactions.
    
    The agent implements a ReAct-style loop:
    1. Think about what tool to use
    2. Call the tool
    3. Observe the result
    4. Repeat or provide final answer
    
    Attributes:
        tool_handler: MCP tool handler for Neo4j operations
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer
        max_iterations: Maximum tool call iterations
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantize: str = "4bit",
        device: str = "auto",
        max_iterations: int = 3,
        allow_cypher: bool = False,
    ):
        """
        Initialize the MCP Agent.
        
        Args:
            neo4j_store: Neo4jStore instance
            embedder: Embedder instance for semantic search
            model_name: HuggingFace model name
            quantize: Quantization mode ("4bit", "8bit", "none")
            device: Device for model ("auto", "cuda", "cpu")
            max_iterations: Maximum tool call iterations
            allow_cypher: Whether to allow raw Cypher queries
        """
        from .mcp_neo4j_server import Neo4jMCPToolHandler
        
        self.tool_handler = Neo4jMCPToolHandler(
            neo4j_store,
            embedder,
            allow_cypher=allow_cypher,
        )
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # Load model
        logger.info(f"Loading {model_name} with {quantize} quantization...")
        self._load_model(model_name, quantize, device)
        logger.info("Model loaded successfully")
    
    def _load_model(self, model_name: str, quantize: str, device: str):
        """Load the LLM with appropriate quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Configure quantization
        quant_config = None
        torch_dtype = torch.float16
        
        if quantize == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            torch_dtype = None  # Let bitsandbytes handle dtype
        elif quantize == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            torch_dtype = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": True,
        }
        
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
    
    def run(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        verbose: bool = False,
    ) -> AgentResult:
        """
        Run the agentic loop for a query.
        
        Args:
            query: User question
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens to generate per step
            verbose: Whether to print intermediate steps
            
        Returns:
            AgentResult with answer, tool calls, and metadata
        """
        logger.info(f"Agent running for query: {query}")
        
        # Build initial messages with tool definitions
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        tool_calls_log = []
        reasoning_steps = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Generate response
            response = self._generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if verbose:
                print(f"Model response: {response[:200]}...")
            
            # Check if model wants to call a tool
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                # Extract tool info
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                logger.info(f"Executing tool: {tool_name}({tool_args})")
                if verbose:
                    print(f"Tool call: {tool_name}({json.dumps(tool_args)})")
                
                # Execute tool
                tool_result = self.tool_handler.handle_tool_call(tool_name, tool_args)
                
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": json.loads(tool_result),
                })
                
                if verbose:
                    print(f"Tool result: {tool_result[:300]}...")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{tool_result}\n\nBased on this information, please provide your answer or call another tool if needed."
                })
                
                reasoning_steps.append(f"Called {tool_name} to get information")
            else:
                # No tool call - this is the final answer
                # Clean up the response
                answer = self._extract_answer(response)
                
                return AgentResult(
                    answer=answer,
                    query=query,
                    tool_calls=tool_calls_log,
                    iterations=iteration + 1,
                    reasoning="; ".join(reasoning_steps) if reasoning_steps else None,
                )
        
        # Max iterations reached - extract best answer from last response
        answer = self._extract_answer(response)
        
        return AgentResult(
            answer=answer,
            query=query,
            tool_calls=tool_calls_log,
            iterations=self.max_iterations,
            reasoning="; ".join(reasoning_steps) if reasoning_steps else None,
            warning="Max iterations reached - answer may be incomplete",
        )
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt with tool definitions."""
        tools = self.tool_handler.get_tools()
        tools_json = json.dumps(tools, indent=2)
        
        return load_prompt("mcp_agent_system", tools_json=tools_json)
    
    def _generate(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response from the model."""
        import torch
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """
        Extract tool call from model response if present.
        
        Handles multiple formats that the model might use.
        """
        # Pattern 1: {"tool": "name", "arguments": {...}}
        patterns = [
            r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
            r'```json\s*(\{[^`]+\})\s*```',
            r'```\s*(\{[^`]+\})\s*```',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    if i < 2:  # First two patterns extract name and args separately
                        return {
                            "name": match.group(1),
                            "arguments": json.loads(match.group(2)),
                        }
                    else:  # JSON block patterns
                        data = json.loads(match.group(1))
                        if "tool" in data:
                            return {
                                "name": data["tool"],
                                "arguments": data.get("arguments", {}),
                            }
                        elif "name" in data:
                            return {
                                "name": data["name"],
                                "arguments": data.get("arguments", {}),
                            }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Failed to parse tool call pattern {i}: {e}")
                    continue
        
        # Try to find any JSON object in the response
        try:
            # Look for a JSON object that might be a tool call
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                if "tool" in data:
                    return {
                        "name": data["tool"],
                        "arguments": data.get("arguments", {}),
                    }
        except (json.JSONDecodeError, KeyError):
            pass
        
        return None
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the response.
        
        Removes any JSON tool calls or formatting artifacts.
        """
        # Remove any JSON blocks
        answer = re.sub(r'```json\s*\{[^`]+\}\s*```', '', response)
        answer = re.sub(r'```\s*\{[^`]+\}\s*```', '', answer)
        
        # Remove tool call JSON
        answer = re.sub(r'\{\s*"tool"\s*:[^}]+\}', '', answer)
        
        # Clean up whitespace
        answer = answer.strip()
        
        # If the answer is empty, return the original response
        if not answer:
            return response.strip()
        
        return answer


def create_mcp_agent(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = None,
    neo4j_database: str = "neo4j",
    embedding_model: str = None,
    llm_model: str = None,
    quantize: str = None,
    device: str = "auto",
    max_iterations: int = 3,
    allow_cypher: bool = False,
) -> MCPAgent:
    """
    Convenience function to create an MCP Agent.
    
    Uses environment variables for configuration when parameters are not provided:
    - NEO4J_PASSWORD or default "password123"
    - LOCAL_EMBEDDER_MODEL or default "BAAI/bge-small-en-v1.5"
    - LOCAL_LLM_MODEL or default "Qwen/Qwen2.5-7B-Instruct"
    - LOCAL_LLM_QUANTIZE or default "4bit"
    
    Args:
        neo4j_uri: Neo4j Bolt URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        embedding_model: Embedding model name
        llm_model: LLM model name
        quantize: Quantization mode
        device: Device for models
        max_iterations: Maximum tool call iterations
        allow_cypher: Whether to allow raw Cypher queries
        
    Returns:
        Configured MCPAgent instance
    """
    from .neo4j_store import Neo4jStore
    from .embedder import get_embedder
    
    # Get configuration from environment or use defaults
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    if embedding_model is None:
        embedding_model = os.environ.get(
            "LOCAL_EMBEDDER_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
    
    if llm_model is None:
        llm_model = os.environ.get(
            "LOCAL_LLM_MODEL",
            "Qwen/Qwen2.5-7B-Instruct"
        )
    
    if quantize is None:
        quantize = os.environ.get("LOCAL_LLM_QUANTIZE", "4bit")
    
    logger.info(f"Creating MCP Agent:")
    logger.info(f"  Neo4j: {neo4j_uri}")
    logger.info(f"  Embedding model: {embedding_model}")
    logger.info(f"  LLM model: {llm_model}")
    logger.info(f"  Quantization: {quantize}")
    
    # Create Neo4j store
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    
    # Create embedder
    embedder = get_embedder(model_name=embedding_model)
    
    # Create agent
    agent = MCPAgent(
        neo4j_store=neo4j_store,
        embedder=embedder,
        model_name=llm_model,
        quantize=quantize,
        device=device,
        max_iterations=max_iterations,
        allow_cypher=allow_cypher,
    )
    
    return agent


class MCPAgentLite:
    """
    Lightweight MCP Agent that uses an external LLM (API-based).
    
    This version doesn't load a local model, instead using the
    openai_chat_completion function which can route to various backends.
    
    Useful when:
    - Running on CPU-only machines
    - Using cloud LLM APIs (Google AI, SambaNova, OpenAI)
    - Memory is constrained
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        max_iterations: int = 3,
        allow_cypher: bool = False,
    ):
        """
        Initialize the lightweight MCP Agent.
        
        Args:
            neo4j_store: Neo4jStore instance
            embedder: Embedder instance
            max_iterations: Maximum tool call iterations
            allow_cypher: Whether to allow raw Cypher queries
        """
        from .mcp_neo4j_server import Neo4jMCPToolHandler
        
        self.tool_handler = Neo4jMCPToolHandler(
            neo4j_store,
            embedder,
            allow_cypher=allow_cypher,
        )
        self.max_iterations = max_iterations
    
    def run(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        verbose: bool = False,
    ) -> AgentResult:
        """
        Run the agentic loop using external LLM API.
        
        Args:
            query: User question
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens per step
            verbose: Print intermediate steps
            
        Returns:
            AgentResult with answer and metadata
        """
        from .edc.edc.utils.llm_utils import openai_chat_completion
        
        system_prompt = self._get_system_prompt()
        history = [{"role": "user", "content": query}]
        
        tool_calls_log = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Generate response via API
            response = openai_chat_completion(
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if verbose:
                print(f"Model response: {response[:200]}...")
            
            # Check for tool call
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                if verbose:
                    print(f"Tool call: {tool_name}({json.dumps(tool_args)})")
                
                # Execute tool
                tool_result = self.tool_handler.handle_tool_call(tool_name, tool_args)
                
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": json.loads(tool_result),
                })
                
                # Update history
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{tool_result}\n\nBased on this, provide your answer or call another tool."
                })
            else:
                # Final answer
                return AgentResult(
                    answer=response.strip(),
                    query=query,
                    tool_calls=tool_calls_log,
                    iterations=iteration + 1,
                )
        
        return AgentResult(
            answer=response.strip(),
            query=query,
            tool_calls=tool_calls_log,
            iterations=self.max_iterations,
            warning="Max iterations reached",
        )
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt with tool definitions."""
        from .mcp_neo4j_server import format_tools_for_prompt
        
        tools_text = format_tools_for_prompt(self.tool_handler.get_tools())
        
        return load_prompt("mcp_agent_lite_system", tools_text=tools_text)
    
    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """Extract tool call from response."""
        # Same parsing logic as MCPAgent
        patterns = [
            r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return {
                        "name": match.group(1),
                        "arguments": json.loads(match.group(2)),
                    }
                except json.JSONDecodeError:
                    continue
        
        return None


@dataclass
class ValidatedAgentResult(AgentResult):
    """Result from an agent run with validation support."""
    knowledge_gap_detected: bool = False
    proposed_triplets: List[str] = field(default_factory=list)
    validated_triplets: List[Dict[str, Any]] = field(default_factory=list)
    rejected_triplets: List[Dict[str, Any]] = field(default_factory=list)
    persisted_count: int = 0
    new_facts_notification: str = ""
    # New fields for detailed intermediate messaging
    intermediate_messages: List[str] = field(default_factory=list)
    llm_assessment: Dict[str, Any] = field(default_factory=dict)
    validation_failed: bool = False
    validation_attempts: int = 0
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    # Iterative knowledge expansion tracking
    knowledge_iterations: int = 1  # Number of search-assess-expand cycles completed
    # Remote model tracking
    remote_model_name: str = ""  # Which frontier model validated the triplets
    # Persistence justification
    persistence_justification: Dict[str, Any] = field(default_factory=dict)
    skipped_triplets: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result.update({
            "knowledge_gap_detected": self.knowledge_gap_detected,
            "proposed_triplets": self.proposed_triplets,
            "validated_triplets": self.validated_triplets,
            "rejected_triplets": self.rejected_triplets,
            "persisted_count": self.persisted_count,
            "new_facts_notification": self.new_facts_notification,
            "intermediate_messages": self.intermediate_messages,
            "llm_assessment": self.llm_assessment,
            "validation_failed": self.validation_failed,
            "validation_attempts": self.validation_attempts,
            "validation_history": self.validation_history,
            "knowledge_iterations": self.knowledge_iterations,
            "remote_model_name": self.remote_model_name,
            "persistence_justification": self.persistence_justification,
            "skipped_triplets": self.skipped_triplets,
        })
        return result


class MCPAgentWithValidation:
    """
    MCP Agent with dual-LLM validation for knowledge expansion.
    
    This agent implements the complete workflow:
    1. Query the knowledge graph for relevant facts
    2. Detect if there's a knowledge gap (insufficient facts)
    3. If gap detected, use local LLM to propose new triplets
    4. Send proposed triplets to remote LLM for validation
    5. Generate final answer using existing + validated facts
    6. Local LLM justifies which validated triplets to persist
    7. Persist only justified triplets to Neo4j
    
    Persistence is deferred until after the answer is generated, and the
    local LLM acts as a quality gate to prevent useless/vague triplets
    from polluting the knowledge graph.
    
    Usage:
        agent = MCPAgentWithValidation(neo4j_store, embedder)
        result = agent.run("What awards did Einstein win?")
        
        print(result.answer)
        print(result.persisted_count)        # How many triplets were persisted
        print(result.skipped_triplets)       # Triplets the LLM decided not to persist
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        max_iterations: int = 5,
        allow_cypher: bool = False,
        enable_validation: bool = True,
        auto_expand: bool = True,
        max_validation_retries: int = 3,
        max_knowledge_iterations: int = 3,
        local_model_name: str = "local_llm",
    ):
        """
        Initialize the agent with validation support.
        
        This agent uses an LLM-based approach to detect knowledge gaps
        (instead of heuristics) and includes a retry mechanism for
        validation with remote LLM feedback.
        
        Args:
            neo4j_store: Neo4jStore instance
            embedder: Embedder instance for semantic search
            max_iterations: Maximum tool call iterations
            allow_cypher: Whether to allow raw Cypher queries
            enable_validation: Whether to enable remote validation
            auto_expand: Whether to automatically expand when gaps detected
            max_validation_retries: Maximum retries for validation (default: 3)
            max_knowledge_iterations: Maximum search-assess-expand cycles (default: 3)
            local_model_name: Name of the local LLM for tracking
        """
        from .mcp_neo4j_server import Neo4jMCPToolHandler
        
        self.neo4j_store = neo4j_store
        self.embedder = embedder
        self.max_iterations = max_iterations
        self.auto_expand = auto_expand
        self.max_validation_retries = max_validation_retries
        self.max_knowledge_iterations = max_knowledge_iterations
        self.local_model_name = local_model_name
        
        # Initialize tool handler with validation enabled
        self.tool_handler = Neo4jMCPToolHandler(
            neo4j_store,
            embedder,
            allow_cypher=allow_cypher,
            enable_expansion=True,
            enable_validation=enable_validation,
            local_model_name=local_model_name,
        )
        
        logger.info(
            f"Initialized MCPAgentWithValidation "
            f"(validation={enable_validation}, auto_expand={auto_expand}, "
            f"max_retries={max_validation_retries}, max_knowledge_iterations={max_knowledge_iterations})"
        )
    
    def run(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        verbose: bool = False,
        force_expand: bool = False,
    ) -> ValidatedAgentResult:
        """
        Run the agent with iterative knowledge expansion.
        
        This method implements an iterative workflow:
        1. Search the knowledge graph for relevant facts
        2. Use local LLM to assess if facts are sufficient
        3. If insufficient, propose triplets and validate with remote LLM
        4. Collect validated triplets in memory (no persistence yet)
        5. Repeat until sufficient knowledge or max iterations reached
        6. Generate final answer using existing + validated facts
        7. Local LLM justifies which validated triplets to persist
        8. Persist only the justified triplets to Neo4j
        
        Args:
            query: User question
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens per step
            verbose: Print intermediate steps
            force_expand: Force triplet expansion even without detected gap
            
        Returns:
            ValidatedAgentResult with answer, validation info, intermediate messages, and new facts
        """
        from .edc.edc.utils.llm_utils import openai_chat_completion
        
        logger.info(f"Agent running for query: {query}")
        
        # Initialize tracking variables for cumulative results across all iterations
        tool_calls_log = []
        intermediate_messages = []
        all_validation_history = []
        all_proposed_triplets = []
        all_validated_triplets = []
        all_rejected_triplets = []
        total_persisted_count = 0
        llm_assessment = {}
        validation_failed = False
        total_validation_attempts = 0
        knowledge_iterations_completed = 0
        remote_model_name = ""
        
        # Track all known facts (from search + validated) to avoid re-proposing
        all_known_fact_strings = set()
        
        # Helper function for triplet normalization (used across iterations)
        def normalize_triplet_string(s: str) -> str:
            """Normalize a triplet string for comparison."""
            return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "").replace(",", "|")
        
        # ============================================================
        # ITERATIVE KNOWLEDGE EXPANSION LOOP
        # ============================================================
        for knowledge_iteration in range(1, self.max_knowledge_iterations + 1):
            knowledge_iterations_completed = knowledge_iteration
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"KNOWLEDGE ITERATION {knowledge_iteration}/{self.max_knowledge_iterations}")
                print(f"{'='*60}")
            
            # ----------------------------------------------------------
            # Step 1: Search knowledge graph
            # ----------------------------------------------------------
            if verbose:
                print(f"\n--- Iteration {knowledge_iteration}, Step 1: Searching knowledge graph ---")
            
            search_result = self.tool_handler.handle_tool_call(
                "search_knowledge_graph",
                {"query": query, "top_k": 10}
            )
            search_data = json.loads(search_result)
            
            tool_calls_log.append({
                "iteration": len(tool_calls_log) + 1,
                "knowledge_iteration": knowledge_iteration,
                "tool": "search_knowledge_graph",
                "arguments": {"query": query, "top_k": 10},
                "result": search_data,
            })
            
            if verbose:
                print(f"Found {search_data.get('num_results', 0)} facts")
            
            # Extract facts for assessment
            existing_fact_strings = []
            for fact in search_data.get("facts", []):
                fact_str = fact.get("fact", "")
                existing_fact_strings.append(fact_str)
                all_known_fact_strings.add(normalize_triplet_string(fact_str))
            
            # ----------------------------------------------------------
            # Step 2: LLM-based knowledge sufficiency assessment
            # ----------------------------------------------------------
            if verbose:
                print(f"\n--- Iteration {knowledge_iteration}, Step 2: LLM Assessing Knowledge Sufficiency ---")
            
            llm_assessment = self._assess_knowledge_sufficiency(
                query, existing_fact_strings, temperature
            )
            
            # On first iteration or if forced, respect force_expand
            knowledge_gap_detected = llm_assessment["assessment"] == "INSUFFICIENT"
            if knowledge_iteration == 1 and force_expand:
                knowledge_gap_detected = True
            
            if verbose:
                print(f"Assessment: {llm_assessment['assessment']} (confidence: {llm_assessment['confidence']:.2f})")
                if knowledge_gap_detected:
                    print(f"Missing information: {len(llm_assessment.get('missing_information', []))} items")
                    print(f"Proposed triplets: {len(llm_assessment.get('proposed_triplets', []))}")
            
            # ----------------------------------------------------------
            # Check if we can answer now (SUFFICIENT)
            # ----------------------------------------------------------
            if not knowledge_gap_detected:
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] Knowledge is SUFFICIENT - proceeding to answer generation")
                break
            
            # ----------------------------------------------------------
            # Step 3: If gap detected, propose and validate triplets
            # ----------------------------------------------------------
            if not self.auto_expand:
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] Auto-expand disabled - proceeding to answer generation")
                break
            
            proposed_triplets = llm_assessment.get("proposed_triplets", [])
            
            # Deduplicate: Remove triplets that already exist OR were previously proposed/validated
            if proposed_triplets:
                original_count = len(proposed_triplets)
                
                unique_triplets = []
                for triplet in proposed_triplets:
                    if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                        triplet_str = f"{triplet[0]}|{triplet[1]}|{triplet[2]}"
                        normalized = normalize_triplet_string(triplet_str)
                        if normalized not in all_known_fact_strings:
                            unique_triplets.append(triplet)
                            # Add to known facts to prevent re-proposing in future iterations
                            all_known_fact_strings.add(normalized)
                        else:
                            logger.info(f"Skipping duplicate triplet (already known): {triplet}")
                
                proposed_triplets = unique_triplets
                
                if verbose and original_count != len(proposed_triplets):
                    print(f"  Filtered out {original_count - len(proposed_triplets)} duplicate triplets")
                
                llm_assessment["proposed_triplets"] = proposed_triplets
            
            # If no new triplets to propose, we can't make progress
            if not proposed_triplets:
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] No new triplets to propose - proceeding to answer generation")
                break
            
            # Track all proposed triplets
            all_proposed_triplets.extend([str(t) for t in proposed_triplets])
            
            # Generate and show gap detection message
            gap_message = self._generate_gap_detection_message(llm_assessment)
            intermediate_messages.append(gap_message)
            if verbose:
                print(gap_message)
            
            # ----------------------------------------------------------
            # Step 4: Validate with retry mechanism
            # ----------------------------------------------------------
            if not self.tool_handler.enable_validation:
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] Validation disabled - proceeding to answer generation")
                break
            
            validated_triplets, success, failure_msg, validation_history, iter_remote_model = self._validate_with_retry(
                query=query,
                proposed_triplets=proposed_triplets,
                missing_information=llm_assessment.get("missing_information", []),
                existing_facts=existing_fact_strings,
                max_retries=self.max_validation_retries,
                temperature=temperature,
                verbose=verbose,
                intermediate_messages=intermediate_messages,
                knowledge_iteration=knowledge_iteration,
            )
            
            # Track the remote model name from validation
            if iter_remote_model and iter_remote_model != "unknown":
                remote_model_name = iter_remote_model
            
            iteration_validation_attempts = len(validation_history)
            total_validation_attempts += iteration_validation_attempts
            all_validation_history.extend(validation_history)
            
            # Collect rejected triplets from this iteration
            iteration_rejected = []
            for vh in validation_history:
                iteration_rejected.extend(vh.get("rejected", []))
            all_rejected_triplets.extend(iteration_rejected)
            
            if not success:
                # All validation attempts failed for this iteration
                validation_failed = True
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] Validation failed - proceeding to answer generation")
                break
            
            # ----------------------------------------------------------
            # Step 5: Collect validated triplets (defer persistence)
            # ----------------------------------------------------------
            if validated_triplets:
                # Track validated triplets and add to known facts
                all_validated_triplets.extend(validated_triplets)
                for vt in validated_triplets:
                    triplet_str = vt.get("triplet", "")
                    all_known_fact_strings.add(normalize_triplet_string(triplet_str))
                    # Add to existing_fact_strings so next iteration sees them
                    existing_fact_strings.append(triplet_str)
            
            # ----------------------------------------------------------
            # Generate iteration summary
            # ----------------------------------------------------------
            will_continue = knowledge_iteration < self.max_knowledge_iterations
            iteration_summary = self._generate_iteration_summary_message(
                iteration=knowledge_iteration,
                max_iterations=self.max_knowledge_iterations,
                validated_count=len(validated_triplets),
                rejected_count=len(iteration_rejected),
                persisted_count=0,  # Deferred until after answer
                assessment=llm_assessment["assessment"],
                will_continue=will_continue,
            )
            intermediate_messages.append(iteration_summary)
            if verbose:
                print(iteration_summary)
            
            # If no triplets were validated, we can't make progress
            if not validated_triplets:
                if verbose:
                    print(f"\n[Iteration {knowledge_iteration}] No triplets validated - cannot make progress")
                break
        
        # ============================================================
        # GENERATE FINAL ANSWER
        # ============================================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERATING FINAL ANSWER (after {knowledge_iterations_completed} iteration(s))")
            print(f"{'='*60}")
        
        # Build final facts: existing KG facts + validated (not-yet-persisted) triplets
        final_fact_strings = list(existing_fact_strings)
        validated_fact_strings = []
        for vt in all_validated_triplets:
            triplet_str = vt.get("triplet", "")
            if triplet_str and triplet_str not in final_fact_strings:
                final_fact_strings.append(triplet_str)
                validated_fact_strings.append(triplet_str)
        
        system_prompt = self._get_answer_prompt(
            knowledge_gap_detected,
            len(all_validated_triplets) > 0,
            remote_model_name=remote_model_name,
            num_validated=len(all_validated_triplets),
        )
        
        facts_context = "\n".join(f"- {f}" for f in final_fact_strings)
        
        # Build context about validated triplets for the answer prompt
        if validated_fact_strings:
            validated_context = "\n".join(f"- {f} [validated by {remote_model_name}]" for f in validated_fact_strings)
            user_message = f"""Question: {query}

Available facts from knowledge graph:
{facts_context}

Additionally, the following {len(validated_fact_strings)} fact(s) were proposed to fill knowledge gaps and validated by {remote_model_name}:
{validated_context}

The knowledge graph did not have sufficient information to answer this question directly. Please answer the question using both the existing facts and the validated facts above. Be concise and accurate. Clearly state that the knowledge graph was insufficient and that you are supplementing with {len(validated_fact_strings)} validated fact(s)."""
        else:
            user_message = f"""Question: {query}

Available facts from knowledge graph:
{facts_context}

Please answer the question based on these facts. Be concise and accurate."""
        
        history = [{"role": "user", "content": user_message}]
        
        answer = openai_chat_completion(
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if verbose:
            print(f"\nFinal answer: {answer[:200]}...")
        
        # ============================================================
        # PERSISTENCE JUSTIFICATION & EXECUTION (after answer)
        # ============================================================
        persistence_justification = {}
        skipped_triplets = []
        
        if all_validated_triplets:
            if verbose:
                print(f"\n{'='*60}")
                print(f"PERSISTENCE JUSTIFICATION")
                print(f"{'='*60}")
            
            # Ask local LLM whether each validated triplet should be persisted
            persistence_justification = self._justify_persistence(
                query=query,
                answer=answer,
                validated_triplets=all_validated_triplets,
                temperature=temperature,
            )
            
            triplets_to_persist = persistence_justification.get("persist", [])
            skipped_triplets = persistence_justification.get("skip", [])
            
            if verbose:
                print(f"  Persist: {len(triplets_to_persist)} triplets")
                for tp in triplets_to_persist:
                    print(f"    + {tp.get('triplet', '')} - {tp.get('reason', '')}")
                print(f"  Skip: {len(skipped_triplets)} triplets")
                for sk in skipped_triplets:
                    print(f"    - {sk.get('triplet', '')} - {sk.get('reason', '')}")
            
            # Persist only the justified triplets
            if triplets_to_persist:
                triplet_strings = [t["triplet"] for t in triplets_to_persist]
                persist_result = self.tool_handler.handle_tool_call(
                    "validate_and_persist_triplets",
                    {
                        "query": query,
                        "proposed_triplets": triplet_strings,
                        "existing_facts": existing_fact_strings,
                        "persist_validated": True,
                        "skip_validation": True,
                    }
                )
                persist_data = json.loads(persist_result)
                total_persisted_count = persist_data.get("persisted_count", 0)
                
                tool_calls_log.append({
                    "iteration": len(tool_calls_log) + 1,
                    "tool": "validate_and_persist_triplets",
                    "arguments": {"query": query, "persist": True},
                    "result": persist_data,
                })
                
                if verbose:
                    print(f"\n  Persisted {total_persisted_count} triplets to Neo4j")
        
        return ValidatedAgentResult(
            answer=answer,
            query=query,
            tool_calls=tool_calls_log,
            iterations=len(tool_calls_log),
            knowledge_gap_detected=knowledge_gap_detected,
            proposed_triplets=all_proposed_triplets,
            validated_triplets=all_validated_triplets,
            rejected_triplets=all_rejected_triplets,
            persisted_count=total_persisted_count,
            new_facts_notification="",
            intermediate_messages=intermediate_messages,
            llm_assessment=llm_assessment,
            validation_failed=validation_failed,
            validation_attempts=total_validation_attempts,
            validation_history=all_validation_history,
            knowledge_iterations=knowledge_iterations_completed,
            remote_model_name=remote_model_name,
            persistence_justification=persistence_justification,
            skipped_triplets=skipped_triplets,
        )
    
    def _get_answer_prompt(
        self,
        gap_detected: bool,
        new_facts_added: bool,
        remote_model_name: str = "",
        num_validated: int = 0,
    ) -> str:
        """Generate system prompt for answer generation."""
        if gap_detected and new_facts_added:
            return load_prompt(
                "answer_generation_with_new_facts",
                remote_model_name=remote_model_name or "remote LLM",
                num_validated=num_validated,
            )
        return load_prompt("answer_generation")
    
    def _get_assessment_prompt(self) -> str:
        """Return the system prompt for knowledge assessment."""
        return load_prompt("knowledge_assessment")
    
    def _assess_knowledge_sufficiency(
        self,
        query: str,
        facts: List[str],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Ask the local LLM to assess if facts are sufficient.
        
        Args:
            query: The user's question
            facts: List of fact strings from the knowledge graph
            temperature: LLM sampling temperature
            
        Returns:
            Dict with: assessment, confidence, answer, missing_information, proposed_triplets
        """
        from .edc.edc.utils.llm_utils import openai_chat_completion
        
        system_prompt = self._get_assessment_prompt()
        
        facts_text = "\n".join(f"- {f}" for f in facts) if facts else "No facts available."
        
        user_message = f"""## Question
{query}

## Available Facts
{facts_text}

Analyze these facts and respond with the JSON assessment."""
        
        history = [{"role": "user", "content": user_message}]
        
        response = openai_chat_completion(
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=1024,
        )
        
        return self._parse_assessment_response(response)
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the structured JSON response from the LLM.
        
        Handles:
        - JSON extraction from markdown code blocks
        - Fallback parsing if LLM doesn't follow format exactly
        - Validation of triplet format
        - Default to "INSUFFICIENT" if parsing fails (conservative approach)
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed assessment dictionary
        """
        # Default response for parsing failures
        default_response = {
            "assessment": "INSUFFICIENT",
            "confidence": 0.0,
            "answer": "Unable to assess facts.",
            "missing_information": ["Unable to parse LLM response"],
            "proposed_triplets": [],
        }
        
        try:
            # Try to extract JSON from markdown code blocks
            json_patterns = [
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
                r'(\{[\s\S]*"assessment"[\s\S]*\})',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    break
            
            if not json_str:
                # Try to parse the entire response as JSON
                json_str = response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate and normalize the response
            assessment = data.get("assessment", "INSUFFICIENT").upper()
            if assessment not in ["SUFFICIENT", "INSUFFICIENT"]:
                assessment = "INSUFFICIENT"
            
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            answer = data.get("answer", "")
            missing_information = data.get("missing_information", [])
            
            # Parse and validate triplets
            proposed_triplets = []
            raw_triplets = data.get("proposed_triplets", [])
            for triplet in raw_triplets:
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    proposed_triplets.append([str(t) for t in triplet[:3]])
            
            return {
                "assessment": assessment,
                "confidence": confidence,
                "answer": answer,
                "missing_information": missing_information,
                "proposed_triplets": proposed_triplets,
            }
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse assessment response: {e}")
            logger.debug(f"Raw response: {response}")
            return default_response
    
    def _generate_gap_detection_message(
        self,
        assessment: Dict[str, Any],
    ) -> str:
        """
        Generate message showing knowledge gap and proposed triplets.
        
        Args:
            assessment: The LLM's assessment result
            
        Returns:
            Formatted message string for the user
        """
        missing = assessment.get("missing_information", [])
        proposed = assessment.get("proposed_triplets", [])
        
        lines = [
            "=" * 50,
            "KNOWLEDGE GAP DETECTED",
            "=" * 50,
            "The knowledge graph does not contain sufficient information to answer your question.",
            f"I will propose {len(proposed)} fact(s) from my own understanding for external validation.",
            "",
            "What's missing:",
        ]
        for item in missing:
            lines.append(f"  - {item}")
        
        lines.extend(["", f"Proposed facts ({len(proposed)}) to fill these gaps:"])
        for i, triplet in enumerate(proposed, 1):
            if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                lines.append(f"  {i}. ({triplet[0]}, {triplet[1]}, {triplet[2]})")
            else:
                lines.append(f"  {i}. {triplet}")
        
        lines.extend([
            "",
            f"Next action: Sending {len(proposed)} proposed facts to remote LLM for validation...",
            "=" * 50,
        ])
        return "\n".join(lines)
    
    def _generate_validation_attempt_message(
        self,
        attempt: int,
        max_attempts: int,
        validated: List[Dict],
        rejected: List[Dict],
        remote_model: str = "unknown",
    ) -> str:
        """
        Generate message showing validation attempt results.
        
        Args:
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            validated: List of validated triplet dicts
            rejected: List of rejected triplet dicts
            remote_model: Name of the remote model used for validation
            
        Returns:
            Formatted message string for the user
        """
        total = len(validated) + len(rejected)
        all_rejected = len(validated) == 0 and len(rejected) > 0
        
        header = f"VALIDATION ATTEMPT {attempt}/{max_attempts} (by {remote_model})"
        if all_rejected:
            header += " - ALL REJECTED"
        
        lines = [
            "=" * 50,
            header,
            "=" * 50,
            f"Sending {total} proposed triplets to {remote_model} for validation...",
            "",
            f"Response from {remote_model}:",
        ]
        
        if validated:
            lines.append("  ACCEPTED:")
            for v in validated:
                triplet = v.get("triplet", "")
                reason = v.get("reason", "Validated")
                status = v.get("status", "validated")
                status_note = " (corrected)" if status == "corrected" else ""
                lines.append(f"    - {triplet}{status_note} ✓")
                lines.append(f"      Reason: {reason}")
        
        if rejected:
            lines.append("  REJECTED:")
            for r in rejected:
                triplet = r.get("triplet", "")
                reason = r.get("reason", "Rejected")
                lines.append(f"    - {triplet} ✗")
                lines.append(f"      Reason: {reason}")
        
        lines.extend(["", f"Result: {len(validated)} accepted, {len(rejected)} rejected", ""])
        
        # Next action
        if len(validated) > 0:
            lines.append(f"Next action: Using {len(validated)} validated facts to generate answer (persistence deferred until after answer)...")
        elif attempt < max_attempts:
            lines.append("Next action: Analyzing rejection feedback and proposing new facts...")
        else:
            lines.append("Next action: All attempts exhausted. Informing user...")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _generate_persistence_message(
        self,
        validated_triplets: List[Dict],
    ) -> str:
        """
        Generate message confirming triplet persistence.
        
        Args:
            validated_triplets: List of validated triplet dicts that were persisted
            
        Returns:
            Formatted message string for the user
        """
        lines = [
            "=" * 50,
            "PERSISTING VALIDATED FACTS",
            "=" * 50,
            f"Adding {len(validated_triplets)} validated triplets to knowledge base:",
        ]
        for i, v in enumerate(validated_triplets, 1):
            triplet = v.get("triplet", "")
            lines.append(f"  {i}. {triplet}")
        
        lines.extend([
            "",
            "Status: Successfully persisted to Neo4j",
            "",
            "Next action: Generating complete answer with new facts...",
            "=" * 50,
        ])
        return "\n".join(lines)
    
    def _generate_iteration_summary_message(
        self,
        iteration: int,
        max_iterations: int,
        validated_count: int,
        rejected_count: int,
        persisted_count: int,
        assessment: str,
        will_continue: bool,
    ) -> str:
        """
        Generate message summarizing a knowledge iteration cycle.
        
        Args:
            iteration: Current iteration number (1-indexed)
            max_iterations: Maximum number of iterations
            validated_count: Number of validated triplets this iteration
            rejected_count: Number of rejected triplets this iteration
            persisted_count: Number of persisted triplets this iteration (0 when deferred)
            assessment: LLM assessment result (SUFFICIENT/INSUFFICIENT)
            will_continue: Whether another iteration will be attempted
            
        Returns:
            Formatted message string summarizing the iteration
        """
        lines = [
            "=" * 50,
            f"KNOWLEDGE ITERATION {iteration}/{max_iterations} COMPLETE",
            "=" * 50,
            f"Assessment: {assessment}",
            f"Validated: {validated_count} triplets",
            f"Rejected: {rejected_count} triplets",
            f"Persistence: deferred until after answer",
            "",
        ]
        
        if will_continue:
            lines.extend([
                "Status: Knowledge still insufficient",
                "Next action: Re-assessing with validated facts included...",
            ])
        else:
            if assessment == "SUFFICIENT":
                lines.extend([
                    "Status: Knowledge is now sufficient",
                    "Next action: Generating final answer...",
                ])
            else:
                lines.extend([
                    "Status: Max iterations reached or no progress",
                    "Next action: Generating best possible answer with available facts...",
                ])
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _generate_retry_message(
        self,
        attempt: int,
        max_attempts: int,
        new_triplets: List[List[str]],
    ) -> str:
        """
        Generate message showing new proposal after rejection.
        
        Args:
            attempt: Current attempt number (before the retry)
            max_attempts: Maximum number of attempts
            new_triplets: List of new proposed triplets
            
        Returns:
            Formatted message string for the user
        """
        lines = [
            "=" * 50,
            f"RETRY PROPOSAL (Attempt {attempt + 1}/{max_attempts})",
            "=" * 50,
            "Based on rejection feedback, proposing new facts:",
        ]
        for i, triplet in enumerate(new_triplets, 1):
            if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                lines.append(f"  {i}. ({triplet[0]}, {triplet[1]}, {triplet[2]})")
            else:
                lines.append(f"  {i}. {triplet}")
        
        lines.extend([
            "",
            "Next action: Sending new proposals for validation...",
            "=" * 50,
        ])
        return "\n".join(lines)
    
    def _generate_failure_message(self, query: str, attempts: int) -> str:
        """
        Generate user-friendly message when all validation attempts fail.
        
        Args:
            query: The original user query
            attempts: Number of validation attempts made
            
        Returns:
            Formatted failure message string
        """
        lines = [
            "=" * 50,
            "VALIDATION FAILED",
            "=" * 50,
            f"I apologize, but I was unable to find or generate verified facts",
            f"to answer your question after {attempts} attempts.",
            "",
            "The external verification service could not confirm the accuracy",
            "of the proposed information.",
            "",
            "For reliable information about this topic, I recommend:",
            "- Consulting authoritative sources directly",
            "- Rephrasing your question with more specific details",
            "- Asking about a related topic where verified facts are available",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def _get_rejection_feedback(self, rejected_triplets: List[Dict]) -> str:
        """
        Extract rejection reasons from validation result.
        
        Args:
            rejected_triplets: List of rejected triplet dicts with reasons
            
        Returns:
            Formatted feedback string
        """
        feedback_parts = []
        for rejected in rejected_triplets:
            triplet = rejected.get("triplet", "")
            reason = rejected.get("reason", "Unknown reason")
            feedback_parts.append(f"- {triplet}: {reason}")
        return "\n".join(feedback_parts) if feedback_parts else "No specific feedback available."
    
    def _get_reproposal_prompt(self) -> str:
        """Return the system prompt for re-proposing triplets after rejection."""
        return load_prompt("triplet_reproposal")
    
    def _repropose_triplets(
        self,
        query: str,
        missing_information: List[str],
        rejection_feedback: str,
        previous_triplets: List[List[str]],
        temperature: float = 0.1,
    ) -> List[List[str]]:
        """
        Ask local LLM to propose new triplets based on rejection feedback.
        
        Args:
            query: The original user query
            missing_information: List of missing information items
            rejection_feedback: Formatted rejection reasons
            previous_triplets: List of previously rejected triplets
            temperature: LLM sampling temperature
            
        Returns:
            List of new proposed triplets
        """
        from .edc.edc.utils.llm_utils import openai_chat_completion
        
        system_prompt = self._get_reproposal_prompt()
        
        # Format previous triplets
        prev_triplets_text = "\n".join(
            f"- ({t[0]}, {t[1]}, {t[2]})" if len(t) >= 3 else f"- {t}"
            for t in previous_triplets
        )
        
        # Format missing information
        missing_text = "\n".join(f"- {m}" for m in missing_information)
        
        user_message = f"""## Rejection Feedback
{rejection_feedback}

## Original Question
{query}

## Missing Information (still needed)
{missing_text}

## Previously Rejected Triplets
{prev_triplets_text}

Please propose NEW, DIFFERENT triplets that address the missing information while avoiding the issues that caused rejection."""
        
        history = [{"role": "user", "content": user_message}]
        
        response = openai_chat_completion(
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=1024,
        )
        
        # Parse the response
        try:
            # Try to extract JSON
            json_patterns = [
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
                r'(\{[\s\S]*"proposed_triplets"[\s\S]*\})',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    break
            
            if not json_str:
                json_str = response.strip()
            
            data = json.loads(json_str)
            raw_triplets = data.get("proposed_triplets", [])
            
            # Validate and normalize triplets
            new_triplets = []
            for triplet in raw_triplets:
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    new_triplets.append([str(t) for t in triplet[:3]])
            
            return new_triplets if new_triplets else previous_triplets
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse reproposal response: {e}")
            return previous_triplets
    
    def _justify_persistence(
        self,
        query: str,
        answer: str,
        validated_triplets: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Ask the local LLM to evaluate whether each validated triplet should be persisted.
        
        The LLM decides per-triplet whether it contains specific, useful knowledge
        worth adding to the KG, or whether it is too vague/generic to be worth persisting.
        
        Args:
            query: The user's original question
            answer: The answer that was generated
            validated_triplets: List of validated triplet dicts from remote LLM
            temperature: LLM sampling temperature
            
        Returns:
            Dict with "persist" and "skip" lists, each containing
            {"triplet": str, "reason": str} entries.
        """
        from .edc.edc.utils.llm_utils import openai_chat_completion
        
        system_prompt = load_prompt("persistence_justification")
        
        triplets_text = "\n".join(
            f"- {vt.get('triplet', '')}" for vt in validated_triplets
        )
        
        user_message = f"""## Question
{query}

## Answer Produced
{answer}

## Validated Triplets to Evaluate
{triplets_text}

Decide which triplets should be persisted to the knowledge graph. Respond with JSON."""
        
        history = [{"role": "user", "content": user_message}]
        
        response = openai_chat_completion(
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=1024,
        )
        
        # Parse the response
        default_result = {
            "persist": [{"triplet": vt.get("triplet", ""), "reason": "Fallback: persisting all (parse failure)"} for vt in validated_triplets],
            "skip": [],
        }
        
        try:
            json_patterns = [
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
                r'(\{[\s\S]*"persist"[\s\S]*\})',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    break
            
            if not json_str:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            persist = data.get("persist", [])
            skip = data.get("skip", [])
            
            # Validate structure
            valid_persist = []
            for item in persist:
                if isinstance(item, dict) and "triplet" in item:
                    valid_persist.append({
                        "triplet": item["triplet"],
                        "reason": item.get("reason", "Accepted for persistence"),
                    })
            
            valid_skip = []
            for item in skip:
                if isinstance(item, dict) and "triplet" in item:
                    valid_skip.append({
                        "triplet": item["triplet"],
                        "reason": item.get("reason", "Skipped"),
                    })
            
            # If the LLM returned empty persist and skip, fall back to persisting all
            if not valid_persist and not valid_skip:
                return default_result
            
            return {"persist": valid_persist, "skip": valid_skip}
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse persistence justification response: {e}")
            logger.debug(f"Raw response: {response}")
            return default_result
    
    def _validate_with_retry(
        self,
        query: str,
        proposed_triplets: List[List[str]],
        missing_information: List[str],
        existing_facts: List[str],
        max_retries: int = 3,
        temperature: float = 0.1,
        verbose: bool = False,
        intermediate_messages: List[str] = None,
        knowledge_iteration: int = 1,
    ) -> Tuple[List[Dict], bool, str, List[Dict], str]:
        """
        Validate triplets with remote LLM, retrying up to max_retries if all rejected.
        
        Args:
            query: The user's original query
            proposed_triplets: List of proposed triplets to validate
            missing_information: List of missing information items
            existing_facts: List of existing fact strings
            max_retries: Maximum number of validation attempts
            temperature: LLM sampling temperature for reproposals
            verbose: Whether to print progress messages
            intermediate_messages: List to collect intermediate messages
            knowledge_iteration: Which knowledge iteration this validation belongs to
            
        Returns:
            Tuple of (validated_triplets, success, failure_message, validation_history, remote_model_name)
        """
        if intermediate_messages is None:
            intermediate_messages = []
        
        current_triplets = proposed_triplets
        validation_history = []
        remote_model_name = "unknown"
        
        for attempt in range(1, max_retries + 1):
            # Format triplets as strings for validation
            triplet_strings = [
                f"({t[0]}, {t[1]}, {t[2]})" if len(t) >= 3 else str(t)
                for t in current_triplets
            ]
            
            # Call remote LLM for validation via tool handler
            validate_result = self.tool_handler.handle_tool_call(
                "validate_and_persist_triplets",
                {
                    "query": query,
                    "proposed_triplets": triplet_strings,
                    "existing_facts": existing_facts,
                    "persist_validated": False,  # Don't persist yet - we may retry
                }
            )
            validate_data = json.loads(validate_result)
            
            validated = validate_data.get("validated_triplets", [])
            rejected = validate_data.get("rejected_triplets", [])
            all_rejected = len(validated) == 0 and len(rejected) > 0
            remote_model_name = validate_data.get("remote_model", "unknown")
            
            # Generate validation attempt message
            attempt_message = self._generate_validation_attempt_message(
                attempt=attempt,
                max_attempts=max_retries,
                validated=validated,
                rejected=rejected,
                remote_model=remote_model_name,
            )
            intermediate_messages.append(attempt_message)
            if verbose:
                print(attempt_message)
            
            # Record in history
            validation_history.append({
                "knowledge_iteration": knowledge_iteration,
                "attempt": attempt,
                "proposed": current_triplets,
                "validated": validated,
                "rejected": rejected,
                "all_rejected": all_rejected,
                "remote_model": remote_model_name,
            })
            
            # Check if any triplets were accepted
            if validated:
                return validated, True, "", validation_history, remote_model_name
            
            # All rejected - get feedback for retry
            if attempt < max_retries:
                rejection_feedback = self._get_rejection_feedback(rejected)
                
                # Ask local LLM to re-propose with feedback
                current_triplets = self._repropose_triplets(
                    query=query,
                    missing_information=missing_information,
                    rejection_feedback=rejection_feedback,
                    previous_triplets=current_triplets,
                    temperature=temperature,
                )
                
                # Generate retry proposal message
                retry_message = self._generate_retry_message(
                    attempt=attempt,
                    max_attempts=max_retries,
                    new_triplets=current_triplets,
                )
                intermediate_messages.append(retry_message)
                if verbose:
                    print(retry_message)
        
        # All retries exhausted
        failure_message = self._generate_failure_message(query, max_retries)
        intermediate_messages.append(failure_message)
        if verbose:
            print(failure_message)
        
        return [], False, failure_message, validation_history, remote_model_name


def create_mcp_agent_with_validation(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = None,
    neo4j_database: str = "neo4j",
    embedding_model: str = None,
    max_iterations: int = 5,
    allow_cypher: bool = False,
    enable_validation: bool = True,
    auto_expand: bool = True,
    max_validation_retries: int = 3,
    max_knowledge_iterations: int = 3,
) -> MCPAgentWithValidation:
    """
    Convenience function to create an MCPAgentWithValidation.
    
    This creates an agent that:
    1. Queries the knowledge graph
    2. Uses LLM to assess if facts are sufficient (with complete triplet proposals)
    3. Validates proposed triplets with remote LLM (with retry mechanism)
    4. Persists validated facts
    5. Re-queries and re-assesses iteratively until sufficient or max iterations
    6. Provides detailed intermediate messages throughout the process
    
    Required environment variables:
    - For local LLM: source export_local_llm.sh (or USE_LOCAL_LLM=true)
    - For remote validation: source export_google_ai.sh (or REMOTE_LLM_* vars)
    
    Args:
        neo4j_uri: Neo4j Bolt URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        embedding_model: Embedding model name
        max_iterations: Maximum tool call iterations
        allow_cypher: Whether to allow raw Cypher queries
        enable_validation: Whether to enable remote validation
        auto_expand: Whether to auto-expand when gaps detected
        max_validation_retries: Maximum retries for validation (default: 3)
        max_knowledge_iterations: Maximum search-assess-expand cycles (default: 3)
        
    Returns:
        Configured MCPAgentWithValidation instance
    """
    from .neo4j_store import Neo4jStore
    from .embedder import get_embedder
    
    # Get configuration from environment or use defaults
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    if embedding_model is None:
        embedding_model = os.environ.get(
            "LOCAL_EMBEDDER_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
    
    local_model_name = os.environ.get("LOCAL_LLM_MODEL", "local_llm")
    
    logger.info(f"Creating MCPAgentWithValidation:")
    logger.info(f"  Neo4j: {neo4j_uri}")
    logger.info(f"  Embedding model: {embedding_model}")
    logger.info(f"  Validation: {enable_validation}")
    logger.info(f"  Auto-expand: {auto_expand}")
    logger.info(f"  Max validation retries: {max_validation_retries}")
    logger.info(f"  Max knowledge iterations: {max_knowledge_iterations}")
    
    # Create Neo4j store
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    
    # Create embedder
    embedder = get_embedder(model_name=embedding_model)
    
    # Create agent
    agent = MCPAgentWithValidation(
        neo4j_store=neo4j_store,
        embedder=embedder,
        max_iterations=max_iterations,
        allow_cypher=allow_cypher,
        enable_validation=enable_validation,
        auto_expand=auto_expand,
        max_validation_retries=max_validation_retries,
        max_knowledge_iterations=max_knowledge_iterations,
        local_model_name=local_model_name,
    )
    
    return agent


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test MCP Agent")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-password", default="password123")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("Creating MCP Agent...")
    agent = create_mcp_agent(
        neo4j_uri=args.neo4j_uri,
        neo4j_password=args.neo4j_password,
    )
    
    print(f"\nQuery: {args.query}")
    print("=" * 60)
    
    result = agent.run(args.query, verbose=args.verbose)
    
    print("\n=== Result ===")
    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {len(result.tool_calls)}")
    
    for tc in result.tool_calls:
        print(f"  - {tc['tool']}({tc['arguments']})")
    
    if result.warning:
        print(f"Warning: {result.warning}")
