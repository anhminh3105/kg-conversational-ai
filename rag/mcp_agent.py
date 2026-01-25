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
        
        return f"""You are a helpful knowledge graph assistant. You have access to tools to query a knowledge graph database.

## Available Tools

{tools_json}

## Instructions

1. When the user asks a question, decide if you need to query the knowledge graph.
2. To call a tool, respond with a JSON object in this exact format:
   {{"tool": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
3. After receiving tool results, either:
   - Call another tool if you need more information
   - Provide a final answer based on the gathered information
4. When providing a final answer, respond naturally without JSON - just answer the question directly.

## Tips

- Use `search_knowledge_graph` when you need to find information but don't know exact entity names
- Use `query_entity` when you know the specific entity you want to learn about
- Use `expand_entity` to discover related information and connections
- Always base your answers on the facts retrieved from the knowledge graph
- If no relevant facts are found, say so honestly

Respond concisely and accurately based on the knowledge graph data."""
    
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
        
        return f"""You are a knowledge graph assistant with access to tools.

{tools_text}

To call a tool, respond with JSON: {{"tool": "tool_name", "arguments": {{...}}}}
After receiving results, provide a clear answer based on the facts found.
If you don't need a tool, answer directly."""
    
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
