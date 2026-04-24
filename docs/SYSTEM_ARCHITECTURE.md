# System Architecture: KG-Conversational-AI

Technical reference documentation for the Knowledge Graph RAG system with Dual-LLM Validation.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Structure](#2-module-structure)
3. [Core Classes](#3-core-classes)
4. [Data Structures](#4-data-structures)
5. [Processing Pipeline](#5-processing-pipeline)
6. [Tool Interface (MCP)](#6-tool-interface-mcp)
7. [LLM Integration](#7-llm-integration)
8. [Storage Layer](#8-storage-layer)
9. [Configuration Reference](#9-configuration-reference)
10. [API Reference](#10-api-reference)
11. [PrimeKG Integration](#11-primekg-integration)
12. [Prompt System Details](#12-prompt-system-details)
13. [Interactive Agent Implementation](#13-interactive-agent-implementation)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                   │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        MCPAgentWithValidation                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    ITERATIVE EXPANSION LOOP                        │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │  │
│  │  │  SEARCH  │──▶│  ASSESS  │──▶│ PROPOSE  │──▶│ VALIDATE │        │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │  │
│  │       │              │              │              │               │  │
│  │       ▼              ▼              ▼              ▼               │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │  │
│  │  │  Neo4j   │   │  Local   │   │  Local   │   │  Remote  │        │  │
│  │  │  +FAISS  │   │   LLM    │   │   LLM    │   │   LLM    │        │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │  │
│  │                                                     │              │  │
│  │                                                     ▼              │  │
│  │                                               ┌──────────┐         │  │
│  │                                               │ HOLD IN  │         │  │
│  │                                               │ MEMORY   │         │  │
│  │                                               └──────────┘         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│                           ┌──────────────────┐                           │
│                           │  GENERATE ANSWER │  (uses KG + validated)    │
│                           └────────┬─────────┘                           │
│                                    ▼                                      │
│                           ┌──────────────────┐                           │
│                           │  JUSTIFY         │  (local LLM quality gate) │
│                           │  PERSISTENCE     │                           │
│                           └────────┬─────────┘                           │
│                                    ▼                                      │
│                           ┌──────────────────┐                           │
│                           │  PERSIST TO      │  (only justified facts)   │
│                           │  NEO4J           │                           │
│                           └──────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         ValidatedAgentResult                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Knowledge Store | Neo4j + FAISS | Triplet storage with vector search |
| Local LLM | Qwen2.5-7B-Instruct | Query processing, gap detection, proposals |
| Remote LLM | Gemini/GPT-4/Claude | Factual validation |
| Embedder | BGE-small-en-v1.5 | Semantic similarity |
| MCP Handler | Custom | Tool interface for LLM |

---

## 2. Module Structure

```
rag/
├── mcp_agent.py              # Agent classes (MCPAgent, MCPAgentWithValidation)
│   ├── AgentResult           # Base result dataclass
│   ├── ValidatedAgentResult  # Extended result with validation info
│   ├── MCPAgent              # Basic agent with tool calling
│   └── MCPAgentWithValidation # Full agent with dual-LLM validation
│
├── mcp_neo4j_server.py       # MCP tool handler
│   ├── NEO4J_TOOLS           # Tool definitions
│   └── Neo4jMCPToolHandler   # Tool execution
│
├── neo4j_store.py            # Neo4j backend
│   ├── Neo4jStore            # Graph + vector storage
│   └── SearchResult          # Search result dataclass
│
├── faiss_store.py            # FAISS backend (alternative)
│   └── FaissStore            # Vector-only storage
│
├── embedder.py               # Embedding models
│   ├── Embedder              # Base class
│   └── get_embedder()        # Factory function
│
├── triplet_validator.py      # Remote LLM validation
│   ├── ValidatedTriplet      # Validation result
│   ├── ValidationResult      # Batch validation result
│   └── TripletValidator      # Validator class
│
├── triplet_expander.py       # Local LLM expansion
│   └── TripletExpander       # Expander class
│
├── knowledge_gap_detector.py # (Legacy) Heuristic detection
│   └── KnowledgeGapDetector  # Deprecated, replaced by LLM assessment
│
└── prompts/                  # External prompt templates
    ├── knowledge_assessment.txt
    ├── answer_generation.txt
    ├── answer_generation_with_new_facts.txt
    ├── persistence_justification.txt
    ├── triplet_validation.txt
    └── triplet_reproposal.txt
```

---

## 3. Core Classes

### 3.1 MCPAgentWithValidation

The main orchestrator for the dual-LLM knowledge expansion workflow.

```python
class MCPAgentWithValidation:
    """
    Agent with iterative knowledge expansion using dual-LLM validation.
    """
    
    def __init__(
        self,
        neo4j_store,                    # Neo4jStore instance
        embedder,                       # Embedder instance
        max_iterations: int = 5,        # Max tool call iterations
        allow_cypher: bool = False,     # Allow raw Cypher queries
        enable_validation: bool = True, # Enable remote validation
        auto_expand: bool = True,       # Auto-expand on knowledge gap
        max_validation_retries: int = 3,     # Retries per validation
        max_knowledge_iterations: int = 3,   # Search-assess-expand cycles
        local_model_name: str = "local_llm", # Local LLM identifier
    ):
        ...
    
    def run(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        verbose: bool = False,
        force_expand: bool = False,
    ) -> ValidatedAgentResult:
        """Execute query with iterative knowledge expansion."""
        ...
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `run()` | Main entry point - executes iterative workflow with deferred persistence |
| `_assess_knowledge_sufficiency()` | LLM-based gap detection |
| `_validate_with_retry()` | Validation with retry mechanism (tracks knowledge iteration) |
| `_justify_persistence()` | Local LLM evaluates which validated triplets to persist |
| `_generate_gap_detection_message()` | User-facing gap message |
| `_generate_iteration_summary_message()` | Iteration progress message |
| `_generate_persistence_message()` | Persistence confirmation |
| `_repropose_triplets()` | Generate new proposals after rejection |
| `_get_answer_prompt()` | Select answer prompt (with remote model name + validated count) |

### 3.2 Neo4jMCPToolHandler

Handles MCP tool calls and interfaces with Neo4j.

```python
class Neo4jMCPToolHandler:
    """
    MCP tool handler with Neo4j backend and validation support.
    """
    
    def __init__(
        self,
        neo4j_store,
        embedder,
        allow_cypher: bool = False,
        enable_expansion: bool = True,
        enable_validation: bool = True,
        local_model_name: str = "local_llm",
    ):
        ...
    
    def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Execute a tool call and return JSON result."""
        ...
```

**Available Tools:**

| Tool | Method | Description |
|------|--------|-------------|
| `search_knowledge_graph` | `_search_kg()` | Semantic vector search |
| `query_entity` | `_query_entity()` | Find facts about entity |
| `expand_entity` | `_expand_entity()` | N-hop graph traversal |
| `expand_triplets` | `_expand_triplets()` | Generate new triplets |
| `validate_and_persist_triplets` | `_validate_and_persist_triplets()` | Validate and store |
| `run_cypher` | `_run_cypher()` | Raw Cypher (if enabled) |

### 3.3 TripletValidator

Validates proposed triplets using remote LLM.

```python
class TripletValidator:
    """
    Validates triplets using remote LLM API.
    """
    
    def validate(
        self,
        proposed_triplets: List[Tuple[str, str, str]],
        query: str,
        existing_facts: List[str],
    ) -> ValidationResult:
        """
        Validate triplets with remote LLM.
        
        Returns ValidationResult with:
        - validated: List of accepted triplets
        - rejected: List of rejected triplets
        - remote_model: Model used for validation
        """
        ...
```

### 3.4 Neo4jStore

Hybrid graph + vector storage using Neo4j.

```python
class Neo4jStore:
    """
    Neo4j backend with vector index support.
    """
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Vector similarity search."""
        ...
    
    def graph_search(
        self,
        entity: str,
        relationship: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict]:
        """Graph traversal search."""
        ...
    
    def add(
        self,
        embeddings: np.ndarray,
        items: List[EmbeddableItem],
    ) -> int:
        """Add triplets with embeddings."""
        ...
```

---

## 4. Data Structures

### 4.1 ValidatedAgentResult

Complete result from agent execution.

```python
@dataclass
class ValidatedAgentResult(AgentResult):
    # Inherited from AgentResult
    answer: str                              # Generated answer
    query: str                               # Original query
    tool_calls: List[Dict[str, Any]]         # Tool call log
    iterations: int                          # Total tool iterations
    reasoning: Optional[str]                 # Reasoning trace
    warning: Optional[str]                   # Warning message
    
    # Validation-specific fields
    knowledge_gap_detected: bool             # Gap was detected
    proposed_triplets: List[str]             # All proposed triplets
    validated_triplets: List[Dict[str, Any]] # Accepted triplets
    rejected_triplets: List[Dict[str, Any]]  # Rejected triplets
    persisted_count: int                     # Triplets added to Neo4j
    new_facts_notification: str              # User notification
    intermediate_messages: List[str]         # Progress messages
    llm_assessment: Dict[str, Any]           # Last assessment details
    validation_failed: bool                  # All validations failed
    validation_attempts: int                 # Total validation calls
    validation_history: List[Dict[str, Any]] # History of attempts
    knowledge_iterations: int                # Expansion cycles completed
    
    # Remote model tracking
    remote_model_name: str                   # Which frontier model validated
    
    # Persistence justification (deferred persistence)
    persistence_justification: Dict[str, Any]  # {persist: [...], skip: [...]}
    skipped_triplets: List[Dict[str, Any]]     # Triplets skipped by local LLM
```

### 4.2 LLM Assessment Response

Structure returned by `_assess_knowledge_sufficiency()`.

```python
{
    "assessment": "SUFFICIENT" | "INSUFFICIENT",
    "confidence": float,  # 0.0 - 1.0
    "missing_information": [
        "What is the first missing piece?",
        "What is the second missing piece?",
    ],
    "proposed_triplets": [
        ["Subject1", "predicate1", "Object1"],
        ["Subject2", "predicate2", "Object2"],
    ],
    "answer": "Partial answer based on available facts..."
}
```

### 4.3 Validation History Entry

Structure for each validation attempt.

```python
{
    "knowledge_iteration": int,  # Which expansion cycle (1-indexed)
    "attempt": int,              # Attempt number within cycle (1-indexed)
    "all_rejected": bool,        # All triplets rejected
    "remote_model": str,         # Remote model used (e.g., "gemini-2.5-flash-lite")
    "validated": [
        {
            "triplet": "(S, P, O)",
            "status": "validated",  # or "corrected"
            "reason": "Explanation..."
        }
    ],
    "rejected": [
        {
            "triplet": "(S, P, O)",
            "status": "rejected",
            "reason": "Why rejected..."
        }
    ]
}
```

### 4.4 Persistence Justification Response

Structure returned by `_justify_persistence()`.

```python
{
    "persist": [
        {
            "triplet": "(Einstein, won, Nobel_Prize_Physics)",
            "reason": "Contains specific named entity and factual claim"
        }
    ],
    "skip": [
        {
            "triplet": "(AMD, hasStakeholder, Stakeholder_Entity)",
            "reason": "Uses generic placeholder value"
        }
    ]
}
```

### 4.5 Triplet Formats

```python
# Internal representation (list/tuple)
triplet_internal = ["Nie_Haisheng", "attendedSchool", "Zaoyang_Middle_School"]

# String representation
triplet_string = "(Nie_Haisheng, attendedSchool, Zaoyang_Middle_School)"

# Normalized for deduplication
triplet_normalized = "niehaisheng|attendedschool|zaoyangmiddleschool"

# Neo4j node properties
triplet_node = {
    "subject": "Nie_Haisheng",
    "predicate": "attendedSchool",
    "object": "Zaoyang_Middle_School",
    "document": "(Nie_Haisheng, attendedSchool, Zaoyang_Middle_School)",
    "embedding": [0.123, 0.456, ...],  # 384-dim vector
    "source": "remote_validated",
    "validation_status": "validated",
    "validation_reason": "Matches known facts",
    "validated_at": "2026-02-02T01:23:45",
    "local_llm": "Qwen/Qwen2.5-7B-Instruct",
    "remote_llm": "gemini-2.5-flash-lite"
}
```

---

## 5. Processing Pipeline

### 5.1 Iterative Expansion Algorithm

```python
def run(query):
    all_known_facts = set()
    all_validated = []
    all_rejected = []
    existing_fact_strings = []
    
    for iteration in range(1, max_knowledge_iterations + 1):
        # Step 1: Search knowledge graph
        facts = search_knowledge_graph(query, top_k=10)
        all_known_facts.update(normalize(facts))
        existing_fact_strings = list(facts)
        
        # Step 2: Assess sufficiency with local LLM
        # (includes validated facts from prior iterations in existing_fact_strings)
        assessment = assess_knowledge_sufficiency(query, existing_fact_strings)
        
        # Step 3: Check if sufficient
        if assessment.verdict == "SUFFICIENT":
            break
        
        # Step 4: Get and deduplicate proposals
        proposed = assessment.proposed_triplets
        proposed = filter(lambda t: normalize(t) not in all_known_facts, proposed)
        
        if not proposed:
            break  # No new proposals
        
        # Step 5: Validate with remote LLM (with retry)
        validated, success = validate_with_retry(proposed, knowledge_iteration=iteration)
        
        if not success:
            break  # Validation failed
        
        # Step 6: Hold validated triplets in memory (do NOT persist yet)
        all_validated.extend(validated)
        all_known_facts.update(normalize(validated))
        existing_fact_strings.extend(validated)  # Next iteration sees them
        
        if not validated:
            break  # No progress
    
    # Step 7: Generate final answer using KG facts + all validated (in memory)
    answer = generate_answer(query, existing_fact_strings, all_validated)
    
    # Step 8: Persistence justification - local LLM decides what to persist
    decision = justify_persistence(query, answer, all_validated)
    
    # Step 9: Persist only justified triplets
    persist_to_neo4j(decision.persist)
    
    return ValidatedAgentResult(
        answer=answer,
        persisted_count=len(decision.persist),
        skipped_triplets=decision.skip,
        persistence_justification=decision,
        ...
    )
```

### 5.2 Validation Retry Algorithm

```python
def validate_with_retry(proposed, max_retries=3, knowledge_iteration=1):
    validation_history = []
    
    for attempt in range(1, max_retries + 1):
        # Send to remote LLM
        result = remote_llm.validate(proposed)
        
        # Record in history with both iteration and attempt
        validation_history.append({
            "knowledge_iteration": knowledge_iteration,
            "attempt": attempt,
            "validated": result.validated,
            "rejected": result.rejected,
            "remote_model": result.remote_model,
        })
        
        if result.validated:  # At least one accepted
            return result.validated, True
        
        if attempt < max_retries:
            # Extract rejection feedback
            feedback = extract_feedback(result.rejected)
            
            # Ask local LLM to repropose
            proposed = local_llm.repropose(feedback)
            
            if not proposed:
                break  # No new proposals
    
    return [], False
```

### 5.3 Deduplication Logic

```python
def normalize_triplet(triplet):
    """Normalize for comparison."""
    if isinstance(triplet, (list, tuple)):
        s = f"{triplet[0]}|{triplet[1]}|{triplet[2]}"
    else:
        s = triplet
    return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "").replace(",", "|")

def deduplicate(proposed, known_facts):
    """Filter out duplicates."""
    known_normalized = {normalize_triplet(f) for f in known_facts}
    unique = []
    for t in proposed:
        if normalize_triplet(t) not in known_normalized:
            unique.append(t)
    return unique
```

---

## 6. Tool Interface (MCP)

### 6.1 Tool Definitions

```python
NEO4J_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": "Semantic search for facts related to a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_and_persist_triplets",
            "description": "Validate and persist triplets to knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "proposed_triplets": {"type": "array", "items": {"type": "string"}},
                    "existing_facts": {"type": "array", "items": {"type": "string"}},
                    "persist_validated": {"type": "boolean", "default": True},
                    "skip_validation": {"type": "boolean", "default": False}
                },
                "required": ["query", "proposed_triplets"]
            }
        }
    }
    # ... other tools
]
```

### 6.2 Tool Call Flow

```
LLM generates JSON:
{
    "tool": "search_knowledge_graph",
    "arguments": {"query": "Einstein awards", "top_k": 10}
}
        │
        ▼
Neo4jMCPToolHandler.handle_tool_call()
        │
        ▼
_search_kg(query, top_k)
        │
        ▼
Neo4jStore.search() → FAISS similarity
        │
        ▼
JSON result:
{
    "query": "Einstein awards",
    "num_results": 5,
    "facts": [
        {"fact": "(Einstein, won, Nobel_Prize)", "score": 0.89},
        ...
    ]
}
```

---

## 7. LLM Integration

### 7.1 Local LLM Configuration

```bash
# Environment variables
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LOCAL_LLM_QUANTIZATION="4bit"
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"
```

### 7.2 Remote LLM Configuration

```bash
# Google AI
export REMOTE_LLM_API_KEY="your-api-key"
export REMOTE_LLM_MODEL="gemini-2.5-flash-lite"
export REMOTE_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

# OpenAI
export REMOTE_LLM_API_KEY="sk-..."
export REMOTE_LLM_MODEL="gpt-4o-mini"
export REMOTE_LLM_BASE_URL="https://api.openai.com/v1"

# SambaNova
export REMOTE_LLM_API_KEY="your-key"
export REMOTE_LLM_MODEL="Meta-Llama-3.1-70B-Instruct"
export REMOTE_LLM_BASE_URL="https://api.sambanova.ai/v1"
```

### 7.3 LLM Utility Functions

```python
# rag/edc/edc/utils/llm_utils.py

def openai_chat_completion(
    system_prompt: str,
    history: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Generate completion using local or remote LLM."""
    ...

def is_remote_llm_configured() -> bool:
    """Check if remote LLM is configured."""
    ...

def get_remote_model_name() -> str:
    """Get configured remote model name."""
    ...
```

---

## 8. Storage Layer

The system supports two storage backends, selectable via the `--store_type` CLI flag in `scripts/index_rag.py`:

| Backend | Flag | Description |
|---------|------|-------------|
| **FAISS** | `--store_type faiss` (default) | Fast in-memory vector similarity search. Index files saved locally (`.faiss`, `_meta.json`). No external services required. |
| **Neo4j** | `--store_type neo4j` | Graph database with vector index support. Provides both vector similarity search and native graph traversal. Requires a running Neo4j instance. Connection resolved from `--neo4j_uri`/`--neo4j_password` flags or `NEO4J_URI`/`NEO4J_PASSWORD` environment variables. |

Both backends implement the same `add()` / `search()` / `save()` / `load()` interface via `FaissStore` and `Neo4jStore`, so all downstream components (retriever, generator, expander) work identically regardless of the chosen backend.

### 8.1 Neo4j Schema

```cypher
// Unique constraint
CREATE CONSTRAINT triplet_unique IF NOT EXISTS
FOR (t:Triplet) REQUIRE (t.subject, t.predicate, t.object) IS UNIQUE

// Indexes for fast lookup
CREATE INDEX triplet_subject IF NOT EXISTS FOR (t:Triplet) ON (t.subject)
CREATE INDEX triplet_object IF NOT EXISTS FOR (t:Triplet) ON (t.object)

// Vector index for semantic search
CREATE VECTOR INDEX triplet_embedding IF NOT EXISTS
FOR (t:Triplet) ON t.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
}
```

### 8.2 Triplet Node Properties

| Property | Type | Description |
|----------|------|-------------|
| `subject` | string | Subject entity |
| `predicate` | string | Relationship type |
| `object` | string | Object entity/value |
| `document` | string | Full triplet string |
| `embedding` | float[384] | Vector representation |
| `source` | string | "original", "llm_expanded", "remote_validated" |
| `source_text` | string | Original source text (if any) |
| `representation_mode` | string | "triplet_text" |
| `validation_status` | string | "validated", "corrected" |
| `validation_reason` | string | Why accepted/corrected |
| `validated_at` | datetime | Validation timestamp |
| `local_llm` | string | Local model name |
| `remote_llm` | string | Remote model name |
| `original_proposal` | string | Original if corrected |

### 8.3 Vector Search Query

```cypher
CALL db.index.vector.queryNodes(
    'triplet_embedding',
    $top_k,
    $query_embedding
)
YIELD node, score
RETURN 
    node.subject AS subject,
    node.predicate AS predicate,
    node.object AS object,
    score
ORDER BY score DESC
```

---

## 9. Configuration Reference

### 9.1 MCPAgentWithValidation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neo4j_store` | Neo4jStore | required | Neo4j store instance |
| `embedder` | Embedder | required | Embedding model |
| `max_iterations` | int | 5 | Max tool call iterations |
| `allow_cypher` | bool | False | Allow raw Cypher |
| `enable_validation` | bool | True | Enable remote validation |
| `auto_expand` | bool | True | Auto-expand on gap |
| `max_validation_retries` | int | 3 | Retries per validation |
| `max_knowledge_iterations` | int | 3 | Expansion cycles |
| `local_model_name` | str | "local_llm" | Local LLM identifier |

### 9.2 run() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | User question |
| `temperature` | float | 0.1 | LLM temperature |
| `max_tokens` | int | 512 | Max tokens per call |
| `verbose` | bool | False | Print progress |
| `force_expand` | bool | False | Force expansion |

### 9.3 Environment Variables

| Variable | Description |
|----------|-------------|
| `USE_LOCAL_LLM` | Enable local LLM mode |
| `LOCAL_LLM_MODEL` | Local model name |
| `LOCAL_LLM_QUANTIZATION` | Quantization level |
| `LOCAL_EMBEDDER_MODEL` | Embedding model |
| `REMOTE_LLM_API_KEY` | Remote API key |
| `REMOTE_LLM_MODEL` | Remote model name |
| `REMOTE_LLM_BASE_URL` | Remote API URL |
| `NEO4J_PASSWORD` | Neo4j password |

---

## 10. API Reference

### 10.1 Quick Start

```python
from rag.mcp_agent import create_mcp_agent_with_validation

# Create agent
agent = create_mcp_agent_with_validation(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password123",
    enable_validation=True,
    max_knowledge_iterations=3,
)

# Execute query
result = agent.run("What schools did Einstein attend?")

# Access results
print(result.answer)
print(f"Iterations: {result.knowledge_iterations}")
print(f"Validated: {len(result.validated_triplets)}")
print(f"Persisted: {result.persisted_count}")
```

### 10.2 Verbose Execution

```python
result = agent.run(
    "What awards did Einstein win?",
    verbose=True,
    force_expand=True,
)

# View intermediate messages
for msg in result.intermediate_messages:
    print(msg)
```

### 10.3 Direct Tool Access

```python
from rag.mcp_neo4j_server import Neo4jMCPToolHandler
from rag.neo4j_store import Neo4jStore
from rag.embedder import get_embedder

# Create components
store = Neo4jStore(uri="bolt://localhost:7687", password="password123")
embedder = get_embedder("BAAI/bge-small-en-v1.5")
handler = Neo4jMCPToolHandler(store, embedder, enable_validation=True)

# Call tools directly
result = handler.handle_tool_call(
    "search_knowledge_graph",
    {"query": "Einstein physics", "top_k": 5}
)
print(result)
```

---

## Appendix: Message Formats

### Gap Detection Message

```
==================================================
KNOWLEDGE GAP DETECTED
==================================================
The knowledge graph does not contain sufficient information to answer your question.
I will propose 3 fact(s) from my own understanding for external validation.

What's missing:
  - What primary school did Einstein attend?

Proposed facts (3) to fill these gaps:
  1. (Einstein, primarySchool, Luitpold_Gymnasium)
  2. (Einstein, bornIn, Ulm)
  3. (Einstein, invented, telephone)

Next action: Sending 3 proposed facts to remote LLM for validation...
==================================================
```

### Validation Attempt Message

```
==================================================
VALIDATION ATTEMPT 1/3 (by gemini-2.5-flash-lite)
==================================================
Sending 3 proposed triplets to gemini-2.5-flash-lite for validation...

Response from gemini-2.5-flash-lite:
  ACCEPTED:
    - (Einstein, primarySchool, Luitpold_Gymnasium) ✓
      Reason: Matches known historical records.
    - (Einstein, bornIn, Ulm) ✓
      Reason: Matches known historical records.
  REJECTED:
    - (Einstein, invented, telephone) ✗
      Reason: Factually incorrect. Alexander Graham Bell invented the telephone.

Result: 2 accepted, 1 rejected

Next action: Using validated facts to generate answer (persistence deferred until after answer)...
==================================================
```

### Iteration Summary Message

```
==================================================
KNOWLEDGE ITERATION 1/3 COMPLETE
==================================================
Assessment: INSUFFICIENT
Validated: 2 triplets
Rejected: 1 triplets
Persistence: deferred until after answer

Status: Knowledge still insufficient
Next action: Re-assessing with validated facts included...
==================================================
```

### Persistence Justification Message

Displayed after the final answer is generated.

```
──────────────────────────────────────────────────
Persistence Justification (Local LLM)
──────────────────────────────────────────────────

  Approved for persistence (2):
    + (Einstein, primarySchool, Luitpold_Gymnasium)
      Contains specific named entity, helps answer similar questions
    + (Einstein, bornIn, Ulm)
      Contains specific location, factual and useful

  Skipped (not persisted) (1):
    - (AMD, hasStakeholder, Stakeholder_Entity)
      Uses generic placeholder value, not informative

  Persisted to Neo4j: 2 triplet(s)
```

---

## 11. PrimeKG Integration

### 11.1 Data Pipeline

```
Harvard Dataverse
      │
      │  download_primekb.py
      ▼
┌─────────────┐
│   kg.csv    │  (~580 MB, 8.1M rows)
└──────┬──────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌─────────────────┐           ┌─────────────────┐
│  PATH A         │           │  PATH B         │
│  Triplet Schema │           │  Native Schema  │
│                 │           │                 │
│  PrimeKBLoader  │           │  import_primekb │
│       ↓         │           │  _to_neo4j.py   │
│  :Triplet nodes │           │       ↓         │
│  (subject,      │           │  :Drug, :Disease│
│   pred, object) │           │  :Gene nodes    │
└─────────────────┘           └─────────────────┘
```

### 11.2 PrimeKBLoader Class

```python
class PrimeKBLoader:
    """
    Loads PrimeKG CSV and converts to Triplet objects.
    
    Args:
        path: Path to kg.csv
        node_types: Filter by x_type/y_type (e.g., ["drug", "disease"])
        relation_types: Filter by display_relation (e.g., ["treats"])
        max_rows: Limit CSV rows (for testing)
    """
```

### 11.3 PrimeKG CSV to Triplet Mapping

| CSV Column | Triplet Property |
|------------|------------------|
| `x_name` | `subject` |
| `display_relation` | `predicate` |
| `y_name` | `object` |
| `x_type` | metadata |
| `y_type` | metadata |
| `x_id`, `y_id` | metadata |

### 11.4 Native Import Schema

```cypher
// Node creation
MERGE (n:{NodeType} {primekb_id: $x_id})
SET n.name = $x_name, n.source = $x_source

// Relationship creation
MATCH (a {primekb_id: $x_id}), (b {primekb_id: $y_id})
MERGE (a)-[r:{REL_TYPE}]->(b)
```

### 11.5 Schema Auto-Detection

```python
def detect_schema(conn):
    """Auto-detect whether database uses :Triplet or native PrimeKG schema."""
    # Check for :Triplet nodes
    result = conn.query("MATCH (t:Triplet) RETURN count(t) AS cnt")
    if result and result[0]["cnt"] > 0:
        return "triplet"
    
    # Check for native PrimeKG nodes
    result = conn.query(
        "MATCH (n) WHERE n.primekb_id IS NOT NULL RETURN count(n) AS cnt"
    )
    if result and result[0]["cnt"] > 0:
        return "primekb"
    
    return "triplet"  # default
```

### 11.6 PrimeKB Predicate Detection

```python
PRIMEKB_PREDICATES = {
    "treats", "associated_with", "interacts_with",
    "contraindicates", "side_effect", "indication",
    "off-label_use", "synergistic_interaction",
    "presents", "phenotype_absent", "phenotype_present",
}

def detect_primekb_data(driver):
    """Check whether :Triplet nodes contain PrimeKB biomedical predicates."""
    with driver.session() as session:
        result = session.run(
            "MATCH (t:Triplet) "
            "WITH t.predicate AS p LIMIT 500 "
            "WITH collect(DISTINCT toLower(p)) AS preds "
            "RETURN preds"
        )
        preds = set(result.single()["preds"])
    
    overlap = preds & PRIMEKB_PREDICATES
    return len(overlap) > 3  # threshold
```

---

## 12. Prompt System Details

### 12.1 Prompt Loader Implementation

```python
# rag/prompts/__init__.py

PROMPTS_DIR = Path(__file__).parent
_prompt_cache: Dict[str, str] = {}

def load_prompt(name: str, use_cache: bool = True, **format_kwargs) -> str:
    """
    Load a prompt template from file.
    
    Args:
        name: Prompt name (without .txt extension)
        use_cache: Whether to cache loaded prompts
        **format_kwargs: Variables to format into template
    """
    if use_cache and name in _prompt_cache and not format_kwargs:
        return _prompt_cache[name]
    
    path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r") as f:
        template = f.read()
    
    if use_cache:
        _prompt_cache[name] = template
    
    if format_kwargs:
        return template.format(**format_kwargs)
    return template

def load_prompt_safe(name: str, fallback: str, **kwargs) -> str:
    """Load prompt with fallback if file not found."""
    try:
        return load_prompt(name, **kwargs)
    except FileNotFoundError:
        return fallback.format(**kwargs) if kwargs else fallback
```

### 12.2 Prompt File Reference

| File | Purpose | Variables |
|------|---------|-----------|
| `knowledge_assessment.txt` | Gap detection + proposal | (none) |
| `answer_generation.txt` | Answer synthesis | (none) |
| `answer_generation_with_new_facts.txt` | Answer noting expansion | (none) |
| `persistence_justification.txt` | Quality gate for persistence | (none) |
| `triplet_validation.txt` | Remote LLM validation | `{query}`, `{existing_facts}`, `{proposed_triplets}` |
| `triplet_reproposal.txt` | Re-propose after rejection | `{query}`, `{feedback}` |
| `mcp_agent_system.txt` | Agent system prompt | `{tools_json}` |
| `triplet_expansion_fallback.txt` | Schema-constrained expansion | `{question}`, `{existing_triplets}`, `{schema_relations}` |

### 12.3 Knowledge Assessment Prompt Structure

```
You are a knowledge graph assistant. Given the user's question and available facts:

1. Assess if facts are SUFFICIENT to provide a complete answer
2. If SUFFICIENT: Provide your answer
3. If INSUFFICIENT:
   - List ALL missing information
   - Propose factual triplets for EVERY missing point

Response Format (JSON):
{
    "assessment": "SUFFICIENT" or "INSUFFICIENT",
    "confidence": 0.0-1.0,
    "answer": "...",
    "missing_information": ["..."],
    "proposed_triplets": [["subject", "predicate", "object"], ...]
}
```

---

## 13. Interactive Agent Implementation

### 13.1 Command Dispatch

```python
COMMANDS = {
    "/help": cmd_help,
    "/quit": cmd_quit,
    "/exit": cmd_quit,
    "/tools": cmd_tools,
    "/stats": cmd_stats,
    "/verbose": cmd_toggle_verbose,
    "/validate": cmd_toggle_validate,
    "/expand": cmd_toggle_expand,
    "/history": cmd_history,
    "/clear": cmd_clear,
    "/search": cmd_search,      # /search <query>
    "/entity": cmd_entity,      # /entity <name>
    "/cypher": cmd_cypher,      # /cypher <query>
    "/search-demo": cmd_search_demo,
    "/expand-demo": cmd_expand_demo,
}

def process_input(user_input: str) -> str:
    """Route user input to command handler or agent."""
    if user_input.startswith("/"):
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in COMMANDS:
            return COMMANDS[cmd](args)
        return f"Unknown command: {cmd}. Type /help for available commands."
    
    # Not a command - send to agent
    return agent.run(user_input)
```

### 13.2 Session State

```python
@dataclass
class SessionState:
    verbose: bool = False
    validate: bool = False
    expand: bool = False
    history: List[str] = field(default_factory=list)
    agent: Optional[MCPAgent] = None
    neo4j_store: Optional[Neo4jStore] = None
    is_primekb: bool = False
```

### 13.3 Agent Mode Selection

```python
def create_agent(state: SessionState):
    """Create appropriate agent based on session state."""
    if state.validate:
        return MCPAgentWithValidation(
            neo4j_store=state.neo4j_store,
            embedder=state.embedder,
            enable_validation=True,
            auto_expand=state.expand,
        )
    else:
        return MCPAgent(
            neo4j_store=state.neo4j_store,
            embedder=state.embedder,
        )
```

### 13.4 Statistics Command Output

```
Knowledge Graph Statistics
==========================
Total triplets: 15,234
Unique subjects: 3,456
Unique predicates: 127
Unique objects: 8,901

Top predicates:
  - treats: 2,345
  - associated_with: 1,876
  - interacts_with: 1,234

Data source: PrimeKG (detected)
Schema: triplet
```
