# Technical Report: Knowledge Graph RAG with Dual-LLM Validation

## Abstract

This report describes a Retrieval-Augmented Generation (RAG) system built on a knowledge graph backend with automatic knowledge expansion capabilities. The system combines semantic search over graph-structured knowledge with a novel dual-LLM architecture that proposes and validates new facts before persisting them. This approach addresses the challenge of sparse knowledge graphs by enabling controlled, fact-checked expansion during query time.

---

## 1. System Overview

### 1.1 Problem Statement

Traditional RAG systems retrieve from static document stores. When knowledge is sparse or missing, they either hallucinate or fail to answer. Knowledge graphs provide structured facts but are expensive to curate manually. This system addresses both issues by:

1. Using a knowledge graph for precise, structured retrieval
2. Automatically detecting when retrieved knowledge is insufficient
3. Proposing new facts using LLM parametric knowledge
4. Validating proposals with a separate LLM before persistence

### 1.2 Core Technologies

| Technology | Role |
|------------|------|
| **Neo4j** | Graph database storing knowledge triplets or native typed graphs |
| **FAISS** | Vector index for semantic similarity search |
| **MCP (Model Context Protocol)** | Tool interface between LLM and database |
| **Sentence Transformers** | Embedding model for semantic search |
| **Local LLM** | Query processing, gap detection, triplet proposal, persistence justification |
| **Remote LLM** | Fact validation and correction |
| **PrimeKG** | Optional biomedical knowledge graph (129K nodes, 8.1M relationships, 10 entity types) |

### 1.3 High-Level Architecture

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

---

## 2. Knowledge Representation

### 2.1 Triplet Model

Knowledge is represented as subject-predicate-object triplets:

```
(Einstein, won_award, Nobel_Prize)
(Einstein, field_of_work, Physics)
(Nobel_Prize, category, Physics)
```

Each triplet is stored as a node in Neo4j with properties:
- `subject`: The entity performing or being described
- `predicate`: The relationship type
- `object`: The target entity or value
- `source`: Origin of the fact (original, llm_expanded, llm_validated)
- `embedding`: Vector representation for semantic search

### 2.2 Dual Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Knowledge Store                       │
├───────────────────────┬─────────────────────────────────┤
│      Neo4j Graph      │         FAISS Index             │
│                       │                                  │
│  • Triplet nodes      │  • Embedding vectors            │
│  • Graph traversal    │  • Approximate nearest neighbor │
│  • Cypher queries     │  • Fast similarity search       │
│  • Relationship hops  │  • Top-k retrieval              │
└───────────────────────┴─────────────────────────────────┘
```

**Neo4j** handles structured queries (e.g., "find all facts about Einstein") via graph traversal. **FAISS** handles semantic queries (e.g., "who won physics awards") via embedding similarity. Both operate over the same triplet data.

---

## 3. MCP Tool Interface

### 3.1 Model Context Protocol

MCP provides a standardized way for LLMs to invoke external tools. The system exposes knowledge graph operations as MCP tools that the LLM can call during reasoning.

### 3.2 Available Tools

| Tool | Purpose | Backend |
|------|---------|---------|
| `search_knowledge_graph` | Semantic similarity search | FAISS + Neo4j |
| `query_entity` | Find all facts about an entity | Neo4j traversal |
| `expand_entity` | N-hop neighborhood exploration | Neo4j traversal |
| `expand_triplets` | Generate new facts via LLM | Local LLM |
| `validate_and_persist_triplets` | Fact-check and store | Remote LLM + Neo4j |

### 3.3 Tool Execution Flow

```
┌─────────┐    Tool Call JSON     ┌──────────────┐
│   LLM   │ ──────────────────►   │ MCP Handler  │
└─────────┘                       └──────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌───────────┐       ┌─────────────┐      ┌─────────────┐
             │   Neo4j   │       │    FAISS    │      │ Remote LLM  │
             │  (graph)  │       │  (vectors)  │      │ (validate)  │
             └───────────┘       └─────────────┘      └─────────────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                                  ┌──────────────┐
                                  │ JSON Result  │
                                  └──────────────┘
```

---

## 4. Semantic Search Pipeline

### 4.1 Query Processing

When a user asks a question, the system converts it to an embedding and finds similar triplets:

```
PROCEDURE semantic_search(query, top_k):
    query_embedding = embed(query)
    candidate_ids = FAISS.search(query_embedding, top_k * 2)
    triplets = Neo4j.fetch(candidate_ids)
    scored_triplets = rank_by_similarity(triplets, query_embedding)
    RETURN top_k(scored_triplets)
```

### 4.2 Embedding Strategy

Each triplet is embedded as a concatenated string:

```
embedding_text = "{subject} {predicate} {object}"
embedding_vector = SentenceTransformer.encode(embedding_text)
```

This allows semantic matching where "Einstein received Nobel Prize" matches queries about "physics awards" or "famous scientists."

---

## 5. Knowledge Gap Detection

### 5.1 Purpose

Before generating an answer, the system assesses whether retrieved facts are sufficient. This prevents hallucination by explicitly identifying when expansion is needed.

### 5.2 LLM-Based Assessment

The local LLM evaluates whether retrieved facts are sufficient to answer the query. If insufficient, it also proposes new triplets in the same pass:

```
PROCEDURE detect_gap_llm(query, retrieved_facts):
    prompt = format_assessment_prompt(query, retrieved_facts)
    response = LocalLLM.generate(prompt)
    assessment = parse_json(response)
    
    RETURN {
        sufficient: assessment.verdict == "SUFFICIENT",
        missing_info: assessment.missing_information,
        proposed_triplets: assessment.proposed_triplets
    }
```

The LLM receives the query and all retrieved facts, then outputs a structured JSON response containing:
- **Verdict**: SUFFICIENT or INSUFFICIENT
- **Missing information**: What knowledge is needed but not present
- **Proposed triplets**: New facts that could help answer the question

---

## 6. Dual-LLM Validation Architecture

### 6.1 Motivation

LLMs can generate plausible-sounding but incorrect facts. Using a single LLM for both proposal and validation creates confirmation bias. The dual-LLM architecture separates these concerns:

- **Local LLM**: Fast, cost-effective, generates proposals
- **Remote LLM**: More capable, acts as independent fact-checker

### 6.2 Validation Workflow (Deferred Persistence)

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │  Local LLM   │  Proposes: (Einstein, born_in, Germany)       │
│  │  (Qwen 7B)   │           (Einstein, died_in, USA)            │
│  └──────────────┘           (Einstein, invented, telephone) ✗   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Remote LLM  │  Validates:                                   │
│  │  (Gemini)    │    ✓ (Einstein, born_in, Germany) - correct   │
│  └──────────────┘    ✓ (Einstein, died_in, USA) - correct       │
│         │            ✗ (Einstein, invented, telephone) - reject │
│         │                                                       │
│         │  Validated triplets held in memory                    │
│         │  (NOT persisted yet)                                  │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ GENERATE     │  Answer uses existing + validated facts       │
│  │ ANSWER       │                                               │
│  └──────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Local LLM   │  Persistence Justification:                   │
│  │  (Justify)   │    ✓ (Einstein, born_in, Germany) - specific  │
│  └──────────────┘    ✓ (Einstein, died_in, USA) - specific      │
│         │            ✗ (AMD, hasStakeholder, Entity) - vague    │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │    Neo4j     │  Persists ONLY justified triplets             │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deferred persistence:** Validated triplets are not persisted during the iterative expansion loop. Instead, they are held in memory and used for answer generation. After the answer is produced, the local LLM evaluates each validated triplet and decides whether it is specific and useful enough to persist. This prevents vague or placeholder triplets (e.g., `(AMD, hasStakeholder, Stakeholder_Entity)`) from polluting the knowledge graph.

### 6.3 Validation Categories

| Category | Description | Action |
|----------|-------------|--------|
| **Validated** | Factually correct as proposed | Persist to graph |
| **Corrected** | Correct concept, wrong details | Persist corrected version |
| **Rejected** | Factually incorrect | Discard, do not persist |

### 6.4 Retry Mechanism

If validation fails (network error, malformed response), the system retries with feedback from the remote LLM to refine proposals:

```
                                ┌─────────────────┐
                                │ Proposed        │
                                │ Triplets        │
                                └────────┬────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │         ATTEMPT 1              │
              ┌─────────┴─────────┐                      │
              ▼                   │                      │
       ┌─────────────┐            │                      │
       │ Remote LLM  │            │                      │
       │  Validate   │            │                      │
       └──────┬──────┘            │                      │
              │                   │                      │
       ┌──────┴──────┐            │                      │
       │             │            │                      │
   SUCCESS       FAILURE          │                      │
       │             │            │                      │
       ▼             ▼            │                      │
  ┌─────────┐  ┌───────────┐      │                      │
  │ Return  │  │ Extract   │      │                      │
  │Validated│  │ Feedback  │      │                      │
  └─────────┘  └─────┬─────┘      │                      │
                     │            │                      │
                     ▼            │                      │
              ┌─────────────┐     │                      │
              │   Refine    │     │                      │
              │  Proposals  │◄────┘                      │
              └──────┬──────┘                            │
                     │                                   │
                     ▼                                   │
        ┌────────────────────────────────┐               │
        │         ATTEMPT 2              │               │
        │    (repeat validation)         │───────────────┤
        └────────────────────────────────┘               │
                     │                                   │
                     ▼                                   │
        ┌────────────────────────────────┐               │
        │         ATTEMPT 3              │               │
        │      (final attempt)           │───────────────┘
        └────────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
        SUCCESS           ALL FAILED
            │                 │
            ▼                 ▼
       ┌─────────┐     ┌───────────┐
       │ Return  │     │  Return   │
       │Validated│     │  Empty    │
       └─────────┘     └───────────┘
```

**Feedback-driven refinement:** When validation fails, the remote LLM provides reasons (e.g., "predicate too vague", "object entity misspelled"). The local LLM uses this feedback to regenerate improved proposals before the next attempt.

### 6.5 Persistence Justification

After the final answer is generated, the local LLM acts as a quality gate for persistence. It evaluates each validated triplet and decides whether it should be stored permanently in the knowledge graph.

```
PROCEDURE justify_persistence(query, answer, validated_triplets):
    prompt = persistence_justification_prompt
    response = LocalLLM.generate(prompt, query, answer, validated_triplets)
    decision = parse_json(response)
    
    RETURN {
        persist: [{triplet, reason}, ...],   # Worth keeping
        skip:    [{triplet, reason}, ...]    # Too vague / generic
    }
```

**Rejection criteria:**
| Criterion | Example |
|-----------|---------|
| Generic placeholders | `(AMD, hasStakeholder, Stakeholder_Entity)` |
| Too vague | `(Entity, type, Entity)` |
| Duplicates existing facts | Already in the knowledge graph |
| Trivially obvious | Adds no real knowledge |

**Acceptance criteria:**
| Criterion | Example |
|-----------|---------|
| Named entities | `(AMD, CEO, Lisa_Su)` |
| Quantifiable facts | `(Shareholder1, sharesHeld, 5000000)` |
| Specific relationships | `(Stakeholder1, role, Board_of_Directors)` |
| Future utility | Would help answer similar questions |

This step ensures that only high-quality, specific facts are persisted, preventing knowledge graph pollution from vague LLM-generated triplets.

---

## 7. Iterative Knowledge Expansion

### 7.1 Expansion Loop

A single search-validate cycle may not yield sufficient knowledge. The system iterates until knowledge is sufficient or maximum iterations reached:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ITERATIVE EXPANSION LOOP                              │
│                 (max 3 iterations, persistence deferred)                      │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │    Query     │
     └──────┬───────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│ ╔═══════════════════════════════════════════════════════════════╗ │
│ ║                    ITERATION 1                                 ║ │
│ ╚═══════════════════════════════════════════════════════════════╝ │
│                                                                   │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│   │ SEARCH  │───►│ ASSESS  │───►│ PROPOSE │───►│VALIDATE │       │
│   │         │    │         │    │         │    │         │       │
│   │ Neo4j + │    │ Local   │    │ Local   │    │ Remote  │       │
│   │ FAISS   │    │ LLM     │    │ LLM     │    │ LLM     │       │
│   └─────────┘    └────┬────┘    └─────────┘    └────┬────┘       │
│                       │                             │             │
│                       ▼                             ▼             │
│                 ┌───────────┐                ┌─────────────┐      │
│                 │SUFFICIENT?│                │ HOLD IN     │      │
│                 └─────┬─────┘                │ MEMORY      │      │
│                       │                      │ (no persist)│      │
│              ┌────────┴────────┐             └─────────────┘      │
│              │                 │                                  │
│             YES               NO                                  │
│              │                 │                                  │
└──────────────┼─────────────────┼──────────────────────────────────┘
               │                 │
               │                 ▼
               │    ┌────────────────────────┐
               │    │ Any new proposals?     │
               │    └───────────┬────────────┘
               │           ┌────┴────┐
               │           │         │
               │          YES        NO
               │           │         │
               │           ▼         │
               │    ┌─────────────┐  │
               │    │ ITERATION 2 │  │
               │    │  (repeat)   │  │
               │    └──────┬──────┘  │
               │           │         │
               │           ▼         │
               │    ┌─────────────┐  │
               │    │ ITERATION 3 │  │
               │    │  (if needed)│  │
               │    └──────┬──────┘  │
               │           │         │
               ▼           ▼         ▼
        ┌─────────────────────────────────────┐
        │          GENERATE ANSWER            │
        │                                     │
        │  Combines:                          │
        │   • Originally retrieved facts      │
        │   • All validated facts (in memory) │
        └──────────────────┬──────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │    PERSISTENCE JUSTIFICATION        │
        │                                     │
        │  Local LLM evaluates each triplet:  │
        │   ✓ Specific, useful → persist      │
        │   ✗ Vague, generic  → skip          │
        └──────────────────┬──────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │       PERSIST TO NEO4J             │
        │                                     │
        │  Only justified triplets stored     │
        └─────────────────────────────────────┘


  ═══════════════════════════════════════════════════════════════════
                        STATE ACCUMULATION
  ═══════════════════════════════════════════════════════════════════
  
  Iteration 1:  known_facts = { } ──────────────────► { A, B }
                                    validated: A, B (held in memory)
                                    
  Iteration 2:  known_facts = { A, B } ─────────────► { A, B, C, D }
                                    validated: C, D (held in memory)
                                    (A, B skipped as duplicates)
                                    
  Iteration 3:  known_facts = { A, B, C, D } ───────► { A, B, C, D, E }
                                    validated: E (held in memory)
                                    (A, B, C, D skipped)
  
  Answer:       Generated with { KG facts + A, B, C, D, E }
  
  Justify:      A (specific) → persist
                B (specific) → persist
                C (vague)    → skip
                D (specific) → persist
                E (generic)  → skip
  
  Persist:      Only { A, B, D } written to Neo4j
  ═══════════════════════════════════════════════════════════════════
```

**Key behaviors:**
- Each iteration builds on accumulated knowledge from previous iterations
- Validated triplets are held in memory (not persisted) during the loop
- Deduplication ensures the same triplet is never proposed twice
- Loop terminates early if: (a) knowledge is sufficient, (b) no new proposals, or (c) max iterations reached
- After the answer is generated, the local LLM justifies which triplets are worth persisting
- Only specific, useful triplets are written to Neo4j; vague/placeholder triplets are skipped

### 7.2 Deduplication

To prevent infinite loops, the system tracks all known facts:

```
known = {normalized(fact) for fact in retrieved_facts}
known += {normalized(fact) for fact in validated_facts}

FOR proposal IN new_proposals:
    IF normalized(proposal) NOT IN known:
        accept(proposal)
```

---

## 8. Agent Architecture

### 8.1 Component Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCPAgentWithValidation                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Embedder   │  │  Neo4j      │  │  MCP Tool Handler       │  │
│  │  (BGE)      │  │  Store      │  │                         │  │
│  └─────────────┘  └─────────────┘  │  • search_knowledge_graph│  │
│        │               │           │  • query_entity          │  │
│        └───────────────┼───────────│  • expand_triplets       │  │
│                        │           │  • validate_and_persist  │  │
│                        │           └─────────────────────────┘  │
│                        │                      │                  │
│                        ▼                      ▼                  │
│              ┌─────────────────────────────────────┐            │
│              │           Local LLM                 │            │
│              │     (reasoning + proposals)         │            │
│              └─────────────────────────────────────┘            │
│                                │                                 │
│                                ▼                                 │
│              ┌─────────────────────────────────────┐            │
│              │          Remote LLM                 │            │
│              │        (validation only)            │            │
│              └─────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Execution Trace Example

```
Query: "What awards did Einstein win?"

[Iteration 1]
  → Tool: search_knowledge_graph("Einstein awards")
  ← Result: [(Einstein, field, Physics), (Einstein, birthplace, Ulm)]
  → Assessment: INSUFFICIENT - no award information found
  → Proposals: [(Einstein, won, Nobel_Prize_Physics),
                (Einstein, award_year, 1921)]
  → Validation (gemini-2.5-flash-lite): 2 validated, 0 rejected
  → Validated triplets held in memory (persistence deferred)

[Iteration 2]
  → Tool: search_knowledge_graph("Einstein awards")
  ← Result: [(Einstein, field, Physics), (Einstein, birthplace, Ulm)]
  → Assessment: SUFFICIENT (validated facts from iter 1 also considered)

[Generate Answer]
  Uses: KG facts + 2 validated facts from iteration 1
  "Einstein won the Nobel Prize in Physics in 1921..."

[Persistence Justification]
  Local LLM evaluates validated triplets:
  ✓ (Einstein, won, Nobel_Prize_Physics) - specific, persist
  ✓ (Einstein, award_year, 1921) - specific date, persist

[Persist]
  2 triplets written to Neo4j
```

---

## 9. Prompt Management System

### 9.1 Architecture

All LLM prompts are externalized to text files for easier maintenance and experimentation. The system loads prompts at runtime with caching and fallback support.

```
rag/prompts/
├── __init__.py                         # Loader utilities
├── mcp_agent_system.txt               # Main agent system prompt
├── mcp_agent_lite_system.txt          # Lightweight agent prompt
├── knowledge_assessment.txt           # Gap detection prompt
├── answer_generation.txt              # Answer synthesis prompt
├── answer_generation_with_new_facts.txt  # Answer with expansion note
├── persistence_justification.txt      # Persistence quality gate prompt
├── triplet_expansion_fallback.txt     # Triplet generation prompt
├── triplet_validation.txt             # Remote LLM validation prompt
├── triplet_reproposal.txt            # Re-proposal after rejection
├── kg_qa_system.txt                   # QA system prompt
└── kg_qa_fallback.txt                 # Fallback QA template
```

### 9.2 Prompt Loading

Prompts are loaded with automatic caching and graceful fallback:

```
PROCEDURE load_prompt(name, **variables):
    IF name IN cache:
        template = cache[name]
    ELSE:
        template = read_file("prompts/{name}.txt")
        cache[name] = template
    
    IF variables:
        RETURN template.format(**variables)
    RETURN template
```

### 9.3 Key Prompts

| Prompt | Purpose |
|--------|---------|
| `knowledge_assessment` | Instructs LLM to assess sufficiency and propose triplets |
| `mcp_agent_system` | Defines available tools and usage patterns for agent |
| `answer_generation` | Guides final answer synthesis from facts |
| `answer_generation_with_new_facts` | Answer synthesis noting KG insufficiency and remote model used |
| `persistence_justification` | Local LLM decides which validated triplets to persist |
| `triplet_validation` | Remote LLM validates proposed triplets for accuracy |
| `triplet_reproposal` | Local LLM re-proposes triplets after rejection feedback |
| `triplet_expansion_fallback` | Schema-constrained triplet generation |

### 9.4 Benefits

- **Maintainability**: Edit prompts without changing Python code
- **Experimentation**: A/B test different prompt strategies
- **Transparency**: Prompts are readable text files, not buried in code
- **Caching**: Prompts loaded once and reused across requests

---

## 10. PrimeKG Dataset Support

### 10.1 Overview

The system supports [PrimeKG](https://github.com/mims-harvard/PrimeKG) (Precision Medicine Knowledge Graph), a large-scale biomedical knowledge graph with ~8.1 million relationships and 129,375 nodes covering drugs, diseases, genes, and biological processes.

### 10.2 Node Types

PrimeKG contains **10 node types** representing different biological entities:

| Node Type | Count | Percentage | Description | Data Sources |
|-----------|-------|------------|-------------|--------------|
| **biological_process** | 28,642 | 22.1% | GO biological processes | CTD, Entrez Gene, Gene Ontology |
| **gene/protein** | 27,671 | 21.4% | Genes and protein products | Bgee, CTD, DisGeNET, DrugBank, Entrez Gene |
| **disease** | 17,080 | 13.2% | Human diseases | CTD, DisGeNET, Disease Ontology, Drug Central |
| **effect/phenotype** | 15,311 | 11.8% | Clinical phenotypes | DisGeNET, Human Phenotype Ontology, SIDER |
| **anatomy** | 14,035 | 10.8% | Anatomical structures | Bgee, UBERON |
| **molecular_function** | 11,169 | 8.6% | GO molecular functions | CTD, Entrez Gene, Gene Ontology |
| **drug** | 7,957 | 6.2% | Approved/experimental drugs | DrugBank, Drug Central, SIDER |
| **cellular_component** | 4,176 | 3.2% | GO cellular components | CTD, Entrez Gene, Gene Ontology |
| **pathway** | 2,516 | 1.9% | Biological pathways | Reactome |
| **exposure** | 818 | 0.6% | Environmental exposures | CTD |

**Total: 129,375 nodes across 10 types**

### 10.3 Relation Types

PrimeKG contains **30 relation types** representing different biological relationships:

| Category | Relations |
|----------|-----------|
| **Drug-Disease** | `treats`, `palliates`, `indication`, `contraindication`, `off-label use` |
| **Drug-Protein** | `target`, `carrier`, `enzyme`, `transporter` |
| **Drug-Drug** | `synergistic interaction` |
| **Drug-Effect** | `side effect` |
| **Protein-Disease** | `associated with` |
| **Protein-Protein** | `interacts with`, `ppi` |
| **Protein-GO** | `bioprocess`, `molfunc`, `cellcomp` |
| **Protein-Pathway** | `pathway` |
| **Protein-Anatomy** | `expressed in`, `present in` |
| **Disease-Phenotype** | `phenotype present`, `phenotype absent` |
| **Disease-Anatomy** | `localizes` |
| **Disease-Disease** | `parent-child` |
| **Exposure-Disease** | `linked to` |
| **Exposure-Protein** | `interacts with` |

### 10.4 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRIMEKG DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

  Harvard Dataverse
        │
        │  download_primekb.py
        ▼
  ┌─────────────┐
  │   kg.csv    │  (~580 MB, 8.1M rows)
  │  (raw data) │
  └──────┬──────┘
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
  ┌─────────────────────┐               ┌─────────────────────┐
  │  PATH A: Triplet    │               │  PATH B: Native     │
  │                     │               │  PrimeKG Graph      │
  │  PrimeKBLoader      │               │                     │
  │       ↓             │               │  import_primekb_    │
  │  index_rag.py       │               │  to_neo4j.py        │
  │       ↓             │               │       ↓             │
  │  :Triplet nodes     │               │  :Drug, :Disease,   │
  │  (subject, pred,    │               │  :Gene nodes with   │
  │   object)           │               │  typed rels         │
  └─────────────────────┘               └─────────────────────┘
         │                                         │
         └─────────────────┬───────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Neo4j    │
                    │  Knowledge  │
                    │    Graph    │
                    └─────────────┘
```

### 10.5 Two Import Modes

| Mode | Schema | Use Case |
|------|--------|----------|
| **Triplet** | `:Triplet` nodes with `(subject, predicate, object)` | Unified RAG pipeline, semantic search |
| **Native** | Typed nodes (`:Drug`, `:Disease`, `:Gene`) with typed relationships | Graph analytics, Cypher queries |

### 10.6 PrimeKBLoader

Loads PrimeKG CSV and converts to triplets compatible with the RAG pipeline:

```
PROCEDURE load_primekb(path, node_types, relation_types, max_rows):
    df = read_csv(path, nrows=max_rows)
    
    IF node_types:
        df = filter_by_node_type(df, node_types)
    IF relation_types:
        df = filter_by_relation_type(df, relation_types)
    
    triplets = []
    FOR row IN df:
        triplet = Triplet(
            subject = row.x_name,
            predicate = row.display_relation,
            object = row.y_name,
            metadata = {x_type, y_type, x_id, y_id, source}
        )
        triplets.append(triplet)
    
    RETURN triplets
```

**Filtering options:**
- `node_types`: Filter to specific entity types (e.g., `["drug", "disease"]`)
- `relation_types`: Filter to specific relationships (e.g., `["treats", "interacts_with"]`)
- `max_rows`: Limit rows for testing with large dataset

### 10.7 Native PrimeKG Import

For graph analytics, the native import creates typed nodes and relationships:

```
CSV row:
  x_name: "aspirin", x_type: "drug"
  display_relation: "treats"
  y_name: "headache", y_type: "disease"

Neo4j result:
  (:Drug {name: "aspirin", primekb_id: "..."})
    -[:TREATS]->
  (:Disease {name: "headache", primekb_id: "..."})
```

### 10.8 Auto-Detection

The visualization and interactive agent automatically detect which schema is in use:

```
PROCEDURE detect_schema():
    IF count(:Triplet) > 0:
        RETURN "triplet"
    ELIF count(nodes with primekb_id) > 0:
        RETURN "primekb"
    ELSE:
        RETURN "triplet"  # default
```

---

## 11. Interactive Agent

### 11.1 REPL Interface

The interactive agent provides a chat-style interface for exploring the knowledge graph:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE AGENT                                │
└─────────────────────────────────────────────────────────────────────────┘

  User Input ─────────────────────────────────────────────────────────────►
       │
       ▼
  ┌─────────────┐
  │ Command?    │───► YES ───► Execute command (/help, /stats, /tools...)
  └──────┬──────┘
         │ NO
         ▼
  ┌─────────────────────┐
  │  MCPAgent or        │
  │  MCPAgentWith       │
  │  Validation         │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Tool calls         │───► search_knowledge_graph
  │  (visible in        │───► query_entity
  │   verbose mode)     │───► expand_triplets
  └──────────┬──────────┘     validate_and_persist
             │
             ▼
       Final Answer ◄─────────────────────────────────────────────────────
```

### 11.2 Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit`, `/exit` | Exit the session |
| `/tools` | List available MCP tools |
| `/stats` | Show knowledge graph statistics |
| `/verbose` | Toggle verbose mode (show tool calls) |
| `/validate` | Toggle dual-LLM validation mode |
| `/expand` | Toggle triplet expansion |
| `/history` | Show query history |
| `/search <query>` | Direct semantic search (no agent reasoning) |
| `/entity <name>` | Query all facts about an entity |
| `/cypher <query>` | Execute raw Cypher query |
| `/search-demo` | Run simple search demo |
| `/expand-demo` | Run triplet expansion demo |

### 11.3 PrimeKB Detection

When PrimeKB data is detected, the agent adapts its behavior:

```
PROCEDURE detect_primekb_data():
    sample_predicates = query("MATCH (t:Triplet) RETURN t.predicate LIMIT 500")
    
    primekb_predicates = {"treats", "associated_with", "interacts_with", 
                          "contraindicates", "side_effect", "indication", ...}
    
    overlap = intersection(sample_predicates, primekb_predicates)
    
    RETURN len(overlap) > threshold
```

When PrimeKB is detected, the agent uses domain-appropriate prompts for biomedical queries.

---

## 12. Configuration

### 12.1 LLM Providers

The system supports multiple LLM backends via OpenAI-compatible APIs:

| Provider | Use Case | Model |
|----------|----------|-------|
| Local (Qwen) | Queries, proposals | Qwen2.5-7B-Instruct |
| Google AI | Validation | Gemini 2.5 Flash |
| OpenAI | Validation | GPT-4o-mini |
| SambaNova | Validation | Llama-3.1-70B |

### 12.2 Resource Requirements

| Component | Requirement |
|-----------|-------------|
| Local LLM | GPU with 6GB+ VRAM (4-bit quantization) |
| Neo4j | 2GB RAM minimum |
| FAISS | CPU, scales with triplet count |
| Remote LLM | API key, network access |

---

## 13. Summary

This system extends traditional RAG with:

1. **Structured knowledge** via triplet-based graph storage
2. **Hybrid retrieval** combining graph traversal and vector similarity
3. **Automatic expansion** when knowledge gaps are detected
4. **Fact validation** using independent LLM verification
5. **Iterative refinement** until sufficient knowledge is gathered
6. **Deferred persistence** with local LLM quality gate
7. **Persistence justification** filtering vague/placeholder triplets
8. **Remote model transparency** showing which frontier model validated facts
9. **Externalized prompts** for maintainability and experimentation
10. **PrimeKG support** for biomedical knowledge graphs (129K nodes, 8.1M relationships, 10 node types)
11. **Dual schema support** (triplet nodes or native typed graph)
12. **Interactive agent** with REPL interface and domain auto-detection

The dual-LLM architecture ensures that automatically generated knowledge is verified before persistence, and a second-stage quality gate by the local LLM prevents vague or placeholder triplets from polluting the knowledge graph. Only specific, useful facts are persisted, while the full set of validated facts is used for answer generation.

The system supports both custom domain knowledge (via EDC triplet extraction) and large-scale biomedical knowledge (via PrimeKG), with automatic schema detection enabling seamless switching between data sources.