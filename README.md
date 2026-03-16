# Knowledge Graph Query

This component aims to build a system that translates questions into queries for Neo4j knowledge graphs.

## Overview

This project implements three approaches for knowledge graph question answering:

1. **ML-based Approach** ([`scripts/nlp_to_cypher.py`](scripts/nlp_to_cypher.py)): Uses FastText embeddings and scikit-learn classifiers

      ```
            Natural Language Question
                  ↓
            Intent Classification
                  ↓
            Entity Extraction
                  ↓
            Cypher Query Generation
                  ↓
            Neo4j Execution
                  ↓
            Results
      ```

2. **Transformer-based Approach** ([`scripts/llm_light_train.py`](scripts/llm_light_train.py)): Fine-tunes T5-small model for direct question-to-Cypher translation

      ```
            Natural Language Question
                  ↓
            Cypher Query Generation
                  ↓
            Neo4j Execution
                  ↓
            Results
      ```

3. **RAG-based Approach** ([`scripts/index_rag.py`](scripts/index_rag.py)): Retrieval-Augmented Generation over KG triplets

      ```
            KG Triplets (EDC canon_kg.txt or PrimeKG kg.csv)
                  ↓
            Embed & Index (FAISS / Neo4j)
                  ↓
            Natural Language Question
                  ↓
            Semantic Retrieval
                  ↓
            [Optional] Triplet Expansion (LLM)
                  ↓
            LLM Generation
                  ↓
            Answer with Sources
      ```

## 🚀 Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate nlp-kg
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start Neo4j

**Option A: Local installation (no root required)**
```bash
# First time setup (download Neo4j)
mkdir -p ~/tools && cd ~/tools
curl -L -o neo4j-community-5.26.0-unix.tar.gz "https://neo4j.com/artifact.php?name=neo4j-community-5.26.0-unix.tar.gz"
tar -xzf neo4j-community-5.26.0-unix.tar.gz
cd neo4j-community-5.26.0
bin/neo4j-admin dbms set-initial-password password123

# Start Neo4j
~/tools/neo4j-community-5.26.0/bin/neo4j start

# Other commands
~/tools/neo4j-community-5.26.0/bin/neo4j stop     # Stop
~/tools/neo4j-community-5.26.0/bin/neo4j status   # Check status
```

**Option B: System service (requires root)**
```bash
sudo systemctl start neo4j
```

### Download SpaCy model (optional, for enhanced NER)
```bash
python -m spacy download en_core_web_sm
```

## 🗄️ Neo4j Setup

| Setting | Value |
|---------|-------|
| Web UI | http://localhost:7474 |
| Bolt URL | `bolt://localhost:7687` |
| Username | `neo4j` |
| Password | `password123` |

#### Import sample data
```bash
python scripts/import_kg_data_from_json.py
```

#### Troubleshooting: Authentication Errors

If you encounter `neo4j.exceptions.AuthError: authentication failure`, the password may be out of sync. The `set-initial-password` command only works before the first database start.

**Fix: Complete data reset**
```bash
# Stop Neo4j
~/tools/neo4j-community-5.26.0/bin/neo4j stop

# Remove all data (this deletes your database!)
rm -rf ~/tools/neo4j-community-5.26.0/data/*

# Set password fresh
~/tools/neo4j-community-5.26.0/bin/neo4j-admin dbms set-initial-password password123

# Start Neo4j
~/tools/neo4j-community-5.26.0/bin/neo4j start
```

## 📊 Project Structure

```
kg-conversational-ai/
├── data/
│   ├── movie_data.csv              # Sample movie dataset
│   ├── training_data_complete.json # Training examples (with Cypher)
│   └── kg.csv                      # PrimeKG dataset (download with scripts/download_primekb.py)
├── models/                         # Saved models (generated)
│   ├── intent_classifier.pkl       # ML classifier
│   └── question_to_cypher/         # Fine-tuned T5 model
├── rag/                            # RAG module
│   ├── __init__.py                 # Module exports
│   ├── triplet_loader.py           # Step 1: Load EDC triplets + get_loader() factory
│   ├── primekb_loader.py           # Step 1b: Load PrimeKG triplets from kg.csv
│   ├── representation.py           # Step 2: Convert to text
│   ├── embedder.py                 # Step 3: Generate embeddings
│   ├── faiss_store.py              # Step 4: FAISS vector store
│   ├── neo4j_store.py              # Step 4b: Neo4j vector store
│   ├── kg_rag_indexer.py           # Main orchestrator (Steps 1-4)
│   ├── retriever.py                # Step 5: Retrieval interface
│   ├── triplet_expander.py         # Step 5.5: LLM triplet expansion
│   ├── prompt_builder.py           # Step 6: Prompt augmentation
│   ├── generator.py                # Step 7: LLM generation
│   ├── mcp_neo4j_server.py         # MCP tool handler for Neo4j KG
│   ├── mcp_agent.py                # MCP Agent (agentic KG Q&A)
│   └── edc/                        # EDC pipeline
│       ├── prompt_templates/       # All prompt templates
│       │   ├── kg_qa.txt           # QA prompt template
│       │   └── triplet_expansion.txt # Expansion prompt template
│       └── schemas/                # Schema definitions
│           └── primekb_schema.csv  # PrimeKG relation definitions
├── scripts/
│   ├── import_kg_data_from_json.py # Movie data import script
│   ├── import_primekb_to_neo4j.py  # PrimeKG -> Neo4j graph import
│   ├── download_primekb.py         # Download PrimeKG from Harvard Dataverse
│   ├── nlp_to_cypher.py            # ML-based NLP-to-Cypher
│   ├── llm_light_train.py          # Transformer training
│   ├── llm_light_demo.py           # Transformer demo/inference
│   ├── index_rag.py                # RAG CLI script
│   ├── demo_mcp_agent.py           # MCP Agent batch demo
│   ├── interactive_agent.py        # MCP Agent interactive REPL
│   └── visulize_graph.py           # Graph visualization
├── export_google_ai.sh             # Google AI Studio config
├── export_sambanova.sh             # SambaNova config
├── export_local_llm.sh             # Local LLM config
├── export_dual_llm.sh              # Dual-LLM validation config
├── environment.yml                 # Conda environment
├── requirements.txt                # pip requirements
├── CHANGELOG.md                    # Release changelog
└── README.md
```

## 🎓 Usage

### 1. ML-Based Approach (Lightweight)

**Train the model:**
```bash
python scripts/nlp_to_cypher.py
```

This will:
- Load training data from [`data/training_data.json`](data/training_data.json)
- Train intent classifier using FastText embeddings
- Save model to [`models/intent_classifier.pkl`](models/intent_classifier.pkl)
- Run interactive demo

#### 📚 Supported Query Types

The system supports the following intent categories:

| Intent | Example Question | Generated Cypher |
|--------|-----------------|------------------|
| `FIND_ACTORS_IN_MOVIE` | "Who acted in Forrest Gump?" | `MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'Forrest Gump'}) RETURN p.name` |
| `FIND_MOVIES_BY_ACTOR` | "What movies did Tom Hanks star in?" | `MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN m.title` |
| `FIND_DIRECTOR` | "Who directed Titanic?" | `MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: 'Titanic'}) RETURN p.name` |
| `FIND_MOVIES_BY_DIRECTOR` | "What did Robert Zemeckis direct?" | `MATCH (p:Person {name: 'Robert Zemeckis'})-[:DIRECTED]->(m:Movie) RETURN m.title` |
| `FIND_MOVIES_BY_YEAR` | "Movies from 1999" | `MATCH (m:Movie) WHERE m.year = 1999 RETURN m.title` |
| `FIND_MOVIES_BY_GENRE` | "Show me sci-fi movies" | `MATCH (m:Movie) WHERE m.genre = 'Sci-Fi' RETURN m.title` |
| `COUNT_MOVIES` | "How many movies did Tom Hanks make?" | `MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m) RETURN count(m)` |
| `FIND_COACTORS` | "Who acted with Tom Hanks?" | `MATCH (p1:Person {name: 'Tom Hanks'})-[:ACTED_IN]->()<-[:ACTED_IN]-(p2) RETURN p2.name` |
| `FIND_PATH` | "How are Tom Hanks and Kate Winslet connected?" | `MATCH path = shortestPath((p1 {name: 'Tom Hanks'})-[*]-(p2 {name: 'Kate Winslet'})) RETURN path` |

### 2. Transformer-Based Approach

**Train the T5 model:**
```bash
python scripts/llm_light_train.py
```

Training options:
```bash
python scripts/llm_light_train.py \
  --data data/training_data_complete.json \
  --epochs 10 \
  --batch-size 4 \
  --lr 3e-4
```

**Run inference:**
```bash
python scripts/llm_light_demo.py
```

Interactive demo:
```bash
python scripts/llm_light_demo.py --model models/question_to_cypher
```

Single question:
```bash
python scripts/llm_light_demo.py --question "Who acted in The Matrix?"
```

Batch processing:
```bash
python scripts/llm_light_demo.py --batch questions.txt
```

### 3. RAG-based Approach (KG Triplet Q&A)

The RAG module provides semantic search and LLM-powered question answering over knowledge graph triplets.

#### Prerequisites

Install FAISS for vector search:
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

#### Step 1: Index Triplets

Index triplets from EDC pipeline output (`canon_kg.txt`):
```bash
python scripts/index_rag.py --input ./rag/edc/output_webnlg/iter2 --output_dir ./output/rag
```

#### Step 2: Search (Retrieval Only)

Search for relevant triplets without LLM generation:
```bash
# Single query
python scripts/index_rag.py --load ./output/rag --query "What do you know about Morelos?"

# Interactive mode
python scripts/index_rag.py --load ./output/rag --interactive
```

#### Step 3: Generate Answers with LLM

Configure an LLM provider first:
```bash
# Option 1: Google AI Studio (recommended - free, no GPU required)
source export_google_ai.sh

# Option 2: SambaNova (free, no GPU required)
source export_sambanova.sh

# Option 3: Local LLM (requires GPU + bitsandbytes)
source export_local_llm.sh
```

Then generate answers:
```bash
# Single query with LLM answer
python scripts/index_rag.py --load ./output/rag --generate --query "What do you know about Trane?"

# Interactive Q&A with LLM
python scripts/index_rag.py --load ./output/rag --generate --interactive
```

#### Step 4: Triplet Expansion (Optional)

When retrieved triplets are sparse or insufficient, use `--expand` to have the LLM generate additional related triplets based on its parametric knowledge:

```bash
# Generate with triplet expansion
python scripts/index_rag.py --load ./output/rag --generate --expand --query "Where was Alan Shepard born?"

# Interactive mode with expansion
python scripts/index_rag.py --load ./output/rag --generate --expand --interactive

# Control expansion parameters
python scripts/index_rag.py --load ./output/rag --generate --expand \
  --max_expansion 10 \
  --schema ./rag/edc/schemas/webnlg_schema.csv \
  --query "What do you know about Morelos?"
```

**Expansion Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--expand` | disabled | Enable LLM triplet expansion |
| `--max_expansion` | 10 | Maximum triplets to generate |
| `--schema` | auto-detect | Path to schema CSV for valid predicates |

The expansion uses the schema to constrain generated predicates, ensuring consistency with the knowledge graph ontology.

#### Programmatic Usage

See [scripts/PROGRAMMATIC_USAGE.md](scripts/PROGRAMMATIC_USAGE.md) for Python API examples (indexing, search, generation, triplet expansion, and PrimeKG loader).

#### PrimeKG Dataset (Biomedical Knowledge Graph)

The system supports [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/), a precision medicine knowledge graph with ~8.1 million relationships across 10 biological scales (genes, drugs, diseases, pathways, etc.).

**Download PrimeKG:**
```bash
python scripts/download_primekb.py              # downloads kg.csv (~580 MB)
python scripts/download_primekb.py --all         # also drug/disease feature files
```

**Index PrimeKG for RAG:**
```bash
# Full dataset (may take a while)
python scripts/index_rag.py --input ./data/kg.csv --output_dir ./output/rag_primekb

# Subset: only drug-disease relationships
python scripts/index_rag.py --input ./data/kg.csv --format primekb \
  --node_types drug,disease --max_rows 100000 --output_dir ./output/rag_primekb

# Then search / generate as usual
python scripts/index_rag.py --load ./output/rag_primekb --auto_filter --query "What drugs treat diabetes?" --top_k 50
```

**Import PrimeKG into Neo4j (native graph):**
```bash
python scripts/import_primekb_to_neo4j.py --input ./data/kg.csv
python scripts/import_primekb_to_neo4j.py --input ./data/kg.csv \
  --node_types drug,disease --max_rows 50000
```

**PrimeKG filtering options:**

| Option | CLI flag | Description |
|--------|----------|-------------|
| Node types | `--node_types` | Comma-separated list (e.g. `drug,disease,gene/protein`) |
| Relation types | `--relation_types` | Comma-separated list (e.g. `treats,associates`) |
| Row limit | `--max_rows` | Cap CSV rows loaded (useful for quick experiments) |
| Format | `--format primekb` | Force PrimeKG format (auto-detected for `.csv` files) |

### 4. Visualize Knowledge Graph

```bash
python scripts/visulize_graph.py
```

Generates visualization files in `outputs/`:
- `knowledge_graph.png` - Full graph
- `knowledge_graph_drama.png` - Drama movies subgraph
- `knowledge_graph_sci-fi.png` - Sci-Fi movies subgraph
- `knowledge_graph_action.png` - Action movies subgraph

### 5. MCP Agent -- Agentic Knowledge Graph Q&A

The MCP Agent provides tool-augmented LLM reasoning over the Neo4j knowledge graph. The agent decides which MCP tools to call (semantic search, entity lookup, Cypher queries, triplet expansion) to answer questions autonomously.

Two scripts are provided: a **batch demo** for running predefined workflows and an **interactive REPL** for conversational exploration.

#### Prerequisites

All MCP Agent workflows require Neo4j with indexed data:

```bash
# Ensure Neo4j is running (see Neo4j Setup above)
# Migrate FAISS index into Neo4j (one-time)
python scripts/migrate_faiss_to_neo4j.py --input ./output/rag

# Configure an LLM provider
source export_sambanova.sh    # or export_google_ai.sh, export_local_llm.sh
```

#### Tutorial A: Batch Demo ([`scripts/demo_mcp_agent.py`](scripts/demo_mcp_agent.py))

Runs predefined demo workflows end-to-end, useful for evaluating the pipeline or showcasing features without interactive input.

**Simple search (no LLM required):**
```bash
python scripts/demo_mcp_agent.py --simple
```
Lists available MCP tools and runs a sample semantic search against the knowledge graph.

**Full agent demo with predefined queries:**
```bash
# Using local GPU
source export_local_llm.sh
python scripts/demo_mcp_agent.py

# Using API backend (no GPU needed)
source export_sambanova.sh
python scripts/demo_mcp_agent.py --lite

# With tool call traces
python scripts/demo_mcp_agent.py --lite --verbose
```
Connects to Neo4j, creates the MCP Agent, and runs three predefined queries showing tool call traces and final answers.

**Triplet expansion demo:**
```bash
# Show LLM-generated triplets (read-only)
python scripts/demo_mcp_agent.py --expand

# Generate and persist new triplets to Neo4j
python scripts/demo_mcp_agent.py --expand --persist
```
Searches for existing facts, uses the LLM to generate additional related triplets, and optionally persists them to the graph.

**Dual-LLM validated expansion:**
```bash
# Configure remote validator LLM first
source export_dual_llm.sh

# Run validated expansion
python scripts/demo_mcp_agent.py --validated-expand

# With detailed output
python scripts/demo_mcp_agent.py --validated-expand --verbose

# Custom query
python scripts/demo_mcp_agent.py --validated-expand --query "What awards did Einstein win?"
```
Full dual-LLM workflow: local LLM proposes triplets, remote LLM validates them, then justified triplets are persisted to Neo4j.

**Demo CLI options:**

| Flag | Description |
|------|-------------|
| `--simple` | Simple search demo (no LLM required) |
| `--lite` | Use API backend instead of local GPU |
| `--expand` | Triplet expansion demo |
| `--persist` | Persist expanded triplets (use with `--expand`) |
| `--validated-expand` | Dual-LLM validated expansion demo |
| `--verbose`, `-v` | Show tool call details |
| `--query`, `-q` | Custom query for demos |
| `--neo4j-password` | Neo4j password (default: env or `password123`) |

#### Tutorial B: Interactive REPL ([`scripts/interactive_agent.py`](scripts/interactive_agent.py))

A conversational interface for exploring the knowledge graph. Ask questions in natural language, toggle features on the fly, and inspect tool calls in real-time.

**Start an interactive session:**
```bash
# With API backend (recommended for quick start)
source export_sambanova.sh
python scripts/interactive_agent.py --lite

# With local GPU
source export_local_llm.sh
python scripts/interactive_agent.py

# With dual-LLM validation enabled from the start
source export_dual_llm.sh
python scripts/interactive_agent.py --validate --verbose
```

**One-shot demo modes (run and exit):**
```bash
# Simple search demo (no LLM)
python scripts/interactive_agent.py --simple

# Triplet expansion demo
python scripts/interactive_agent.py --expand

# Expansion with persistence
python scripts/interactive_agent.py --expand --persist
```

**Interactive commands (inside the REPL):**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/quit`, `/exit` | Exit the session |
| `/tools` | List available MCP tools |
| `/stats` | Knowledge graph statistics (predicates, entities) |
| `/verbose` | Toggle verbose mode (tool calls + validation details) |
| `/validate` | Toggle dual-LLM validation mode |
| `/expand` | Toggle triplet expansion for agent queries |
| `/history` | Show recent query history |
| `/clear` | Clear the terminal screen |
| `/search <query>` | Direct semantic search (bypasses agent reasoning) |
| `/entity <name>` | Query all facts about a specific entity |
| `/cypher <query>` | Execute a raw Cypher query against Neo4j |
| `/search-demo` | Run the simple search demo |
| `/expand-demo` | Run triplet expansion demo (add `--persist` to save) |

**Example session:**
```
You: What do you know about Einstein?
Agent: Based on the knowledge graph, Einstein...

You: /verbose
  Verbose mode: ON

You: Tell me about his work in physics
Agent: [search_knowledge_graph] → Found 8 facts
       Einstein is known for the theory of relativity...

You: /entity Einstein
  Found 12 facts:
    (Einstein, born_in, Germany)
    (Einstein, field, Physics)
    ...

You: /expand-demo
  [Runs triplet expansion demo using current session]

You: /quit
```

**Interactive CLI options:**

| Flag | Description |
|------|-------------|
| `--lite` | Use API backend instead of local GPU |
| `--verbose`, `-v` | Start with verbose mode enabled |
| `--validate` | Enable dual-LLM validation from start |
| `--no-expansion` | Disable triplet expansion by default |
| `--simple` | Run simple search demo then exit |
| `--expand` | Run triplet expansion demo then exit |
| `--persist` | Persist expanded triplets (use with `--expand`) |
| `--neo4j-password` | Neo4j password (default: env or `password123`) |

## 📝 Adding New Training Data

1. **Edit training data** in [`data/training_data_complete.json`](data/training_data_complete.json):
```json
{
  "question": "Your question here?",
  "intent": "INTENT_NAME",
  "entities": {"actor": "Name"},
  "cypher": "MATCH (p:Person {name: 'Name'})..."
}
```

2. **Retrain models**:
```bash
# ML approach
python scripts/nlp_to_cypher.py

# Transformer approach
python scripts/llm_light_train.py
```

## 🙏 Acknowledgments

- **Neo4j** for the graph database
- **Hugging Face** for transformer models
- **FastText** for word embeddings
- **SpaCy** for NLP tools
- **FAISS** for vector similarity search
- **Sentence Transformers** for embeddings

## 📚 References

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [FastText Documentation](https://fasttext.cc/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [EDC Framework](https://arxiv.org/abs/2404.03868)
