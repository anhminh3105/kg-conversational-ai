# Knowledge Graph Query

This component aims to build a system that translates questions into queries for Neo4j knowledge graphs.

## Overview

This project implements three approaches for knowledge graph question answering:

1. **ML-based Approach** (`[scripts/nlp_to_cypher.py](scripts/nlp_to_cypher.py)`): Uses FastText embeddings and scikit-learn classifiers
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
  ```
2. **Transformer-based Approach** (`[scripts/llm_light_train.py](scripts/llm_light_train.py)`): Fine-tunes T5-small model for direct question-to-Cypher translation
  ```
            Natural Language Question
                  ↓
            Cypher Query Generation
                  ↓
            Neo4j Execution
                  ↓
            Results
      ```
  ```
3. **RAG-based Approach** (`[scripts/index_rag.py](scripts/index_rag.py)`): Retrieval-Augmented Generation over KG triplets
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


| Setting  | Value                                          |
| -------- | ---------------------------------------------- |
| Web UI   | [http://localhost:7474](http://localhost:7474) |
| Bolt URL | `bolt://localhost:7687`                        |
| Username | `neo4j`                                        |
| Password | `password123`                                  |


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

#### Fast database wipe (without Cypher)

Deleting millions of nodes via Cypher (`MATCH (n) DETACH DELETE n`) can take hours on
large databases. The fastest way to wipe a Neo4j Community database is to delete the
data files directly:

```bash
~/tools/neo4j-community-5.26.0/bin/neo4j stop

rm -rf ~/tools/neo4j-community-5.26.0/data/databases/neo4j
rm -rf ~/tools/neo4j-community-5.26.0/data/transactions/neo4j

~/tools/neo4j-community-5.26.0/bin/neo4j start
```

Neo4j recreates an empty `neo4j` database on startup. This takes seconds regardless
of how many nodes were stored. Use this when re-importing data from scratch.

## 📊 Project Structure

```
kg-conversational-ai/
├── data/
│   ├── movie_data.csv              # Sample movie dataset
│   ├── training_data_complete.json # Training examples (with Cypher)
│   ├── kg.csv                      # PrimeKG dataset (download with scripts/download_primekb.py)
│   ├── kg_drug_disease.csv         # PrimeKG drug-disease subset (filtered from kg.csv)
│   ├── disease_features.csv        # DrugBank feature columns (half_life, mechanism_of_action, ...)
│   └── eval/                       # Outputs of scripts/eval/* (generated)
│       ├── train.csv                # Train split (drug-aware, 70%)
│       ├── test.csv                 # Held-out split (30%)
│       ├── tier_assignments.json    # Per-(drug, relation) tier + held-out answers
│       ├── interactive_test_questions.txt   # Sampled questions
│       └── interactive_test_questions.json  # Same, with gold triples
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
│   ├── import_primekb_to_neo4j.py  # PrimeKG CSV -> Neo4j :Triplet nodes (with embeddings)
│   ├── download_primekb.py         # Download PrimeKG from Harvard Dataverse
│   ├── nlp_to_cypher.py            # ML-based NLP-to-Cypher
│   ├── llm_light_train.py          # Transformer training
│   ├── llm_light_demo.py           # Transformer demo/inference
│   ├── index_rag.py                # RAG CLI script
│   ├── demo_mcp_agent.py           # MCP Agent batch demo
│   ├── interactive_agent.py        # MCP Agent interactive REPL (supports --no-persist)
│   ├── sample_test_questions.py    # Sample PrimeKG drug questions for interactive agent
│   ├── migrate_faiss_to_neo4j.py   # Migrate FAISS index → Neo4j
│   ├── visualize_kg.py             # KG visualization (Triplet + PrimeKG schemas)
│   ├── visulize_graph.py           # Graph visualization (movie data)
│   └── eval/                       # Evaluation suite (PrimeKG drug-disease)
│       ├── split_primekb.py        # Drug-aware 2-tier train/test split (Full/Partial)
│       ├── generate_qa.py          # Tier-aware LLM-generated QA from test triplets
│       ├── evaluate.py             # Run system + compute metrics (Gold Recall, etc.)
│       ├── merge_eval_reports.py   # Merge supplementary results into existing reports
│       ├── run_eval_pipeline.sh    # End-to-end automation (split → import → QA → eval)
│       ├── run_eval_clean.sh       # Clean eval: clear Neo4j → reimport → evaluate
│       └── README.md               # Evaluation suite documentation
├── export_google_ai.sh             # Google AI Studio config
├── export_sambanova.sh             # SambaNova config
├── export_local_llm.sh             # Local LLM config
├── export_local_qwen3.sh           # Local Qwen3 config (eval QA generation)
├── export_dual_llm.sh              # Dual-LLM: local primary + remote validation
├── export_dual_remote_llm.sh       # Dual-Remote: both LLMs via API (no GPU)
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

- Load training data from `[data/training_data.json](data/training_data.json)`
- Train intent classifier using FastText embeddings
- Save model to `[models/intent_classifier.pkl](models/intent_classifier.pkl)`
- Run interactive demo

#### 📚 Supported Query Types

The system supports the following intent categories:


| Intent                    | Example Question                                | Generated Cypher                                                                                  |
| ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `FIND_ACTORS_IN_MOVIE`    | "Who acted in Forrest Gump?"                    | `MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'Forrest Gump'}) RETURN p.name`                   |
| `FIND_MOVIES_BY_ACTOR`    | "What movies did Tom Hanks star in?"            | `MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN m.title`                      |
| `FIND_DIRECTOR`           | "Who directed Titanic?"                         | `MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: 'Titanic'}) RETURN p.name`                        |
| `FIND_MOVIES_BY_DIRECTOR` | "What did Robert Zemeckis direct?"              | `MATCH (p:Person {name: 'Robert Zemeckis'})-[:DIRECTED]->(m:Movie) RETURN m.title`                |
| `FIND_MOVIES_BY_YEAR`     | "Movies from 1999"                              | `MATCH (m:Movie) WHERE m.year = 1999 RETURN m.title`                                              |
| `FIND_MOVIES_BY_GENRE`    | "Show me sci-fi movies"                         | `MATCH (m:Movie) WHERE m.genre = 'Sci-Fi' RETURN m.title`                                         |
| `COUNT_MOVIES`            | "How many movies did Tom Hanks make?"           | `MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m) RETURN count(m)`                           |
| `FIND_COACTORS`           | "Who acted with Tom Hanks?"                     | `MATCH (p1:Person {name: 'Tom Hanks'})-[:ACTED_IN]->()<-[:ACTED_IN]-(p2) RETURN p2.name`          |
| `FIND_PATH`               | "How are Tom Hanks and Kate Winslet connected?" | `MATCH path = shortestPath((p1 {name: 'Tom Hanks'})-[*]-(p2 {name: 'Kate Winslet'})) RETURN path` |


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

# Option 4: Dual-LLM — local primary + remote validation (Config C evaluation)
source export_dual_llm.sh

# Option 5: Dual-Remote — both LLMs via API, no GPU needed
source export_dual_remote_llm.sh
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


| Option            | Default     | Description                             |
| ----------------- | ----------- | --------------------------------------- |
| `--expand`        | disabled    | Enable LLM triplet expansion            |
| `--max_expansion` | 10          | Maximum triplets to generate            |
| `--schema`        | auto-detect | Path to schema CSV for valid predicates |


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

**Import PrimeKG into Neo4j as :Triplet nodes (one step):**

Reads the CSV and writes flat `:Triplet` nodes with vector embeddings directly
to Neo4j, ready for the demo scripts and RAG pipeline.

```bash
# Default: import the drug-disease subset (~42K rows)
python scripts/import_primekb_to_neo4j.py

# Import the full PrimeKG dataset
python scripts/import_primekb_to_neo4j.py --input data/kg.csv

# Wipe all existing nodes first, then import
python scripts/import_primekb_to_neo4j.py --clear

# Limit rows for a quick test
python scripts/import_primekb_to_neo4j.py --max-rows 500

# Filter by relation types
python scripts/import_primekb_to_neo4j.py --relation-types contraindication,indication

# Now demos work with PrimeKG data
python scripts/demo_mcp_agent.py --simple
python scripts/interactive_agent.py --lite
```

**PrimeKG filtering options:**


| Option         | CLI flag           | Description                                             |
| -------------- | ------------------ | ------------------------------------------------------- |
| Node types     | `--node_types`     | Comma-separated list (e.g. `drug,disease,gene/protein`) |
| Relation types | `--relation_types` | Comma-separated list (e.g. `treats,associates`)         |
| Row limit      | `--max_rows`       | Cap CSV rows loaded (useful for quick experiments)      |
| Format         | `--format primekb` | Force PrimeKG format (auto-detected for `.csv` files)   |


### Migrate FAISS Index to Neo4j (`[scripts/migrate_faiss_to_neo4j.py](scripts/migrate_faiss_to_neo4j.py)`)

After indexing triplets into a FAISS vector store (Step 1 above), use the migration script to transfer the embeddings and triplet metadata into Neo4j. This enables graph traversal, Cypher queries, and the MCP Agent workflows alongside vector similarity search.

**Migrate EDC / WebNLG triplets:**

The EDC pipeline produces `canon_kg.txt` in `rag/edc/output_webnlg/iter2`. After indexing those triplets into FAISS, migrate them to Neo4j:

```bash
# 1. Index EDC triplets into FAISS (if not already done)
python scripts/index_rag.py --input ./rag/edc/output_webnlg/iter2 --output_dir ./output/rag

# 2. Migrate to Neo4j
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag

# 3. Clear existing data first, then migrate
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag --clear

# 4. Verify migration (compares FAISS and Neo4j counts + sample queries)
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag --verify
```

**Migrate PrimeKG triplets:**

For the biomedical PrimeKG dataset, first index into FAISS then migrate:

```bash
# 1. Index PrimeKG into FAISS (full or filtered)
python scripts/index_rag.py --input ./data/kg.csv --output_dir ./output/rag_primekb
python scripts/index_rag.py --input ./data/kg.csv --format primekb \
    --node_types drug,disease --max_rows 100000 --output_dir ./output/rag_primekb

# 2. Migrate to Neo4j (clear to avoid mixing with WebNLG data)
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag_primekb --clear

# 3. Re-embed during migration (useful if switching embedding models)
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag_primekb \
    --re-embed --embedding-model BAAI/bge-small-en-v1.5

# 4. Dry run to preview without writing
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag_primekb --dry-run
```

**Migration CLI options:**


| Flag                | Default                                                | Description                                                                           |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `--faiss-dir`       | *(required)*                                           | Directory containing FAISS index files (`kg_triplets.faiss`, `kg_triplets_meta.json`) |
| `--prefix`          | `kg_triplets`                                          | FAISS file prefix                                                                     |
| `--neo4j-uri`       | `bolt://localhost:7687`                                | Neo4j Bolt URI                                                                        |
| `--neo4j-user`      | `neo4j`                                                | Neo4j username                                                                        |
| `--neo4j-password`  | env `NEO4J_PASSWORD` or `password123`                  | Neo4j password                                                                        |
| `--neo4j-database`  | `neo4j`                                                | Neo4j database name                                                                   |
| `--batch-size`      | `100`                                                  | Triplets per batch                                                                    |
| `--re-embed`        | disabled                                               | Re-generate embeddings during migration                                               |
| `--embedding-model` | env `LOCAL_EMBEDDER_MODEL` or `BAAI/bge-small-en-v1.5` | Model for `--re-embed`                                                                |
| `--clear`           | disabled                                               | Delete existing `:Triplet` nodes before migrating                                     |
| `--verify`          | disabled                                               | Compare FAISS and Neo4j counts after migration                                        |
| `--dry-run`         | disabled                                               | Show stats and sample triplets without writing                                        |


### 4. Visualize Knowledge Graph

**Movie data visualization:**

```bash
python scripts/visulize_graph.py
```

Generates visualization files in `outputs/`:

- `knowledge_graph.png` - Full graph
- `knowledge_graph_drama.png` - Drama movies subgraph
- `knowledge_graph_sci-fi.png` - Sci-Fi movies subgraph
- `knowledge_graph_action.png` - Action movies subgraph

**KG triplet / PrimeKG visualization:**

```bash
# Auto-detect schema (Triplet or PrimeKG)
python scripts/visualize_kg.py --output outputs/kg_full.png

# Visualize native PrimeKG graph
python scripts/visualize_kg.py --schema primekb --output outputs/primekb_graph.png

# Visualize PrimeKG entity subgraph
python scripts/visualize_kg.py --schema primekb --entity "aspirin" --depth 2
```

### 5. MCP Agent -- Agentic Knowledge Graph Q&A

The MCP Agent provides tool-augmented LLM reasoning over the Neo4j knowledge graph. The agent decides which MCP tools to call (semantic search, entity lookup, Cypher queries, triplet expansion) to answer questions autonomously.

Two scripts are provided: a **batch demo** for running predefined workflows and an **interactive REPL** for conversational exploration.

#### Prerequisites

All MCP Agent workflows require Neo4j with indexed data:

```bash
# Ensure Neo4j is running (see Neo4j Setup above)
# Migrate FAISS index into Neo4j (one-time)
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag

# Or use PrimeKG data instead:
python scripts/index_rag.py --input ./data/kg.csv --format primekb \
    --node_types drug,disease --output_dir ./output/rag_primekb
python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag_primekb

# Configure an LLM provider
source export_sambanova.sh    # or export_google_ai.sh, export_local_llm.sh
```

#### Tutorial A: Batch Demo (`[scripts/demo_mcp_agent.py](scripts/demo_mcp_agent.py)`)

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


| Flag                 | Description                                     |
| -------------------- | ----------------------------------------------- |
| `--simple`           | Simple search demo (no LLM required)            |
| `--lite`             | Use API backend instead of local GPU            |
| `--expand`           | Triplet expansion demo                          |
| `--persist`          | Persist expanded triplets (use with `--expand`) |
| `--validated-expand` | Dual-LLM validated expansion demo               |
| `--verbose`, `-v`    | Show tool call details                          |
| `--query`, `-q`      | Custom query for demos                          |
| `--neo4j-password`   | Neo4j password (default: env or `password123`)  |


#### Tutorial B: Interactive REPL (`[scripts/interactive_agent.py](scripts/interactive_agent.py)`)

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

# Dual-LLM validation without persisting validated triplets to Neo4j
# (read-only mode: validation runs, but the graph is not mutated)
python scripts/interactive_agent.py --validate --no-persist --verbose

# Read-only validation that lets the local LLM judge per-query whether to
# enter the propose/validate cycle (recommended for reproducible testing)
python scripts/interactive_agent.py --validate --no-persist --expand-mode auto
```

`**--no-persist` (read-only dual-LLM mode):**

When `--validate` is on, the interactive agent normally (a) proposes triplets
for any knowledge gaps, (b) has the remote LLM validate them, and (c) writes
the validated triplets back to Neo4j via `validate_and_persist_triplets`.
Passing `--no-persist` keeps steps (a) and (b) but skips step (c), so the
graph stays unchanged between runs. The verbose log confirms the guard fired:

```
Persistence disabled (enable_persistence=False); skipping justification and
persistence of N validated triplet(s)
```

Use this mode for:

- Replaying the same test questions repeatedly without state accumulating
between runs (pair it with `scripts/sample_test_questions.py`, below).
- Letting the validator surface any proposed fact without committing it to
the production KG.
- Any scenario where you want dual-LLM reasoning but a read-only Neo4j.

`**--expand-mode` (when does the propose/validate cycle run?):**

In `--validate` runs, the agent's *first* iteration always asks the local
LLM to assess whether the retrieved KG facts are sufficient for the query.
What happens next depends on `--expand-mode`:


| Mode              | Behavior on `SUFFICIENT`                              | Behavior on `INSUFFICIENT`         |
| ----------------- | ----------------------------------------------------- | ---------------------------------- |
| `never` (default) | Skip the cycle; answer from KG.                       | Skip the cycle; answer from KG.    |
| `auto`            | Skip the cycle; answer from KG.                       | Run propose/validate, then answer. |
| `always`          | Override assessor; run propose/validate, then answer. | Run propose/validate, then answer. |


```bash
# Default (never): no propose/validate cycle, pure KG retrieval + answer
python scripts/interactive_agent.py --validate

# Auto: let the local LLM judge each query (recommended for read-only testing)
python scripts/interactive_agent.py --validate --no-persist --expand-mode auto

# Always: legacy behavior -- exercise the dual-LLM machinery on every query
python scripts/interactive_agent.py --validate --expand-mode always
```

You can switch modes mid-session with `/expand` (cycles `never` -> `auto`
-> `always`) or set explicitly with `/expand auto` / `/expand always` /
`/expand never`. The current mode is shown in the startup banner and the
status row above the first prompt.

The natural pairing for `scripts/sample_test_questions.py --source partial`
is `--validate --no-persist --expand-mode auto`: the partial-tier
questions are designed to make the local LLM judge `INSUFFICIENT`, which
triggers the propose/validate cycle, while `--no-persist` keeps Neo4j
unchanged so the run is reproducible.

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


| Command           | Description                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/help`           | Show all available commands                                                                                                                                               |
| `/quit`, `/exit`  | Exit the session                                                                                                                                                          |
| `/tools`          | List available MCP tools                                                                                                                                                  |
| `/stats`          | Knowledge graph statistics (predicates, entities)                                                                                                                         |
| `/verbose`        | Toggle verbose mode (tool calls + validation details)                                                                                                                     |
| `/validate`       | Toggle dual-LLM validation mode                                                                                                                                           |
| `/expand [mode]`  | Cycle expand mode (`never` -> `auto` -> `always`), or set explicitly: `/expand auto`. Controls when the dual-LLM propose/validate cycle runs (see `--expand-mode` below). |
| `/history`        | Show recent query history                                                                                                                                                 |
| `/clear`          | Clear the terminal screen                                                                                                                                                 |
| `/search <query>` | Direct semantic search (bypasses agent reasoning)                                                                                                                         |
| `/entity <name>`  | Query all facts about a specific entity                                                                                                                                   |
| `/cypher <query>` | Execute a raw Cypher query against Neo4j                                                                                                                                  |
| `/search-demo`    | Run the simple search demo                                                                                                                                                |
| `/expand-demo`    | Run triplet expansion demo (add `--persist` to save)                                                                                                                      |


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


| Flag                                | Description                                                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--lite`                            | Use API backend instead of local GPU                                                                                         |
| `--verbose`, `-v`                   | Start with verbose mode enabled                                                                                              |
| `--validate`                        | Enable dual-LLM validation from start                                                                                        |
| `--no-persist`                      | In `--validate` mode, skip persisting validated triplets to Neo4j (read-only dual-LLM mode). No effect without `--validate`. |
| `--expand-mode {always,auto,never}` | Triplet expansion policy for `--validate` runs (default: `never`). See the explainer below.                                  |
| `--no-expansion`                    | DEPRECATED: legacy alias for `--expand-mode never` (kept for back-compat). Prefer `--expand-mode never`.                     |
| `--simple`                          | Run simple search demo then exit                                                                                             |
| `--expand`                          | Run triplet expansion demo then exit                                                                                         |
| `--persist`                         | Persist expanded triplets (use with `--expand`)                                                                              |
| `--neo4j-password`                  | Neo4j password (default: env or `password123`)                                                                               |


#### Tutorial C: Sample Test Questions (`[scripts/sample_test_questions.py](scripts/sample_test_questions.py)`)

Deterministic sampler that produces a handful of drug-focused test questions
ready to paste into the interactive REPL. Questions are drawn from one of
four related sources: the full PrimeKG drug-disease subset, the train/test
split CSVs, or the partial-tier entries in `tier_assignments.json`. No LLM
is called during sampling -- the output is fully reproducible for a given
`--seed`.

**Quick start:**

```bash
# Default: 12 questions from the partial-tier pool (best for dual-LLM testing)
python scripts/sample_test_questions.py

# Sample from the full PrimeKG drug-disease subset (no eval split needed)
python scripts/sample_test_questions.py --source kg --n 12

# Sample from the train / test splits
python scripts/sample_test_questions.py --source train --n 9
python scripts/sample_test_questions.py --source test  --n 9

# Text-only output (copy-paste friendly)
python scripts/sample_test_questions.py --format txt

# Write the text file to a custom path (implies --format txt)
python scripts/sample_test_questions.py --txt-path data/eval/my_questions.txt
```

**What each `--source` selects:**

- `**partial`** (default) -- drugs labelled Partial tier in
`data/eval/tier_assignments.json`: each `(drug, relation)` pair has some
gold answers in Neo4j and some held out. Forward-only (drug -> disease)
questions; each record carries `kg_triples`, `held_out_triples`, and the
union as `gold_triples`. Pairs cleanly with `--no-persist` since the
held-out answers reliably trigger the dual-LLM proposal/validation path.
- `**kg**` -- the full `data/kg_drug_disease.csv` subset (no eval-pipeline
prerequisites). Stratified across the three drug-disease relations
(`indication`, `contraindication`, `off-label use`) with a 50/50 split
of forward (drug -> disease) and reverse (disease -> drug) phrasings
drawn from the same templates as `scripts/eval/generate_qa.py`.
- `**train**` -- drugs whose triples were placed in Neo4j by
`scripts/eval/split_primekb.py`. Best for sanity-checking RAG on facts
that should be retrievable directly.
- `**test**` -- drugs whose triples were held out from Neo4j entirely.
Best for exercising the agent's knowledge-gap behaviour at the drug
level (whereas `partial` exercises gaps at the per-answer level).

All sources reuse the same templates, so the rendered questions look
identical across them; only the underlying `(entity, gold_triples)` set
changes.

**Prerequisites:**

- `--source kg` only needs `data/kg_drug_disease.csv` (run
`python scripts/download_primekb.py --skip_summary`).
- `--source train` / `test` / `partial` (default) all need the eval split
artifacts under `--split-dir` (default `data/eval/`):
  ```bash
  python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv
  ```
  The sampler exits with a single-line remediation message if any required
  file is missing.

**Outputs (in `--out-dir`, default `data/eval/`):**

- `interactive_test_questions.txt` -- one question per line, grouped by
`source/relation` with `# header` comments; ready to paste into the REPL.
- `interactive_test_questions.json` -- structured records with `id`,
`entity`, `relation`, `direction`, `source`, and `gold_triples`. For
`--source partial`, records additionally carry `tier`, `kg_triples`,
and `held_out_triples` so you can see at a glance which gold answers
should be retrievable from Neo4j and which require the dual-LLM path.

**CLI options:**


| Flag          | Default                    | Description                                                                                                        |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--source`    | `partial`                  | Where to sample from: `kg`, `train`, `test`, or `partial`                                                          |
| `--n`         | `12`                       | Total number of questions to sample                                                                                |
| `--seed`      | `42`                       | RNG seed for reproducibility                                                                                       |
| `--kg-csv`    | `data/kg_drug_disease.csv` | Drug-disease KG CSV (used when `--source kg`)                                                                      |
| `--split-dir` | `data/eval`                | Directory with `train.csv`, `test.csv`, `tier_assignments.json` (used when `--source` is `train`/`test`/`partial`) |
| `--out-dir`   | `data/eval`                | Output directory for default filenames                                                                             |
| `--format`    | `both`                     | Which file(s) to emit: `txt`, `json`, or `both`                                                                    |
| `--txt-path`  | --                         | Explicit text path (implies `--format txt`)                                                                        |
| `--json-path` | --                         | Explicit JSON path (implies `--format json`)                                                                       |


**Pair with `--no-persist` for reproducible interactive testing:**

```bash
# 1. Generate a reproducible batch of partial-tier questions
python scripts/sample_test_questions.py --n 12 --seed 42

# 2. Run the agent in read-only dual-LLM mode with --expand-mode auto so the
#    local LLM judges per-query whether to enter the propose/validate cycle.
source export_dual_llm.sh
python scripts/interactive_agent.py \
    --validate --no-persist --expand-mode auto --verbose

# 3. Paste questions from data/eval/interactive_test_questions.txt one at
#    a time. Partial-tier questions reliably trigger INSUFFICIENT, so the
#    propose/validate cycle exercises end-to-end. Validation runs as usual,
#    but Neo4j is not mutated, so the same seed + same questions hit the
#    same KG state on every run.
```

### 6. Evaluation Suite -- PrimeKG Drug-Disease

Quantitative evaluation of the KG-RAG system using the PrimeKG drug-disease subset (indication, contraindication, off-label use). Compares configurations under controlled knowledge-availability conditions using a 2-tier (Full/Partial) drug-aware split.

#### Quick start (automated pipeline)

```bash
# Run everything end-to-end: split → import → QA gen → evaluate
source export_dual_remote_llm.sh   # or export_dual_llm.sh (needs GPU)
bash scripts/eval/run_eval_pipeline.sh
```

#### Step 1: Extract and split the dataset

```bash
# Extract drug-disease subset from PrimeKG (~43K rows)
python scripts/download_primekb.py --skip_summary

# Drug-aware 2-tier split
python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv \
  --max-rows 1000 --full-ratio 0.30
```

The split uses drug-aware sampling (all triples for each drug stay together), runs a per-relation **entity-disjoint** train/test split (no drug appears in both sides for the same relation), and then assigns drugs to two tiers:


| Tier        | Default share | Neo4j coverage         | Purpose                                              |
| ----------- | ------------- | ---------------------- | ---------------------------------------------------- |
| **Full**    | 30%           | 100% of drug's triples | Baseline -- gold answers fully in KG                 |
| **Partial** | 70%           | Subset (some held out) | Core test -- system must recover gaps via validation |


The contrast between what `train.csv` (and therefore Neo4j) sees and what `test.csv` (the QA gold ground truth) sees is what makes Config B vs Config C a clean recovery test:

| Drug's tier for relation **r** | `train.csv` sees                      | `test.csv` sees                  | `kg_answers`         | `held_out_answers`        |
| ------------------------------ | ------------------------------------- | -------------------------------- | -------------------- | ------------------------- |
| **Full**                       | every `(drug, r, *)` triple           | the **same** complete set        | = `gold_answers`     | `[]`                      |
| **Partial**                    | a subset (`partial_include` fraction) | every `(drug, r, *)` triple      | a strict subset      | the rest of `gold_answers` |

Per-relation entity-disjointness is what makes this contrast hold: because the same drug never appears on both sides, the only triples Config B can retrieve for a test drug are the ones the splitter *deliberately* placed in `train.csv`. There is no leakage path from the train side that would let Config B silently recover a "held-out" answer.

Only drugs with >= 2 triples per relation are eligible for the Partial tier. Held-out answers are recorded in `tier_assignments.json` for recovery measurement.

Outputs: `train.csv`, `test.csv`, `tier_assignments.json`, `split_stats.json`. See `[scripts/eval/README.md](scripts/eval/README.md)` for the full Stage 1-4 algorithm, the per-tier visibility derivation, and worked examples.

**Methodology notes:**

- **Subject-only disjointness.** Drug (`x_name`) is split; disease (`y_name`) stays shared as KG connective tissue. Reverse-question overlap is handled by the **any-Partial** inheritance rule.
- **Role of the 80 % background.** Train-only drugs never appear in `gold_answers` (so they don't move Gold Recall) but they populate Neo4j to ~93 % density and supply the realistic distractors and latency a deployment would face.
- **Alternative considered.** A simpler tier-only-on-all-drugs design (no Stage 2 split) is documented as a v2 option: honest hallucination metric and ~5× more questions, at the cost of ~5× eval throughput and a smaller (but more honest) Δ(B → C) headline.

See [scripts/eval/README.md § Why entity-disjoint per relation?](scripts/eval/README.md#why-entity-disjoint-per-relation) for derivations and worked examples.

#### Step 2: Generate QA dataset

Tier-aware, **bidirectional** QA generation:

- **Forward** questions are grouped by drug ("What conditions is *Drug X* indicated for?"). The drug's tier record (`kg_answers`, `held_out_answers`) is copied straight onto the question.
- **Reverse** questions are grouped by disease ("Which drugs are indicated for *Disease Y*?"). Each contributing drug's tier record is re-projected onto the disease, so `kg_answers` lists the drugs whose `(drug, r, Y)` edge sits in `train.csv` and `held_out_answers` lists the ones whose edge was held out. The reverse question inherits **Partial** if *any* contributing drug is Partial — a single missing edge is enough to make Neo4j alone unable to return the complete gold set.

Full-tier questions get the standard prompt; Partial-tier questions use the completeness-demanding `PARTIAL_RELATION_CONTEXT` prompt ("List ALL drugs...", "Provide every...") that reliably trips Config B's INSUFFICIENT assessment and routes Config C into propose / validate. See `[scripts/eval/README.md § QA Dataset Generation](scripts/eval/README.md#qa-dataset-generation)` for the per-record derivation and a worked Fever example.

```bash
# Remote LLM (recommended for quality)
source export_dual_remote_llm.sh
python scripts/eval/generate_qa.py --remote-llm

# Or local LLM
source export_local_qwen3.sh
python scripts/eval/generate_qa.py
```

The output file (`qa_dataset.json`) doubles as an incremental cache. If interrupted, re-running resumes from where it left off.

#### Step 3: Evaluate

```bash
# Configure LLM backend
source export_dual_llm.sh          # local primary + remote validation
# OR
source export_dual_remote_llm.sh   # both remote (no GPU needed)

# Run Configs B and C (default)
python scripts/eval/evaluate.py --verbose

# Filter by tier
python scripts/eval/evaluate.py --configs B C --tier partial

# Re-run specific failed questions
python scripts/eval/evaluate.py --configs C \
  --questions "q0014,q0034,q0039" --output-dir data/eval --verbose

# Clean evaluation (clear Neo4j, reimport, evaluate)
bash scripts/eval/run_eval_clean.sh
```

**Evaluation configurations:**


| Config | Alias                | Description                          | Behavior on INSUFFICIENT               |
| ------ | -------------------- | ------------------------------------ | -------------------------------------- |
| **A**  | `rag`                | Pure RAG (vector retrieval only)     | N/A (no assessment)                    |
| **B**  | `without-validation` | Agent with KG tools, no validation   | Returns refusal                        |
| **C**  | `with-validation`    | Agent + dual-LLM validated expansion | Proposes triplets, validates, persists |


**Primary metric: Gold Recall** (mean fraction of ALL gold answers found per question). Replaces the earlier Exact Match (binary) and KG Recall (KG-only) with a single measure of retrieval + answer completeness. Also reports Answer Rate, Hallucination Proxy, Format Failures, and Latency -- all broken down overall, per-relation, and per-tier.

**Key results** (gemma-4-26b-a4b-it primary, gemma-4-31b-it validation):


| Metric        | Config B | Config C  | Delta     |
| ------------- | -------- | --------- | --------- |
| Gold Recall   | 0.669    | **0.720** | **+7.7%** |
| Answer Rate   | 90.2%    | **98.5%** | **+9.1%** |
| Hallucination | 0.09%    | **0.04%** | **-56%**  |
| Latency (s)   | 23.1     | 172.8     | +7.5x     |


See `[data/eval/EVALUATION_REPORT.md](data/eval/EVALUATION_REPORT.md)` for the full results analysis and `[scripts/eval/README.md](scripts/eval/README.md)` for CLI reference.

### 7. LLM Configuration Scripts

Six configuration scripts set environment variables for different LLM backends. Use `source` to load them before running any agent or evaluation command.


| Script                      | GPU required | Primary LLM               | Validation LLM           | Typical use                  |
| --------------------------- | ------------ | ------------------------- | ------------------------ | ---------------------------- |
| `export_google_ai.sh`       | No           | Google AI Studio (Gemini) | --                       | Quick demos, Config A/B eval |
| `export_sambanova.sh`       | No           | SambaNova API             | --                       | Alternative remote backend   |
| `export_local_llm.sh`       | Yes          | Local HuggingFace model   | --                       | Config B eval                |
| `export_local_qwen3.sh`     | Yes          | Qwen3 (local)             | --                       | QA dataset generation        |
| `export_dual_llm.sh`        | Yes          | Local HuggingFace model   | Remote API (Gemma)       | Config C eval (GPU)          |
| `export_dual_remote_llm.sh` | No           | Remote API (Gemma 4-26b)  | Remote API (Gemma 4-31b) | Config C eval (no GPU)       |


**Key environment variables:**


| Variable                                                          | Set by                    | Description                                                                |
| ----------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `USE_LOCAL_LLM`                                                   | all scripts               | `true` for local GPU inference, `false` for API                            |
| `LOCAL_LLM_MODEL`                                                 | local scripts             | HuggingFace model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`)                     |
| `LOCAL_LLM_QUANTIZE`                                              | local scripts             | Quantization level (`4bit`, `8bit`, or empty)                              |
| `LOCAL_LLM_ENABLE_THINKING`                                       | `dual_llm`, `local_qwen3` | Enable chain-of-thought for Qwen3+ models (see below)                      |
| `LOCAL_EMBEDDER_MODEL`                                            | all scripts               | Embedding model (default: `BAAI/bge-small-en-v1.5`)                        |
| `OPENAI_KEY` / `OPENAI_API_BASE` / `OPENAI_MODEL`                 | remote scripts            | Primary LLM via OpenAI-compatible API                                      |
| `REMOTE_LLM_API_KEY` / `REMOTE_LLM_BASE_URL` / `REMOTE_LLM_MODEL` | dual scripts              | Validation LLM for Config C                                                |
| `HF_TOKEN`                                                        | `dual_llm`                | HuggingFace token for gated models (MedGemma, etc.)                        |
| `OPENAI_HTTP_TIMEOUT`                                             | `dual_remote_llm`         | HTTP timeout in seconds for primary LLM API calls (default: SDK 600s)      |
| `REMOTE_LLM_HTTP_TIMEOUT`                                         | `dual_remote_llm`         | HTTP timeout in seconds for validation LLM API calls                       |
| `SHOW_THOUGHT_BLOCKS`                                             | --                        | Set `true` to show `<thought>` blocks in verbose output (default: `false`) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`                     | all scripts               | Neo4j connection settings                                                  |


#### Chain-of-thought thinking mode (Qwen3+ models)

When using a local Qwen3 or newer model, you can enable an internal chain-of-thought reasoning step. The model produces `<think>...</think>` blocks that are automatically stripped from the final answer but allow deeper multi-step reasoning.

```bash
# Enable before sourcing the config
export LOCAL_LLM_ENABLE_THINKING=true
source export_dual_llm.sh

# Or toggle inline
LOCAL_LLM_ENABLE_THINKING=true python scripts/demo_mcp_agent.py --validated-expand
```

Thinking mode is off by default. It increases latency but can improve answer quality for complex reasoning tasks. Only Qwen3+ models support this; the flag is silently ignored for other models.

## 📝 Adding New Training Data

1. **Edit training data** in `[data/training_data_complete.json](data/training_data_complete.json)`:

```json
{
  "question": "Your question here?",
  "intent": "INTENT_NAME",
  "entities": {"actor": "Name"},
  "cypher": "MATCH (p:Person {name: 'Name'})..."
}
```

1. **Retrain models**:

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

