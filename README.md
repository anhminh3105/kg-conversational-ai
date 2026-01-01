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
            KG Triplets (canon_kg.txt)
                  ↓
            Embed & Index (FAISS)
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
```bash
sudo systemctl start neo4j
```

### Download SpaCy model (optional, for enhanced NER)
```bash
python -m spacy download en_core_web_sm
```

## 🗄️ Neo4j Setup

#### Import sample data
```bash
python scripts/import_kg_data_from_json.py
```

## 📊 Project Structure

```
kg-conversational-ai/
├── data/
│   ├── movie_data.csv              # Sample movie dataset
│   └── training_data_complete.json # Training examples (with Cypher)
├── models/                         # Saved models (generated)
│   ├── intent_classifier.pkl       # ML classifier
│   └── question_to_cypher/         # Fine-tuned T5 model
├── rag/                            # RAG module
│   ├── __init__.py                 # Module exports
│   ├── triplet_loader.py           # Step 1: Load triplets
│   ├── representation.py           # Step 2: Convert to text
│   ├── embedder.py                 # Step 3: Generate embeddings
│   ├── faiss_store.py              # Step 4: FAISS vector store
│   ├── kg_rag_indexer.py           # Main orchestrator (Steps 1-4)
│   ├── retriever.py                # Step 5: Retrieval interface
│   ├── triplet_expander.py         # Step 5.5: LLM triplet expansion
│   ├── prompt_builder.py           # Step 6: Prompt augmentation
│   ├── generator.py                # Step 7: LLM generation
│   └── edc/                        # EDC pipeline
│       ├── prompt_templates/       # All prompt templates
│       │   ├── kg_qa.txt           # QA prompt template
│       │   └── triplet_expansion.txt # Expansion prompt template
│       └── schemas/                # Schema definitions
├── scripts/
│   ├── import_kg_data_from_json.py # Data import script
│   ├── nlp_to_cypher.py            # ML-based NLP-to-Cypher
│   ├── llm_light_train.py          # Transformer training
│   ├── llm_light_demo.py           # Transformer demo/inference
│   ├── index_rag.py                # RAG CLI script
│   └── visulize_graph.py           # Graph visualization
├── export_google_ai.sh             # Google AI Studio config
├── export_sambanova.sh             # SambaNova config
├── export_local_llm.sh             # Local LLM config
├── environment.yml                 # Conda environment
├── requirements.txt                # pip requirements
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

```python
from rag import KGRagIndexer, KGRagGenerator

# Index triplets
indexer = KGRagIndexer()
indexer.index_from_path("./rag/edc/output/tmp", mode="triplet_text")
indexer.save("./output/rag")

# Load and search
indexer = KGRagIndexer.load("./output/rag")
results = indexer.search("Where is Trane located?", top_k=5)

# Full RAG with LLM generation
generator = KGRagGenerator(indexer)
result = generator.generate("Where is Trane located?")
print(result.answer)    # "Trane is located in Swords, Dublin."
print(result.sources)   # [(Trane, location, Swords_Dublin)]

# RAG with triplet expansion (enriches sparse facts)
result = generator.generate(
    "Where was Alan Shepard born?",
    expand_triplets=True,      # Enable LLM expansion
    max_expansion=10,          # Max triplets to generate
)
print(result.answer)
print(result.sources)           # All triplets (retrieved + expanded)
print(result.expanded_triplets) # Only the LLM-generated triplets
```

#### TripletExpander (Standalone Usage)

```python
from rag import TripletExpander

# Create expander with schema constraints
expander = TripletExpander(schema_path="./rag/edc/schemas/webnlg_schema.csv")

# Expand sparse triplets
retrieved = [("Alan_Shepard", "birthPlace", "New_Hampshire")]
expanded = expander.expand(
    query="Tell me about Alan Shepard's career",
    retrieved_triplets=retrieved,
    max_new_triplets=5,
)
# expanded might include:
# [("Alan_Shepard", "occupation", "Astronaut"),
#  ("Alan_Shepard", "mission", "Apollo_14"), ...]
```

### 4. Visualize Knowledge Graph

```bash
python scripts/visulize_graph.py
```

Generates visualization files in `outputs/`:
- `knowledge_graph.png` - Full graph
- `knowledge_graph_drama.png` - Drama movies subgraph
- `knowledge_graph_sci-fi.png` - Sci-Fi movies subgraph
- `knowledge_graph_action.png` - Action movies subgraph

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
