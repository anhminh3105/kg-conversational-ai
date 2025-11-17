# Knowledge Graph Query

This component aims to build a system that translates questions into queries for Neo4j knowledge graphs.

## Overview

This project implements two approaches for converting natural language questions into Cypher queries:

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

### Download SpaCy model (optional, for enhanced NER)
python -m spacy download en_core_web_sm

## 🗄️ Neo4j Setup

#### Import sample data
```bash
python scripts/import_kg_data_from_json.py
```

## 📊 Project Structure

```
thesis/
├── data/
│   ├── movie_data.csv              # Sample movie dataset
│   └── training_data_complete.json # Training examples (with Cypher)
├── models/                         # Saved models (generated)
│   ├── intent_classifier.pkl       # ML classifier
│   └── question_to_cypher/         # Fine-tuned T5 model
├── scripts/
│   ├── import_kg_data_from_json.py # Data import script
│   ├── nlp_to_cypher.py            # ML-based NLP-to-Cypher
│   ├── llm_light_train.py          # Transformer training
│   ├── llm_light_demo.py           # Transformer demo/inference
│   └── visulize_graph.py           # Graph visualization
├── environment.yml                  # Conda environment
├── requirements.txt                 # pip requirements
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

### 3. Visualize Knowledge Graph

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

## 📚 References

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [FastText Documentation](https://fasttext.cc/)
