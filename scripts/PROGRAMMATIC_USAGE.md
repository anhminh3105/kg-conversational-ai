# Programmatic Usage & API Reference

This document consolidates all programmatic and CLI usage documentation for the RAG pipelines.

---

## Table of Contents

- [EDC RAG Pipeline](#edc-rag-pipeline)
  - [KGRagIndexer & KGRagGenerator](#kgragindexer--kgraggenerator)
  - [TripletExpander (Standalone)](#tripletexpander-standalone)
  - [MCPAgentWithValidation (Dual-LLM, Optional Persistence)](#mcpagentwithvalidation-dual-llm-optional-persistence)
  - [PrimeKG Programmatic Usage](#primekg-programmatic-usage)
- [Atlas RAG Pipeline](#atlas-rag-pipeline)
  - [CLI Reference](#cli-reference)
  - [Python API](#python-api)

---

# EDC RAG Pipeline

## KGRagIndexer & KGRagGenerator

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

## TripletExpander (Standalone)

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

## MCPAgentWithValidation (Dual-LLM, Optional Persistence)

The dual-LLM agent that powers `scripts/interactive_agent.py --validate` is
also available as a Python API. The `enable_persistence` flag mirrors the
CLI's `--no-persist`: when `False`, validation runs end-to-end (local LLM
proposes triplets, remote LLM accepts/rejects them, validated triplets are
folded into the final answer) but the post-answer
`validate_and_persist_triplets` step is skipped, so Neo4j is left unchanged.
Useful for reproducible regression tests, evaluation runs that should not
mutate the graph, and notebook exploration.

```python
from rag.mcp_agent import create_mcp_agent_with_validation

# Read-only dual-LLM mode (no Neo4j writes)
agent = create_mcp_agent_with_validation(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password123",
    enable_persistence=False,   # <-- skip writing validated triplets to Neo4j
)

# Validation still runs; the response carries any newly-validated triplets
# inline, but Neo4j's :Triplet count is preserved across calls.
result = agent.run("What are the off-label uses of Propranolol?", verbose=True)
print(result["answer"])
```

When `enable_persistence=False` and the run produces validated triplets,
verbose mode prints a confirmation log line:

```
Persistence disabled (enable_persistence=False); skipping justification and persistence of N validated triplet(s)
```

Drop the flag (or pass `enable_persistence=True`, which is the default) to
restore the standard behaviour where validated triplets are justified and
written back to Neo4j as new `:Triplet` nodes.

See also: `scripts/interactive_agent.py --validate --no-persist` (CLI
equivalent), and the
[`scripts/sample_test_questions.py`](sample_test_questions.py) sampler for
generating reproducible question batches that pair well with this mode.

## PrimeKG Programmatic Usage

```python
from rag import PrimeKGLoader, KGRagIndexer

# Load with filters
loader = PrimeKGLoader("data/kg.csv", node_types=["drug", "disease"], max_rows=50000)
triplets = loader.load().parse()
print(f"Node types: {loader.get_unique_node_types()}")
print(f"Relations: {loader.get_unique_relations()}")

# Index via the unified interface (auto-detects CSV as PrimeKG)
indexer = KGRagIndexer()
indexer.index_from_path("data/kg.csv", node_types=["drug", "disease"], max_rows=50000)
indexer.save("./output/rag_primekg")
```

---

# Atlas RAG Pipeline

A unified RAG system built on top of the [AutoSchemaKG](../AutoSchemaKG/) framework. Enables knowledge graph construction from text and question answering using various LLM providers.

**Features:**
- Knowledge Graph Construction from text documents
- Vector Indexing with FAISS for efficient retrieval
- RAG-based Q&A using retrieved knowledge graph context
- Multi-Provider Support: Google AI Studio, SambaNova, local LLMs, any OpenAI-compatible API

## CLI Reference

### LLM Provider Configuration

Choose one of the supported providers:

```bash
# Google AI Studio
source export_google_ai.sh

# SambaNova
source export_sambanova.sh

# Local LLM (HuggingFace with quantization)
source export_local_llm.sh
```

**Supported Providers:**

| Provider | Export Script | Notes |
|----------|---------------|-------|
| Google AI Studio | `export_google_ai.sh` | Gemini models |
| SambaNova | `export_sambanova.sh` | Llama models |
| Local HuggingFace | `export_local_llm.sh` | With 4/8-bit quantization |
| OpenAI | Custom | Set env vars manually |
| Together AI | Custom | OpenAI-compatible |
| DeepInfra | Custom | OpenAI-compatible |
| vLLM Server | Custom | Local inference server |

**API Mode environment variables** (set by export scripts):

```bash
export OPENAI_KEY="your-api-key"
export OPENAI_API_BASE="https://api.provider.com/v1"
export OPENAI_MODEL="model-name"
export EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"
```

**Local Mode environment variables** (set by `export_local_llm.sh`):

```bash
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"
export LOCAL_LLM_QUANTIZE="4bit"  # 4bit, 8bit, or none
```

Check configuration:

```bash
python -c "from atlas_rag_utils.llm_factory import print_config_summary; print_config_summary()"
```

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `--build` | Build KG from input text using KnowledgeGraphExtractor |
| `--index` | Create embeddings and FAISS index from GraphML |
| `--search` | Retrieve triplets without LLM generation |
| `--generate` | Full RAG with LLM answer generation |

### Build Mode

Build a knowledge graph from input text.

```bash
python scripts/atlas_rag_pipeline.py --build [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | Required | Path to input text file (one document per line) |
| `--output` | `./output/atlas_rag` | Output directory for KG |
| `--max_documents` | None | Maximum documents to process |
| `--batch_size` | 64 | Batch size for processing |
| `--include_concept` | False | Include concept nodes in KG |

```bash
# Build KG from first 100 documents
python scripts/atlas_rag_pipeline.py --build \
    --input rag/edc/datasets/webnlg.txt \
    --output ./output/webnlg_100 \
    --max_documents 100
```

### Index Mode

Create embeddings and FAISS index for the knowledge graph.

```bash
python scripts/atlas_rag_pipeline.py --index [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--kg_dir` | Required | Path to KG directory |
| `--kg_name` | Auto-detected | Name/keyword for the KG |
| `--device` | Auto | Device for embeddings (cuda/cpu) |
| `--batch_size` | 64 | Batch size for embedding |
| `--include_events` | True | Include event nodes |
| `--include_concept` | False | Include concept nodes |

```bash
python scripts/atlas_rag_pipeline.py --index \
    --kg_dir ./output/webnlg_100 \
    --device cpu
```

### Search Mode

Retrieve relevant triplets without LLM generation.

```bash
python scripts/atlas_rag_pipeline.py --search [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--kg_dir` | Required | Path to KG directory |
| `--query` | None | Query string |
| `--interactive` | False | Run interactive mode |
| `--top_k` | 10 | Number of triplets to retrieve |

```bash
# Single query
python scripts/atlas_rag_pipeline.py --search \
    --kg_dir ./output/webnlg_100 \
    --query "Ciudad Ayala population"

# Interactive mode
python scripts/atlas_rag_pipeline.py --search --interactive \
    --kg_dir ./output/webnlg_100
```

### Generate Mode

Full RAG pipeline with LLM answer generation.

```bash
python scripts/atlas_rag_pipeline.py --generate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--kg_dir` | Required | Path to KG directory |
| `--query` | None | Query string |
| `--interactive` | False | Run interactive mode |
| `--top_k` | 10 | Number of triplets to retrieve |
| `--temperature` | 0.1 | LLM temperature |
| `--max_tokens` | 512 | Maximum tokens to generate |

```bash
# Single query
python scripts/atlas_rag_pipeline.py --generate \
    --kg_dir ./output/webnlg_100 \
    --query "What is the population of Ciudad Ayala?" \
    --top_k 5

# Interactive Q&A session
python scripts/atlas_rag_pipeline.py --generate --interactive \
    --kg_dir ./output/webnlg_100 \
    --temperature 0.2
```

### Input/Output Format

**Input** -- one document per line in a text file:

```
The location of Trane is Swords, Dublin.
The ALCO RS-3 has a diesel-electric transmission.
Ciudad Ayala is a city in Morelos, Mexico.
```

**Output directory structure:**

```
output/atlas_webnlg/
├── data/
│   └── webnlg.json           # Converted input documents
├── kg_extraction/            # Raw extraction results
├── triples_csv/              # Triple CSVs
├── concept_csv/              # Concept CSVs
├── kg_graphml/
│   └── webnlg_graph.graphml  # Final knowledge graph
└── precompute/
    ├── *_embeddings.pkl      # Cached embeddings
    └── *_faiss.index         # FAISS indices
```

---

## Python API

### Module Overview

```
scripts/atlas_rag_utils/
├── __init__.py          # Module exports
├── data_converter.py    # Text to JSON converter
├── llm_factory.py       # LLM/Embedder factory
└── config.py            # Configuration dataclasses
```

### Setup

```python
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "AutoSchemaKG")

from atlas_rag_utils import (
    convert_text_to_json,
    TextToJsonConverter,
    create_llm_generator,
    create_embedder,
    get_llm_config,
    AtlasConfig,
)
```

### data_converter

Converts text files to AutoSchemaKG-compatible JSON format.

#### convert_text_to_json()

```python
from atlas_rag_utils.data_converter import convert_text_to_json

# Basic usage
count = convert_text_to_json(
    input_path="input.txt",
    output_path="output.json",
)
print(f"Converted {count} documents")

# With options
count = convert_text_to_json(
    input_path="rag/edc/datasets/webnlg.txt",
    output_path="output/data/webnlg.json",
    max_documents=100,    # Limit to first 100 docs
    id_prefix="webnlg",   # Custom ID prefix
)
```

#### TextToJsonConverter

```python
from atlas_rag_utils.data_converter import TextToJsonConverter

converter = TextToJsonConverter(
    id_prefix="doc",
    skip_empty=True,
    strip_whitespace=True,
)

# Iterate over documents (lazy loading)
for doc in converter.iter_documents("input.txt"):
    print(f"{doc.id}: {doc.text[:50]}...")
    print(f"  Metadata: {doc.metadata}")

# Convert to JSON file
count = converter.convert(
    input_path="input.txt",
    output_path="output.json",
    max_documents=None,
    start_index=0,
)

# Convert to JSONL (one JSON object per line)
count = converter.convert_to_jsonl(
    input_path="input.txt",
    output_path="output.jsonl",
)
```

#### TextDocument

```python
from atlas_rag_utils.data_converter import TextDocument

doc = TextDocument(
    id="doc_0",
    text="The location of Trane is Swords, Dublin.",
    metadata={"lang": "en", "source": "webnlg"}
)

data = doc.to_dict()
# {'id': 'doc_0', 'text': '...', 'metadata': {'lang': 'en', 'source': 'webnlg'}}
```

### llm_factory

Factory functions for creating LLM generators and embedders from environment configuration.

#### Configuration Functions

```python
from atlas_rag_utils.llm_factory import (
    get_llm_config,
    get_embedder_config,
    print_config_summary,
)

print_config_summary()

llm_config = get_llm_config()
print(f"Model: {llm_config.model_name}")
print(f"Is Local: {llm_config.is_local}")
print(f"Base URL: {llm_config.base_url}")

embedder_config = get_embedder_config(device="cpu")
print(f"Embedder: {embedder_config.model_name}")
print(f"Device: {embedder_config.device}")
```

#### create_llm_generator()

```python
from atlas_rag_utils.llm_factory import create_llm_generator

# Auto-detect provider from environment
llm_generator = create_llm_generator()

# Or with explicit config
from atlas_rag_utils.llm_factory import get_llm_config

config = get_llm_config()
llm_generator = create_llm_generator(config)

# Use the generator
response = llm_generator.generate_with_context(
    question="Where is Trane located?",
    context="The location of Trane is Swords, Dublin.",
    max_new_tokens=256,
    temperature=0.1,
)
print(response)
```

#### create_embedder()

```python
from atlas_rag_utils.llm_factory import create_embedder, EmbedderConfig

# Auto-detect from environment
embedder = create_embedder()

# With explicit config
config = EmbedderConfig(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
)
embedder = create_embedder(config)

# Encode text
embeddings = embedder.encode(["Hello world", "Another sentence"])
print(f"Shape: {embeddings.shape}")  # (2, 384)

# Encode with options
embeddings = embedder.encode(
    ["Query text"],
    normalize_embeddings=True,
    batch_size=32,
)
```

### config

#### AtlasConfig

```python
from atlas_rag_utils.config import AtlasConfig

config = AtlasConfig(
    input_path="input.txt",
    output_dir="./output/my_kg",
    kg_name="my_kg",
    max_documents=1000,
    batch_size_triple=16,
    batch_size_concept=16,
    max_new_tokens=2048,
    max_workers=3,
    include_concept=False,
    include_events=True,
    remove_doc_spaces=True,
    record=True,
)

# Computed paths
print(config.data_dir)         # ./output/my_kg/data
print(config.graphml_dir)      # ./output/my_kg/kg_graphml
print(config.precompute_dir)   # ./output/my_kg/precompute
print(config.input_json_path)  # ./output/my_kg/data/my_kg.json
print(config.graphml_path)     # ./output/my_kg/kg_graphml/my_kg_graph.graphml
```

#### get_processing_config()

```python
from atlas_rag_utils.config import AtlasConfig, get_processing_config

atlas_config = AtlasConfig(
    input_path="input.txt",
    output_dir="./output/my_kg",
)

processing_config = get_processing_config(
    atlas_config,
    llm_model_name="gemini-2.5-flash"
)

# Use with KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor

extractor = KnowledgeGraphExtractor(
    model=llm_generator,
    config=processing_config,
)
```

### Full Pipeline Example

Complete example of building and querying a knowledge graph.

```python
import os
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "AutoSchemaKG")

# Set environment (or source export script before running)
os.environ["OPENAI_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "https://api.provider.com/v1"
os.environ["OPENAI_MODEL"] = "model-name"
os.environ["EMBEDDER_MODEL"] = "BAAI/bge-small-en-v1.5"

from atlas_rag_utils.data_converter import convert_text_to_json
from atlas_rag_utils.llm_factory import create_llm_generator, create_embedder
from atlas_rag_utils.config import AtlasConfig, get_processing_config

# Step 1: Convert text to JSON
config = AtlasConfig(
    input_path="rag/edc/datasets/webnlg.txt",
    output_dir="./output/webnlg_demo",
    max_documents=10,
)

os.makedirs(config.data_dir, exist_ok=True)
convert_text_to_json(
    config.input_path,
    config.input_json_path,
    max_documents=config.max_documents,
)

# Step 2: Build knowledge graph
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor

llm_generator = create_llm_generator()
processing_config = get_processing_config(config, llm_generator.model_name)

extractor = KnowledgeGraphExtractor(
    model=llm_generator,
    config=processing_config,
)

extractor.run_extraction()
extractor.convert_json_to_csv()
extractor.generate_concept_csv_temp()
extractor.create_concept_csv()
extractor.convert_to_graphml()

print(f"GraphML saved to: {config.graphml_path}")

# Step 3: Create embeddings and index
from atlas_rag.vectorstore import create_embeddings_and_index

embedder = create_embedder()

data = create_embeddings_and_index(
    sentence_encoder=embedder,
    model_name="BAAI/bge-small-en-v1.5",
    working_directory=config.output_dir,
    keyword=config.kg_name,
    include_concept=False,
    include_events=True,
    normalize_embeddings=True,
)

print(f"Indexed {len(data['node_list'])} nodes, {len(data['edge_list'])} edges")

# Step 4: Create retriever and query
from atlas_rag.retriever import HippoRAG2Retriever

retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=embedder,
    data=data,
)

query = "Where is Trane located?"
content, context_ids = retriever.retrieve(query, topN=5)

print(f"\nQuery: {query}")
print("Retrieved context:")
for i, ctx in enumerate(content):
    print(f"  [{i+1}] {ctx[:100]}...")

# Generate answer
context_str = "\n".join(content)
answer = llm_generator.generate_with_context(
    query,
    context_str,
    max_new_tokens=256,
    temperature=0.1,
)

print(f"\nAnswer: {answer}")
```

### Retriever Usage

```python
from atlas_rag.retriever import HippoRAG2Retriever
from atlas_rag.retriever.inference_config import InferenceConfig

inference_config = InferenceConfig(
    hipporag_mode="query2edge",  # or "query2node", "ner2node"
    keyword="my_kg",
)

retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=embedder,
    data=data,
    inference_config=inference_config,
)

content, context_ids = retriever.retrieve(
    query="What is the transmission type of ALCO RS-3?",
    topN=10,
)

# Access the knowledge graph directly
kg = data["KG"]  # NetworkX graph
print(f"Nodes: {kg.number_of_nodes()}")
print(f"Edges: {kg.number_of_edges()}")
```

### Loading Pre-built Index

```python
from atlas_rag.vectorstore import create_embeddings_and_index
from atlas_rag_utils.llm_factory import create_embedder, create_llm_generator
from atlas_rag.retriever import HippoRAG2Retriever

embedder = create_embedder()

data = create_embeddings_and_index(
    sentence_encoder=embedder,
    model_name="BAAI/bge-small-en-v1.5",
    working_directory="./output/webnlg_demo",
    keyword="webnlg_demo",
    include_concept=False,
    include_events=True,
)

llm_generator = create_llm_generator()
retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=embedder,
    data=data,
)

content, _ = retriever.retrieve("Your question here", topN=5)
```

### Custom LLM Configuration

```python
from openai import OpenAI
from atlas_rag.llm_generator.llm_generator import LLMGenerator

client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-custom-endpoint.com/v1",
)

llm_generator = LLMGenerator(
    client=client,
    model_name="your-model-name",
    backend="openai",
    max_workers=8,
)

response = llm_generator.generate_with_context(
    "What is 2+2?",
    "Mathematics context...",
    max_new_tokens=100,
)
print(response)
```

### Error Handling

```python
from atlas_rag_utils.llm_factory import get_llm_config

try:
    config = get_llm_config()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please run: source export_google_ai.sh")

# Check if CUDA is available
from atlas_rag_utils.llm_factory import _check_cuda_available

if _check_cuda_available():
    print("CUDA is available and working")
else:
    print("Using CPU (CUDA not available or not compatible)")
```

---

## Troubleshooting

### CUDA Not Available

The pipeline automatically falls back to CPU:

```
CUDA not usable: CUDA error: no kernel image is available...
Creating SentenceEmbedding
  Device: cpu
```

### Module Not Found

Add AutoSchemaKG to Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/AutoSchemaKG"
```

### Missing Environment Variables

Source the appropriate export script before running:

```bash
source export_google_ai.sh  # or other provider
```

---

## See Also

- [AutoSchemaKG Documentation](../AutoSchemaKG/README.md) - Underlying framework documentation
- [Main README](../README.md) - Project overview
