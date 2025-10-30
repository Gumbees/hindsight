# Entity-Aware Memory System for AI Agents

A proof-of-concept memory system that enables AI agents to store, retrieve, and connect memories using temporal, semantic, and entity-based relationships.

## Overview

This system implements a sophisticated graph-based memory architecture where memories are connected through three complementary networks:

1. **Temporal Network** - Memories linked by time proximity
2. **Semantic Network** - Memories linked by meaning similarity
3. **Entity Network** - Memories linked by shared entities (people, organizations, places)

The combination of these three networks enables powerful memory retrieval that goes beyond simple vector search, allowing agents to find relevant memories through multiple pathways.

## Architecture

### Core Concepts

**Memory Units**: Individual sentence-level memories that are:
- Self-contained (pronouns resolved to actual referents)
- Validated to have subject + verb (complete thoughts)
- Embedded as vectors for semantic similarity
- Timestamped for temporal relationships
- Linked to extracted entities

**Entity Resolution**: Named entities (PERSON, ORG, GPE, etc.) are:
- Extracted using spaCy NER
- Disambiguated using a scoring algorithm
- Tracked with canonical IDs across all memories
- Used to create strong connections between related memories

### Three Types of Memory Links

#### 1. Temporal Links (Time-Based)
**Purpose**: Connect memories that occurred close together in time

**How it works**:
- When storing a new memory, find all memories within a time window (default: 24 hours)
- Create weighted links based on temporal proximity
- Weight formula: `weight = max(0.3, 1.0 - (time_diff / window_size))`
- Closer in time = stronger link

**Visualization**: Cyan, dashed lines

**Use case**: "What happened recently?" or understanding sequences of events

#### 2. Semantic Links (Meaning-Based)
**Purpose**: Connect memories with similar content/meaning

**How it works**:
- Generate embeddings using local `bge-small-en-v1.5` model (384 dimensions)
- Store embeddings in PostgreSQL with pgvector extension
- When storing a new memory, find top-k similar memories using cosine similarity
- Create links only if similarity exceeds threshold (default: 0.7)
- Weight = cosine similarity score

**Visualization**: Pink, solid lines

**Technology**:
- **SentenceTransformers** - Local embedding model (BAAI/bge-small-en-v1.5)
- **pgvector** - PostgreSQL extension for vector operations
- **HNSW index** - Fast approximate nearest neighbor search

**Use case**: "Tell me about hiking" retrieves all semantically related outdoor activities

#### 3. Entity Links (Identity-Based)
**Purpose**: Connect ALL memories about the same person, organization, or place

**How it works**:
- Extract entities from text using spaCy NER
- Resolve entity identity using disambiguation algorithm:
  - Name similarity (50% weight) - using SequenceMatcher
  - Co-occurring entities (30% weight) - entities that appear together
  - Temporal proximity (20% weight) - recent mentions more likely same entity
- If score > threshold (0.4 for PERSON with exact match, 0.6 otherwise): reuse existing entity
- If score < threshold: create new entity
- Link all memories mentioning the same entity with weight 1.0 (no decay)

**Visualization**: Gold, thick lines

**Technology**:
- **spaCy** (`en_core_web_sm`) - Named Entity Recognition
- **difflib.SequenceMatcher** - String similarity matching

**Use case**: "What does Alice do?" returns ALL memories about Alice (hiking, work at Google, Python project) even if semantically distant

**Critical advantage**: Solves the problem where "Alice loves hiking" wouldn't normally connect to "Alice works at Google" through semantic similarity alone.

### Spreading Activation Search

The search algorithm explores the memory graph using spreading activation:

1. **Entry Points**: Find top-3 semantically similar memories to the query (vector search, similarity ≥ 0.5)
2. **Activation Spreading**: Start with activation = actual similarity score (0.5 to 1.0) at entry points
3. **Graph Traversal**: Follow links to neighbors, spreading activation with decay (0.8 factor)
4. **Thinking Budget**: Limit exploration to N units (controls computational cost)
5. **Dynamic Weighting**: Combine activation, semantic similarity, recency, and frequency:
   ```
   final_weight = 0.30 × activation + 0.30 × semantic_similarity + 0.25 × recency + 0.15 × frequency

   semantic_similarity = cosine_similarity(query_embedding, memory_embedding)
   recency = exp(-0.1 × days_since)
   frequency = normalized to [0, 1] from log(access_count + 1) / log(10)
   ```
6. **Return Top-K**: Sort by final weight and return top results

This approach ensures:
- Semantic relevance to query is always considered (30% weight)
- Graph structure influences results through activation (30% weight)
- Recently accessed memories get boosted (25% weight - recency bias)
- Frequently accessed memories get boosted (15% weight - importance signal)

### Self-Contained Memory Units

Every memory unit is self-contained through LLM fact extraction:

**Problem**: "She joined Google last year" - unclear who "she" is

**Solution**: LLM-based fact extraction that:
- Resolves pronouns to actual referents during extraction
- Makes facts readable without original context
- Includes all relevant details (WHO, WHAT, WHERE, WHEN, WHY, HOW)
- Processes facts in parallel for speed

**Result**: "Alice joined Google last year" - fully self-contained

**Technology**:
- LLM fact extraction with detailed prompts for pronoun resolution
- Structured output using Pydantic models
- Batch processing for efficiency

### LLM-Based Fact Extraction

Raw content is processed through an LLM to extract meaningful facts before storage:

**Problem**: Raw text contains noise (greetings, filler words, reactions) that waste storage and reduce retrieval quality

**Solution**: LLM-based extraction with optimized prompting:
- Filters out social pleasantries and non-informative content
- Extracts only facts with substance (biographical, events, opinions, recommendations, descriptions, relationships)
- Creates self-contained statements with subject+action+context
- Categorizes and attributes facts to speakers

**Technology**:
- **OpenAI-compatible API** - Supports Groq (default), OpenAI, and other providers
- **Structured output** - Uses Pydantic models for reliable fact extraction
- **Optimized prompting** - Concise prompts (~300 chars) emphasize dense output with no fluff
- **Automatic chunking** - Large documents (>120k chars) split at sentence boundaries
- **Fast sentence splitting** - Regex-based splitter (no heavy NLP models)
- **Progress tracking** - Logs chunk processing for transparency

**For large documents (e.g., podcast transcripts)**:
- Documents <120k chars: processed in one pass
- Documents >120k chars: automatically chunked at sentence boundaries
- Each chunk kept under ~30k tokens to avoid output token limits
- Facts aggregated across all chunks

### Technology Stack

**Database**:
- PostgreSQL 15+ with extensions:
  - `pgvector` - Vector similarity operations
  - `uuid-ossp` - UUID generation

**Python Libraries**:
- `psycopg2-binary` - PostgreSQL client
- `sentence-transformers` - Local embedding model (bge-small-en-v1.5)
- `torch` - Deep learning framework (for embeddings)
- `spacy` - NLP (NER, dependency parsing, tokenization)
- `langchain-text-splitters` - Intelligent text chunking
- `networkx` - Graph operations
- `pyvis` - Interactive HTML graph visualization
- `matplotlib` - Static graph visualization
- `rich` - Terminal UI

**Models**:
- BAAI/bge-small-en-v1.5 - Local embedding model (384 dimensions)

## Quick Start

### Prerequisites

1. PostgreSQL 15+ with pgvector extension
2. Python 3.11+

### Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install spaCy model:
   ```bash
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

3. Create database and run schema:
   ```bash
   psql -U postgres -c "CREATE DATABASE memory_poc"
   psql -U postgres -d memory_poc -f schema.sql
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your DATABASE_URL
   ```

### Run Tests

Run the full test suite:
```bash
uv run pytest tests/ -v
```

Run specific test files:
```bash
uv run pytest tests/test_memory_operations.py -v
uv run pytest tests/test_entity_linking.py -v
```

Run a single test:
```bash
uv run pytest tests/test_memory_operations.py::test_put_creates_memory_units -v
```

### Run Demo

```bash
uv run python demos/demo_entity.py
```

This will:
1. Clear previous demo data
2. Store sample memories about Alice, Bob, Google, Yosemite
3. Search for "What does Alice do?"
4. Show entity resolution results
5. Generate interactive HTML graph visualization

Open `memory_graph_interactive.html` in your browser to explore the memory graph!

## Project Structure

```
memory-poc/
├── memory/                          # Core memory system package
│   ├── temporal_semantic_memory.py  # Main memory system class
│   ├── entity_resolver.py           # Entity extraction and disambiguation
│   ├── llm_client.py                # LLM-based fact extraction
│   └── utils.py                     # Utility functions
│
├── demos/                           # Demo scripts
│   └── demo_entity.py              # Main entity-aware demo
│
├── visualizations/                  # Visualization tools
│   └── interactive_graph.py        # Interactive HTML graph (pyvis)
│
├── schema.sql                       # Database schema
├── pyproject.toml                  # Dependencies
└── README.md                       # This file
```

## Key Features

✅ **Three-layered linking**: Temporal + Semantic + Entity
✅ **Entity disambiguation**: Resolves "Alice" across different contexts
✅ **Self-contained units**: Pronouns resolved to actual referents
✅ **Spreading activation**: Graph-aware search beyond vector similarity
✅ **Interactive visualization**: Explore memory graph in browser
✅ **Recency & frequency weighting**: Recent and important memories boosted
✅ **Linguistic validation**: Memory units verified to have subject + verb

## API Usage

### Store Memories

```python
from memory import TemporalSemanticMemory

memory = TemporalSemanticMemory()

memory.put(
    agent_id="agent_1",
    content="Alice works at Google as a software engineer. She joined last year.",
    context="Career discussion",
    event_date=datetime.now(timezone.utc)
)
```

### Search Memories

```python
results = memory.search(
    agent_id="agent_1",
    query="What does Alice do?",
    thinking_budget=50,  # How many units to explore
    top_k=10             # Number of results to return
)

for result in results:
    print(f"{result['text']} (weight: {result['weight']:.3f})")
```

## How It Works: Example

**Input memories**:
1. "Alice loves hiking in the mountains" (7 days ago)
2. "She goes hiking every weekend in Yosemite" (7 days ago)
3. "Alice works at Google as a software engineer" (3 days ago)
4. "She joined Google last year" (3 days ago)

**Processing**:
1. ✅ Coreference resolution → "Alice goes hiking...", "Alice joined Google..."
2. ✅ Entity extraction → Identifies "Alice" (PERSON), "Google" (ORG), "Yosemite" (GPE)
3. ✅ Entity resolution → All "Alice" mentions = same person
4. ✅ Create links:
   - Temporal: Memory 1 ↔ Memory 2 (same day)
   - Semantic: "hiking" memories link together, "Google" memories link together
   - Entity: ALL Alice memories strongly linked (weight 1.0)

**Query: "What does Alice do?"**
1. Vector search finds "Alice works at Google" as top entry point
2. Spreading activation follows entity links to find:
   - "Alice joined Google..." (entity link: Alice)
   - "Alice loves hiking..." (entity link: Alice)
   - "Alice goes hiking..." (entity link: Alice)
3. Returns ALL Alice memories, properly ranked by relevance

## Why This Architecture?

**Problem with vector-only search**: "Alice loves hiking" and "Alice works at Google" are semantically distant - pure vector search might miss this connection.

**Solution**: Entity links ensure memories about the same person/place/organization are strongly connected regardless of semantic distance.

**Result**: More human-like memory retrieval that understands identity and relationships.

## License

MIT
