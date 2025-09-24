# Contextual Retrieval System - Educational Implementation

An educational implementation of Anthropic's Contextual Retrieval technique, demonstrating how contextualizing chunks before indexing dramatically improves retrieval accuracy in RAG systems.

## 🌟 Key Insight

**The Problem**: Traditional RAG systems lose context when chunking documents. A chunk saying "The company's revenue grew by 3%" loses meaning without knowing which company or time period.

**The Solution**: Contextual Retrieval prepends chunk-specific explanatory context to each chunk before embedding and indexing, preserving semantic meaning.

## 📚 Educational Features

This implementation includes extensive logging and comparison capabilities to understand:

1. **How Context Generation Works**: Watch the LLM generate context for each chunk
2. **Dual Indexing Strategy**: See how both BM25 and embeddings benefit from context
3. **Comparison Mode**: Run with `use_contextual=False` to compare against standard chunking
4. **Performance Metrics**: Track improvements in retrieval accuracy
5. **Cost Analysis**: Understand the token usage and costs

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Document Input                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          Basic Chunking                 │
│   (Respects paragraph boundaries)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     Context Generation (Optional)       │
│         Using LLM API                   │
│   (Enabled with use_contextual=True)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Enhanced Chunks                    │
│  • Contextual: Context + Original Text  │
│  • Standard: Original Text Only         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Retrieval Pipeline Indexing        │
│   • Sparse Index (BM25)                 │
│   • Dense Index (Embeddings)            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Hybrid Search with Reranking         │
│   Combines BM25 + Embedding scores      │
│   Cross-encoder reranking for accuracy  │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env

# Edit .env and add your API keys:
# - MOONSHOT_API_KEY for Kimi
# - ARK_API_KEY for Doubao
# - OPENAI_API_KEY for OpenAI
# - etc.
```

### 3. Start the Retrieval Pipeline

```bash
# In a separate terminal, start the retrieval pipeline server
cd ../retrieval-pipeline
python main.py
# Server will run on http://localhost:4242
```

### 4. Index Documents

```bash
# Index Chinese law documents with contextual enhancement
python index_local_laws_contextual.py

# Or index without contextual enhancement for comparison
python index_local_laws_contextual.py --no-contextual
```

### 5. Run Queries

```bash
# Interactive mode with contextual retrieval
python main.py

# Query with specific mode
python main.py --query "宪法第一条是什么" --mode agentic

# Compare agentic vs non-agentic modes
python main.py --query "宪法第一条是什么" --mode compare
```

## 📖 Detailed Usage

### Context Generation Process

The system generates context for each chunk by:

1. **Providing the full document** (or surrounding context) to the LLM
2. **Showing the specific chunk** to be contextualized
3. **Asking for concise context** (2-3 sentences) that situates the chunk

Example prompt template:
```
<document>
[Full document or surrounding context]
</document>

Here is the chunk we want to situate:
<chunk>
[Specific chunk text]
</chunk>

Please give a short, succinct context to situate this chunk within the overall document...
```

### Indexing Documents

```python
from contextual_chunking import ContextualChunker
from config import Config

# Initialize with contextual enhancement
config = Config.from_env()
chunker = ContextualChunker(
    chunking_config=config.chunking,
    llm_config=config.llm,
    use_contextual=True  # Set to False for standard chunking
)

# Chunk and contextualize document
chunks = chunker.chunk_document(
    text=document_text,
    doc_id="doc_1",
    doc_metadata={"source": "example.pdf"}
)

# Each chunk will have:
# - chunk.text: Original text
# - chunk.context: Generated context (if contextual=True)
# - chunk.contextualized_text: Context + original text
```

### Searching with the Agentic RAG System

```python
from agent import AgenticRAG
from config import Config

# Initialize agent
config = Config.from_env()
agent = AgenticRAG(config)

# Query with agentic approach (uses tools)
response = agent.query(
    "What is the first article of the constitution?",
    stream=False
)

# Or use non-agentic approach (direct retrieval)
response = agent.query_non_agentic(
    "What is the first article of the constitution?",
    stream=False
)
```

### Comparing Contextual vs Standard Chunking

```python
# Index with contextual enhancement
indexer_contextual = LocalLawsContextualIndexer(
    use_contextual=True
)
indexer_contextual.process_all_documents()

# Index without contextual enhancement
indexer_standard = LocalLawsContextualIndexer(
    use_contextual=False
)
indexer_standard.process_all_documents()

# Compare retrieval quality
# The system will show metrics for both approaches
```

## 📊 Example Results

Based on Anthropic's research, contextual retrieval shows:

| Method | Retrieval Failure Rate | Improvement |
|--------|------------------------|-------------|
| Standard Embeddings | 5.7% | Baseline |
| Contextual Embeddings | 3.7% | 35% reduction |
| Contextual Embeddings + BM25 | 2.9% | 49% reduction |
| + Reranking | 1.9% | 67% reduction |

## 🔍 Understanding the Logs

The system provides detailed educational logging:

### Context Generation Logs
```
2025-09-24 23:03:08,365 - INFO - Generating context for chunk 1/10
2025-09-24 23:03:09,245 - INFO - Context generated: "This section defines the fundamental nature of the People's Republic of China as a socialist state..."
2025-09-24 23:03:09,246 - INFO - Indexed chunk constitution_chunk_0 immediately
```

### Retrieval Logs
```
2025-09-24 23:03:08,365 - INFO - Knowledge base search initiated - Type: LOCAL, Query: '宪法第一条'
2025-09-24 23:03:08,365 - INFO - Searching local knowledge base
2025-09-24 23:03:09,524 - INFO - Local search returned 3 results
2025-09-24 23:03:09,525 - INFO - Tool result: {"status": "success", "results": [...]
```

## 🎓 Educational Insights

### When Context Helps Most

1. **Ambiguous References**: "The company", "it", "this method"
2. **Technical Terms**: Context provides domain information
3. **Temporal Information**: Dates and time periods
4. **Hierarchical Documents**: Section and subsection context
5. **Multi-topic Documents**: Distinguishes between topics

### Trade-offs

| Aspect | Contextual | Standard |
|--------|------------|----------|
| **Indexing Time** | Slower (LLM calls) | Fast |
| **Indexing Cost** | ~$1 per million tokens | Free |
| **Storage** | Larger (context added) | Smaller |
| **Retrieval Quality** | High | Moderate |
| **Best For** | Production, accuracy-critical | Prototyping, cost-sensitive |

## 🛠️ Configuration Options

### Chunking Parameters
```python
config.chunking.chunk_size = 2048  # Characters per chunk
config.chunking.max_chunk_size = 1024  # Maximum chunk size
config.chunking.chunk_overlap = 200  # Overlap between chunks
config.chunking.respect_paragraph_boundary = True  # Preserve paragraphs
```

### Context Generation
```python
config.llm.provider = "openai"  # LLM for context generation
config.llm.model = "gpt-3.5-turbo"  # Model choice
config.llm.temperature = 0.3  # Lower = more consistent
```

### Search Methods
- **BM25**: Best for exact matches, technical terms
- **Embedding**: Best for semantic similarity
- **Hybrid**: Combines both using rank fusion

## 📈 Performance Optimization

### Caching Strategy
The system caches generated contexts to avoid regenerating for identical chunks:
```python
Cache hit rate: 45% (saving ~$0.45 per 1000 chunks)
```

### Batch Processing
Process multiple documents efficiently:
```bash
python batch_index.py --input documents/ --output indexes/
```

## 🔬 Experimental Features

### Multi-level Context
Generate context at different granularities:
- Chunk-level context (current)
- Section-level context
- Document-level summary

### Dynamic Context Length
Adjust context length based on chunk complexity:
- Simple chunks: 1-2 sentences
- Complex chunks: 3-4 sentences

## 📚 References

- [Anthropic's Contextual Retrieval Blog Post](https://www.anthropic.com/engineering/contextual-retrieval)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

## 🤝 Contributing

This is an educational implementation. Contributions welcome for:
- Additional chunking strategies
- Alternative context generation prompts
- Performance optimizations
- Evaluation metrics
- Visualization tools

## 📝 License

Educational project for learning purposes.

## 🙏 Acknowledgments

Based on research by Anthropic's engineering team on improving RAG retrieval accuracy through contextual enhancement.