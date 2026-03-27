# BootDEV RAG Course

A full implementation of a Retrieval-Augmented Generation (RAG) pipeline for movie search, built as part of the [Boot.dev](https://boot.dev) "Build a RAG in Python" course.

## Overview

This project builds a movie search system from scratch, starting with basic keyword search and progressively adding semantic search, hybrid search, re-ranking, and LLM-powered generation. The dataset consists of thousands of movies with titles and descriptions.

## Features

### Keyword Search (`cli/keyword_search_cli.py`)
- **BM25 search** using an inverted index with TF-IDF weighting
- **IDF lookup** for individual terms
- **Index building** with stemming and stopword removal

### Semantic Search (`cli/semantic_search_cli.py`)
- **Dense vector search** using the `all-MiniLM-L6-v2` sentence transformer model
- **Chunked semantic search** using `ChunkedSemanticSearch` with sentence-level chunking and overlap
- **Query enhancement** via Gemini API (spell correction, rewriting, expansion)
- **Text chunking** with configurable chunk size and overlap

### Hybrid Search (`cli/hybrid_search_cli.py`)
- **Weighted hybrid search** combining normalized BM25 and semantic scores with a configurable alpha
- **Reciprocal Rank Fusion (RRF)** combining BM25 and semantic ranks
- **Re-ranking** with three methods:
  - `individual` -- one LLM call per document
  - `batch` -- single LLM call for all documents
  - `cross_encoder` -- local cross-encoder model (`ms-marco-TinyBERT-L2-v2`)
- **LLM evaluation** of search result quality (0-3 scale)
- **Score normalization** utility

### Augmented Generation (`cli/augmented_generation_cli.py`)
- **RAG pipeline** -- retrieves documents and generates a natural language answer
- **Multi-document summarization** -- synthesizes information across multiple results
- **Citation-aware answers** -- references sources with `[1]`, `[2]` notation
- **Conversational Q&A** -- casual, direct answers to movie questions

### Multimodal Search (`cli/multimodal_search_cli.py`)
- **Image embedding** using the CLIP model (`clip-ViT-B-32`)
- **Image-based search** -- find movies by uploading an image
- **Multimodal query rewriting** -- combine image + text into a search query via Gemini

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/StefanosGZ/BootDEV-RAG_course.git
cd BootDEV-RAG_course
uv sync
```

### Environment Variables

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Build the Search Index

Before running searches, build the inverted index and embeddings:

```bash
# Build BM25 index
uv run cli/keyword_search_cli.py build

# Build semantic embeddings
uv run cli/semantic_search_cli.py verify_embeddings

# Build chunked embeddings
uv run cli/semantic_search_cli.py embed_chunks
```

## Usage

### Keyword Search

```bash
uv run cli/keyword_search_cli.py search "bear in london"
uv run cli/keyword_search_cli.py idf "bear"
```

### Semantic Search

```bash
uv run cli/semantic_search_cli.py search "romantic comedy"
uv run cli/semantic_search_cli.py search_chunked "space adventure" --limit 5
```

### Hybrid Search

```bash
# Weighted hybrid
uv run cli/hybrid_search_cli.py weighted-search "family bear movie" --alpha 0.5 --limit 5

# RRF
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --limit 5

# RRF with query enhancement
uv run cli/hybrid_search_cli.py rrf-search "famly moive abuot bears" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "sad movie" --enhance rewrite

# RRF with re-ranking
uv run cli/hybrid_search_cli.py rrf-search "bears in the woods" --limit 3 --rerank-method cross_encoder
uv run cli/hybrid_search_cli.py rrf-search "bears in the woods" --limit 3 --rerank-method batch

# RRF with evaluation
uv run cli/hybrid_search_cli.py rrf-search "bear movie" --limit 5 --evaluate
```

### RAG Generation

```bash
uv run cli/augmented_generation_cli.py rag "what are some good dinosaur movies?"
uv run cli/augmented_generation_cli.py summarize "dinosaur movies" --limit 5
uv run cli/augmented_generation_cli.py citations "best sci-fi movies"
uv run cli/augmented_generation_cli.py question "what should I watch if I like dinosaurs?"
```

### Multimodal Search

```bash
# Search by image
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Rewrite query using image
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "a bear in london"

# Verify image embedding
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
```

## Project Structure

```
bootdev-rag/
├── cli/
│   ├── lib/
│   │   ├── augmented_generation.py   # RAG generation functions
│   │   ├── hybrid_search.py          # HybridSearch class, RRF, weighted search
│   │   ├── invertedindex.py          # InvertedIndex with BM25
│   │   ├── keyword_search.py         # BM25 search, tokenization
│   │   ├── multimodal_search.py      # CLIP-based image search
│   │   ├── search_utils.py           # Shared utilities, load_movies
│   │   └── semantic_search.py        # SemanticSearch, ChunkedSemanticSearch
│   ├── augmented_generation_cli.py
│   ├── describe_image_cli.py
│   ├── hybrid_search_cli.py
│   ├── keyword_search_cli.py
│   ├── multimodal_search_cli.py
│   └── semantic_search_cli.py
├── cache/                            # Cached embeddings and index (git-ignored)
├── data/
│   ├── movies.json                   # Movie dataset
│   └── paddington.jpeg               # Sample image for multimodal search
├── .env                              # API keys (git-ignored)
└── pyproject.toml
```

## Technologies

- **sentence-transformers** -- text and image embeddings
- **NLTK** -- tokenization, stemming, stopwords
- **NumPy** -- vector operations
- **Google Gemini API** -- LLM generation, query enhancement, re-ranking
- **Pillow** -- image loading for multimodal search
- **uv** -- Python package management
