# Cencio 🇮🇹

**Pronounced: *CHEN-cho***

> *"A rag for your codebase."*

## The Meaning
In Italian, **Cencio** refers to a rag or a piece of cloth used for wiping and cleaning.

## The Problem
Large Go codebases are messy. When you're trying to augment an LLM with context from a massive repository, you often end up with "noise"—irrelevant functions, boilerplate, and disconnected snippets that dilute the precision of your prompts.

## The Solution
**Cencio** is a specialized RAG (Retrieval-Augmented Generation) pipeline designed specifically to "clean up" your Go codebase. 

Instead of just dumping raw text into a vector database, Cencio understands the structure of Go. It indexes your code with an awareness of package hierarchies, interfaces, and function signatures, ensuring that when you ask a piece of code a question, the context provided to your LLM is surgically precise.

## ✨ Key Features
- **Go-Native Parsing**: Uses tree-sitter to semantically parse Go source files into typed chunks — packages, functions, methods, structs, interfaces, consts, vars, blocks, and type aliases.
- **Noise Reduction**: Filters out the "lint" of the codebase, leaving only the meaningful logic.
- **Semantic Precision**: Uses advanced embedding techniques to ensure retrieval is based on intent, not just keywords.
- **Context-Aware Retrieval**: Pulls in relevant dependencies and interface implementations to provide a complete picture.
- **Struct–Method Linking**: Automatically wires methods to their parent struct via `parent_id` and `children_ids` for hierarchical traversal.

## 🚀 Getting Started

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

Run all tests:
```bash
.venv/bin/pytest
```

Run only parsing tests:
```bash
.venv/bin/pytest tests/unit/parsing/
```

Run a specific test file:
```bash
.venv/bin/pytest tests/unit/parsing/test_package.py
```

## 📂 Project Structure

```text
cencio/
├── configs/          # Configuration files for the pipeline and indexing
├── data/             # Data files for the pipeline
│   ├── processed/    # Processed and cleaned text data
│   ├── raw/          # Raw, unprocessed source code and data
│   └── vector_store/ # Vector embeddings and searchable index
├── docs/             # Project documentation and guides
├── notebooks/        # Jupyter notebooks for experimentation and prototyping
├── scripts/          # Automation and maintenance scripts
│   ├── ingest.py     # Script for ingesting and indexing code
│   └── query.py      # Script for querying the RAG pipeline
├── src/              # Core source code
│   ├── embedding/    # Logic for creating embeddings
│   ├── evaluation/   # Evaluation frameworks and metrics
│   ├── generation/   # LLM generation logic
│   ├── models/       # Model definitions and configurations
│   │   └── chunk.py  # Chunk dataclass and ChunkType enum
│   ├── parsing/      # Go source file parsing
│   │   └── go_parser.py  # tree-sitter based Go parser
│   ├── processing/   # Text processing and cleaning utilities
│   ├── retrieval/    # Retrieval logic and algorithms
│   └── utils/        # Common utility functions
└── tests/            # Test suites
    ├── data/
    │   └── golang/   # Go source fixtures for parser tests
    ├── evaluation/   # Tests for evaluation components
    ├── integration/  # Integration tests for the full pipeline
    └── unit/
        └── parsing/  # Unit tests for the Go parser
```

## 🛠 Built With
- Python 3.12+
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [tree-sitter-go](https://github.com/tree-sitter/tree-sitter-go)

---
*Clean code. Clean context. Cencio.*
