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
- **Incremental Indexing**: Re-indexes only files that changed since the last run. New, modified, and deleted files are handled automatically. Unchanged files are skipped entirely, making re-indexing fast enough for CI or unit tests.
- **Hybrid Search**: Combines semantic (vector) search via ChromaDB with keyword search via SQLite FTS5, merged with reciprocal rank fusion (RRF). Natural-language queries are preprocessed for FTS5 — stop words stripped, remaining terms OR-joined — so a single word absent from the corpus doesn't zero out keyword results.
- **Pluggable Embeddings**: Defaults to `nomic-embed-text-v1.5` running locally via Ollama. Any callable matching the `EmbeddingFunction` protocol can be substituted.
- **Struct–Method Linking**: Automatically wires methods to their parent struct via `parent_id` and `children_ids` for hierarchical traversal.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) running locally with `nomic-embed-text-v1.5` pulled:

```bash
ollama pull nomic-embed-text:v1.5
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure Repositories

Edit `configs/repositories.json` to list the Go repositories you want to index:

```json
{
  "repositories": [
    {
      "name": "my-project",
      "url": "https://github.com/example/my-project",
      "branch": "main",
      "clone_path": "data/repos/my-project"
    }
  ]
}
```

`clone_path` is relative to the directory you run `ingest.py` from (typically the project root). Cloned repositories land under `data/repos/`, which is excluded from version control via `.gitignore`.

### Run Ingestion

```bash
python scripts/ingest.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/repositories.json` | Path to repository config |
| `--chroma-path` | `data/vector_store/chroma` | ChromaDB storage directory |
| `--sqlite-path` | `data/vector_store/index.db` | SQLite database path |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--model` | `nomic-embed-text:v1.5` | Embedding model name |
| `--summarize` | disabled | Generate LLM summaries for chunks lacking documentation |
| `--chat-model` | `devstral-small-2:latest` | Chat model for summarization |
| `--verbose`, `-v` | disabled | Print detailed progress including summarization |

Re-running `ingest.py` is safe and efficient — only changed files are re-indexed.

#### LLM Summarization

The `--summarize` flag enables automatic generation of summaries for chunks that lack good documentation (empty or short `doc` field, unnamed blocks). This can significantly improve retrieval quality for poorly documented code.

```bash
# Enable summarization with verbose output
python scripts/ingest.py --summarize --verbose

# Use a different chat model
python scripts/ingest.py --summarize --chat-model llama3:8b
```

**Note:** Summarization adds an LLM call per chunk that needs it, which slows down ingestion. For large codebases, consider running with `--summarize` only when needed.

### Query the Index

```bash
python scripts/query.py "how does authentication work"
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--mode` | `hybrid` | Search mode: `hybrid`, `semantic`, or `keyword` |
| `--top-k` | `5` | Number of results to return |
| `--repo` | *(all repos)* | Filter results to a specific repository name |
| `--include-tests` | `false` | Include test code chunks in results (excluded by default) |
| `--include-low-quality` | `false` | Include low-quality chunks in results (excluded by default) |
| `--chroma-path` | `data/vector_store/chroma` | ChromaDB storage directory |
| `--sqlite-path` | `data/vector_store/index.db` | SQLite database path |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--model` | `nomic-embed-text:v1.5` | Embedding model name |

Examples:

```bash
# Hybrid search (default) — best of semantic + keyword
python scripts/query.py "error handling middleware"

# Semantic only
python scripts/query.py "user authentication" --mode semantic --top-k 10

# Keyword only — no Ollama required; queries are preprocessed for FTS5
python scripts/query.py "FindUserByID" --mode keyword

# Filter to one repo
python scripts/query.py "context propagation" --repo my-project

# Include test code in results
python scripts/query.py "test helpers" --include-tests

# Include low-quality chunks (e.g., single-letter names, common identifiers)
python scripts/query.py "error handling" --include-low-quality
```

### Inspecting Chunks

`show.py` lets you browse the raw content of indexed chunks from ChromaDB and SQLite without truncation. Useful for debugging retrieval misses or reviewing what got indexed.

```bash
python scripts/show.py --repo viya-sonder --name WalkFunc
```

At least one filter is required. Filters can be combined (AND logic):

| Flag | Description |
|---|---|
| `--id ID` | Exact chunk ID |
| `--name STR` | Substring match on symbol name |
| `--repo NAME` | Exact repository name |
| `--file STR` | Substring match on file path |
| `--type TYPE` | Chunk type: `package`, `block`, `function`, `method`, `struct`, `interface`, `const`, `var`, `type_alias` |
| `--package STR` | Substring match on package name |
| `--include-low-quality` | Include low-quality chunks (excluded by default) |
| `--limit N` | Max chunks to display (default: 20) |

Examples:

```bash
# All functions in a test file
python scripts/show.py --repo viya-sonder --file mux_test.go --type function

# All type aliases in a repo
python scripts/show.py --repo viya-sonder --type type_alias

# Look up a chunk directly by ID (e.g. from evaluation miss output)
python scripts/show.py --id 4186ed572baa1c6f475336a915f3426b1089601579b183892d6a1f53a8d38b5e

# Include low-quality chunks in results (excluded by default)
python scripts/show.py --repo viya-sonder --type var --include-low-quality
```

Each result shows the full untruncated content from ChromaDB, plus the `doc` and `signature` fields from SQLite.

By default, low-quality chunks are excluded from results. Low-quality chunks have generic names (e.g., `err`, `_`, `ctx`, single-letter variables) that are nearly impossible to retrieve accurately. Use `--include-low-quality` to include them in results.

### Evaluating Retrieval Quality

Cencio includes an evaluation harness that measures how well the search pipeline retrieves the right code chunks. It uses a **golden set** — a JSON file of natural-language queries paired with the chunk each query is expected to find.

#### Generate the golden set

The golden set is generated automatically by sampling chunks from the index and prompting a local chat model to write a realistic search query for each one. This requires the integration test index to already exist (run `pytest tests/integration/` first).

```bash
python scripts/generate_golden_set.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--repo` | all from config | Repository name(s) to include (can be specified multiple times) |
| `--chat-model` | `devstral-small-2:latest` | Ollama chat model used to write queries |
| `--embed-model` | `nomic-embed-text:v1.5` | Ollama embedding model |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--samples-per-type` | `5` | Chunks sampled per chunk type |
| `--output` | `tests/evaluation/golden.json` | Output path |
| `--seed` | `42` | Random seed for reproducible sampling |

Any Ollama chat model works. Larger or code-focused models (e.g. `devstral-small-2`, `qwen3`) produce more precise queries and a more demanding golden set.

The generated file is committed to the repository so the baseline is stable across runs. Re-generate only when you want to refresh the evaluation set (e.g. after a schema change or to add more queries).

#### Run the evaluation

```bash
python scripts/evaluate.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--golden-set` | `tests/evaluation/golden.json` | Path to the golden set |
| `--mode` | `hybrid` | Search mode: `hybrid`, `semantic`, or `keyword` |
| `--top-k` | `5` | Number of results retrieved per query |
| `--keyword-weight` | `1.0` | RRF weight for keyword results relative to semantic (hybrid mode only) |
| `--embed-model` | `nomic-embed-text:v1.5` | Ollama embedding model |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |

Examples:

```bash
# Evaluate hybrid search (default)
python scripts/evaluate.py

# Compare semantic search at top-10
python scripts/evaluate.py --mode semantic --top-k 10

# Evaluate keyword-only (no Ollama required)
python scripts/evaluate.py --mode keyword
```

#### Interpreting the report

```
============================================================
  Evaluation Report  —  mode=hybrid  top_k=5
============================================================
  Queries :  45
  Hit rate:  84%  (38/45 found in top-5)
  MRR     :  0.701
  ...
```

**Hit rate** — the percentage of queries where the expected chunk appeared somewhere in the top-k results. This measures recall: did we find the right answer at all?

**MRR (Mean Reciprocal Rank)** — the average of 1/rank for each query. A result ranked #1 scores 1.0, #2 scores 0.5, #3 scores 0.33, and so on. Misses score 0. This measures precision: not just whether we found it, but how high it was ranked. MRR closer to 1.0 means the right chunk consistently appears near the top.

**How to use the numbers:**

- Run the evaluation before and after a change (parser improvement, embedding strategy, RRF tuning). A higher hit rate and MRR indicate the change helped.
- The per-type breakdown shows which chunk types are hardest to retrieve — useful for targeting parser or embedding improvements.
- The "Misses" section lists every query that returned no correct result, with the query text, so you can inspect whether the query was ambiguous or the retrieval genuinely failed.

### MCP Server

The MCP server exposes Cencio's hybrid search as a tool that AI assistants can call directly during a conversation. Once configured, you can ask an assistant to "find the authentication handler" or "show me error wrapping patterns" and it will query your index without you copying or pasting anything.

#### Running the server

The server communicates over stdio and is started by the client — you don't run it manually. To verify it works before wiring it up:

```bash
python scripts/mcp_server.py
```

It will block waiting for MCP protocol messages. `Ctrl-C` to exit. If it exits immediately with an error, check that Ollama is running and the index exists.

#### Environment variables

All configuration is via environment variables. Every variable has a default that works out of the box after running `ingest.py` from the project root.

| Variable | Default | Description |
|---|---|---|
| `CENCIO_CHROMA_PATH` | `data/vector_store/chroma` | Absolute or relative path to the ChromaDB directory |
| `CENCIO_SQLITE_PATH` | `data/vector_store/index.db` | Absolute or relative path to the SQLite FTS5 database |
| `CENCIO_OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `CENCIO_EMBED_MODEL` | `nomic-embed-text:v1.5` | Embedding model name as registered in Ollama |

The defaults are relative to the project root and match `ingest.py`'s defaults, so no configuration is needed if you run the server from the Cencio directory. Set the path variables to absolute paths when configuring the server inside a client — the client may launch the process from a different working directory.

#### Tool reference

The server exposes one tool:

**`search_code`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | *(required)* | Natural-language or keyword search query |
| `mode` | string | `hybrid` | `hybrid`, `semantic`, or `keyword` |
| `repo` | string | *(all repos)* | Repository name to restrict results to |
| `top_k` | integer | `5` | Number of results (capped at 20) |

#### Configuring clients

Replace `/path/to/cencio` with the absolute path to this repository and `/path/to/cencio/.venv/bin/python` with the absolute path to the Python interpreter in your virtual environment. Use `which python` inside the activated venv to find the exact path.

---

**Claude Code**

Add to your user-level config (`~/.claude/settings.json`) to make the tool available in every project:

```json
{
  "mcpServers": {
    "cencio": {
      "command": "/path/to/cencio/.venv/bin/python",
      "args": ["/path/to/cencio/scripts/mcp_server.py"],
      "env": {
        "CENCIO_CHROMA_PATH": "/path/to/cencio/data/vector_store/chroma",
        "CENCIO_SQLITE_PATH": "/path/to/cencio/data/vector_store/index.db"
      }
    }
  }
}
```

Or add to `.claude/settings.json` in a specific project to scope it to that workspace.

---

**GitHub Copilot (VS Code)**

Create `.vscode/mcp.json` in the workspace root (or add to your VS Code user `settings.json` under the `mcp.servers` key):

```json
{
  "servers": {
    "cencio": {
      "type": "stdio",
      "command": "/path/to/cencio/.venv/bin/python",
      "args": ["/path/to/cencio/scripts/mcp_server.py"],
      "env": {
        "CENCIO_CHROMA_PATH": "/path/to/cencio/data/vector_store/chroma",
        "CENCIO_SQLITE_PATH": "/path/to/cencio/data/vector_store/index.db"
      }
    }
  }
}
```

MCP tools are available in Copilot Chat when using **Agent mode** (`@workspace` with the agent toggle enabled).

---

**Zed**

Add to `~/.config/zed/settings.json`:

```json
{
  "context_servers": {
    "cencio": {
      "command": {
        "path": "/path/to/cencio/.venv/bin/python",
        "args": ["/path/to/cencio/scripts/mcp_server.py"],
        "env": {
          "CENCIO_CHROMA_PATH": "/path/to/cencio/data/vector_store/chroma",
          "CENCIO_SQLITE_PATH": "/path/to/cencio/data/vector_store/index.db"
        }
      },
      "settings": {}
    }
  }
}
```

After saving, open the Assistant panel and the `search_code` tool will appear in the context server list. Zed restarts context servers automatically when settings change.

### Running Tests

Run all unit tests:
```bash
.venv/bin/pytest
```

Run ingestion tests only:
```bash
.venv/bin/pytest tests/unit/ingestion/
```

Run parsing tests only:
```bash
.venv/bin/pytest tests/unit/parsing/
```

### Running Integration Tests

Integration tests index a real repository (gorilla/mux) and require:
- Network access to clone from GitHub (`git@github.com:gorilla/mux.git`)
- Ollama running locally with `nomic-embed-text:v1.5` pulled

```bash
ollama pull nomic-embed-text:v1.5
.venv/bin/pytest tests/integration/
```

On the first run, gorilla/mux is cloned to `build/data/repos/gorilla-mux` and indexed into `build/database/`. Subsequent runs reuse both, so only changed files are re-indexed and the embedding step is skipped entirely.

To force a full re-clone and re-index (e.g. after a schema change):
```bash
.venv/bin/pytest tests/integration/ --clean
```

## 📂 Project Structure

```text
cencio/
├── build/                        # Runtime artifacts (gitignored)
│   ├── data/repos/               # Repositories cloned by integration tests
│   └── database/                 # ChromaDB and SQLite index for integration tests
├── configs/
│   └── repositories.json     # List of Go repos to index (name, url, branch, clone_path)
├── data/
│   ├── repos/                # Cloned git repositories (created at runtime)
│   └── vector_store/         # ChromaDB and SQLite index (created at runtime)
├── scripts/
│   ├── ingest.py                 # CLI: clone/pull repos and run incremental indexing
│   ├── query.py                  # CLI: search the index (hybrid, semantic, or keyword)
│   ├── mcp_server.py             # MCP server: exposes search_code tool over stdio
│   ├── generate_golden_set.py    # CLI: generate the retrieval evaluation golden set
│   ├── evaluate.py               # CLI: run evaluation and print hit rate + MRR report
│   └── show.py                   # CLI: inspect indexed chunks from ChromaDB and SQLite
├── src/
│   ├── embedding/
│   │   └── ollama.py             # EmbeddingFunction protocol + OllamaEmbeddingFunction
│   ├── evaluation/
│   │   ├── ollama_chat.py        # ChatFunction protocol + OllamaChatFunction
│   │   ├── golden_set.py         # Golden set generation and persistence
│   │   └── harness.py            # Evaluation logic: hit rate and MRR
│   ├── ingestion/
│   │   ├── indexer.py            # Incremental indexing orchestration
│   │   ├── repository.py         # GitRepository: clone/pull, list Go files, content hashing
│   │   └── store.py              # ChunkStore: ChromaDB + SQLite FTS5 + file tracking
│   ├── models/
│   │   └── chunk.py              # Chunk dataclass and ChunkType enum
│   └── parsing/
│       └── go_parser.py          # tree-sitter based Go parser
└── tests/
    ├── data/
    │   └── golang/               # Go source fixtures for parser tests
    ├── evaluation/
    │   └── golden.json           # Committed golden set for evaluation
    ├── integration/              # End-to-end tests (require Ollama + network)
    └── unit/
        ├── embedding/            # Tests for OllamaEmbeddingFunction
        ├── ingestion/            # Tests for repository, store, and indexer
        └── parsing/              # Tests for the Go parser
```

## 🛠 Built With
- Python 3.12+
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) + [tree-sitter-go](https://github.com/tree-sitter/tree-sitter-go)
- [ChromaDB](https://www.trychroma.com) — vector store for semantic search
- SQLite FTS5 — keyword search (Python stdlib, no extra dependencies)
- [Ollama](https://ollama.com) — local embedding model server
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — MCP server for AI assistant integration

---
*Clean code. Clean context. Cencio.*
