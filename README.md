(.venv) Zoro@zoro:~/Code/rag-learning$ cat README.md
# RAG Learning — Local Retrieval-Augmented Generation System

A hands-on implementation of a **Retrieval-Augmented Generation (RAG)** system built from the ground up using Python, sentence embeddings, vector search, ChromaDB, and a local LLM.

The project is being developed incrementally to understand how modern RAG systems work internally rather than relying completely on high-level frameworks.

Each major development stage is maintained as a separate Git version so the evolution of the project can be tracked.

---

# 🚀 Project Introduction

The purpose of this project is to build and understand a complete local RAG pipeline step by step.

The system takes documents, converts them into numerical embeddings, stores those embeddings in a vector store, retrieves relevant information for a user's question, and provides that information to a local LLM to generate a grounded answer.

The overall pipeline is:

```text
Documents
    ↓
Document Ingestion
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Storage
    ↓
Semantic Retrieval
    ↓
Context Construction
    ↓
Local LLM
    ↓
Grounded Answer
```

The project currently contains two major implementations:

* **v0.1** — Pickle vector store + local Qwen LLM
* **v0.2** — Persistent ChromaDB semantic retrieval

The next major stage is connecting **ChromaDB + retrieval + local Qwen LLM** into a complete RAG pipeline.

---

# ✨ Features

* 📄 Load text documents
* ✂ Split documents into chunks
* 🧠 Generate sentence embeddings
* 🔎 Perform semantic similarity search
* 📦 Store embeddings using Pickle
* 🗄 Store vectors using persistent ChromaDB
* 🤖 Run a local Qwen LLM
* 🔗 Build RAG prompts using retrieved context
* 📊 Inspect retrieval scores/distances
* 🏷 Maintain versioned Git milestones
* 💻 Designed to run locally

---

# 🛠 Technology Stack

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Main programming language           |
| Sentence Transformers | Embedding generation                |
| all-MiniLM-L6-v2      | Embedding model                     |
| ChromaDB              | Persistent vector database          |
| scikit-learn          | Similarity calculations             |
| Qwen 2.5 0.5B         | Local language model                |
| llama.cpp             | Local LLM inference                 |
| Pickle                | Initial vector-store implementation |
| Git                   | Version control                     |
| GitHub                | Source-code hosting                 |

---

# ⚙ Installation

## 1. Clone the repository

```bash
git clone https://github.com/shanmukha443/rag-learning.git
cd rag-learning
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
```

## 3. Activate the virtual environment

### Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install sentence-transformers scikit-learn chromadb llama-cpp-python
```

---

# 📂 Project Structure

```text
rag-learning/
│
├── data/
│   └── documents/
│       ├── git.txt
│       ├── linux.txt
│       └── python.txt
│
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
│
├── scripts/
│   ├── check_changes.py
│   └── search_documents.py
│
├── src/
│   ├── embeddings/
│   │   └── embedding.py
│   │
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── ingestion/
│   │   ├── build_index.py
│   │   └── chunking.py
│   │
│   ├── rag/
│   │   └── rag.py
│   │
│   └── retrieval/
│       ├── chroma_index.py
│       ├── chroma_search.py
│       ├── chunk_search.py
│       ├── search_index.py
│       └── semantic_search.py
│
├── tests/
│   └── rag_promt_test.py
│
├── main.py
├── .gitignore
└── README.md
```

---

# ▶ How to Run

## v0.1 — Pickle Vector Store

Build the embedding index:

```bash
python src/ingestion/build_index.py
```

Expected output:

```text
Index created successfully!
Total chunks: 6
Embedding dimensions: 384
Saved to: vector_store.pkl
```

Run the RAG pipeline:

```bash
python src/rag/rag.py
```

Example:

```text
Ask a question: What package manager does Fedora use?
```

---

## v0.2 — ChromaDB

Build the ChromaDB index:

```bash
python src/retrieval/chroma_index.py
```

Expected output:

```text
ChromaDB index created successfully!
Total chunks: 6
Embedding dimensions: 384
Collection: rag_documents
```

Run semantic search:

```bash
python src/retrieval/chroma_search.py
```

Example:

```text
Ask a question: What programming language is commonly used for data engineering?
```

The system retrieves the most semantically relevant chunks from ChromaDB.

---

# 🧠 How RAG Works

RAG stands for **Retrieval-Augmented Generation**.

Instead of asking an LLM to answer a question using only its internal knowledge, the system first retrieves relevant information from a knowledge source.

That retrieved information is then provided to the LLM as context.

```text
User Question
      ↓
Question Embedding
      ↓
Vector Search
      ↓
Relevant Chunks
      ↓
Context
      ↓
RAG Prompt
      ↓
Local LLM
      ↓
Answer
```

The important idea is:

> **Retrieve first, generate second.**

---

# 📦 Vector Storage Evolution

## v0.1 — Pickle

The first implementation manually stores embeddings using Python Pickle.

```text
Documents
    ↓
Embeddings
    ↓
vector_store.pkl
    ↓
Load into Python
    ↓
Cosine Similarity
    ↓
Top-K Results
```

This version was useful for understanding the fundamentals of vector retrieval.

---

## v0.2 — ChromaDB

The second implementation introduces persistent ChromaDB storage.

```text
Documents
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Query
    ↓
Top-K Results
```

ChromaDB handles persistent vector storage and retrieval, giving the project a stronger foundation for future RAG development.

---

# 🤖 Local LLM

The project uses a local Qwen model:

```text
qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The model is executed locally using `llama-cpp-python`.

The long-term objective is to keep the RAG pipeline locally runnable without requiring a paid cloud LLM API.

---

# 🌳 Git Version History

Important development milestones are stored using Git tags.

## v0.1

```text
v0.1-rag-pickle-llm
```

Implemented:

```text
Pickle Vector Store
        +
Cosine Similarity Retrieval
        +
RAG Prompt
        +
Local Qwen LLM
```

## v0.2

```text
v0.2-chromadb
```

Implemented:

```text
Persistent ChromaDB
        +
Semantic Retrieval
```

## v0.3 — Upcoming

```text
v0.3-chromadb-llm
```

Planned:

```text
ChromaDB
    ↓
Semantic Retrieval
    ↓
Context Construction
    ↓
RAG Prompt
    ↓
Qwen Local LLM
    ↓
Final Answer
```

---

# 🔮 Future Roadmap

## Retrieval

* ⬜ Improve chunking strategy
* ⬜ Add configurable chunk size
* ⬜ Add chunk overlap
* ⬜ Experiment with different embedding models
* ⬜ Improve retrieval thresholding
* ⬜ Evaluate retrieval quality

## Document Ingestion

* ⬜ Support PDF documents
* ⬜ Support Markdown documents
* ⬜ Support DOCX documents
* ⬜ Automatically detect modified documents
* ⬜ Automatically re-index changed documents
* ⬜ Handle deleted documents
* ⬜ Improve document metadata

## RAG

* ⬜ Connect ChromaDB to the RAG pipeline
* ⬜ Improve prompt engineering
* ⬜ Add source citations
* ⬜ Return source document names
* ⬜ Improve context management
* ⬜ Reduce hallucinations
* ⬜ Improve "I don't know" handling
* ⬜ Add retrieval evaluation

## Local LLM

* ⬜ Experiment with larger models
* ⬜ Compare different Qwen models
* ⬜ Tune generation parameters
* ⬜ Add streaming responses
* ⬜ Improve context-window management
* ⬜ Measure inference performance

## Application

* ⬜ Create a command-line interface
* ⬜ Add interactive chat mode
* ⬜ Add conversation history
* ⬜ Build a FastAPI backend
* ⬜ Build a web interface
* ⬜ Add API endpoints

## Engineering

* ⬜ Add configuration management
* ⬜ Add structured logging
* ⬜ Improve error handling
* ⬜ Add unit tests
* ⬜ Add integration tests
* ⬜ Add retrieval evaluation tests
* ⬜ Add GitHub Actions CI/CD
* ⬜ Add architecture diagrams
* ⬜ Add performance benchmarks

---

# 📋 Implementation Checklist

## v0.1 — Pickle Vector Store + Local LLM

**Status: ✅ Completed**

* ✅ Load documents
* ✅ Document chunking
* ✅ Generate embeddings
* ✅ Understand embedding dimensions
* ✅ Store embeddings using Pickle
* ✅ Implement cosine similarity
* ✅ Implement Top-K retrieval
* ✅ Add similarity threshold
* ✅ Build RAG prompt
* ✅ Integrate local Qwen LLM
* ✅ Generate answers from retrieved context
* ✅ Create Git commit
* ✅ Create Git tag `v0.1-rag-pickle-llm`
* ✅ Push to GitHub

---

## v0.2 — ChromaDB Semantic Retrieval

**Status: ✅ Completed**

* ✅ Install ChromaDB
* ✅ Create persistent ChromaDB client
* ✅ Create `rag_documents` collection
* ✅ Generate document embeddings
* ✅ Store chunks in ChromaDB
* ✅ Store document metadata
* ✅ Implement persistent vector storage
* ✅ Implement semantic search
* ✅ Implement Top-K retrieval
* ✅ Display retrieval distances
* ✅ Test retrieval across documents
* ✅ Create Git commit
* ✅ Create Git tag `v0.2-chromadb`
* ✅ Push to GitHub

---

## v0.3 — ChromaDB + Local LLM RAG

**Status: ⬜ Pending**

* ⬜ Connect ChromaDB retrieval to RAG
* ⬜ Replace Pickle retrieval with ChromaDB
* ⬜ Generate query embeddings
* ⬜ Retrieve Top-K chunks
* ⬜ Build context from retrieved chunks
* ⬜ Create controlled RAG prompt
* ⬜ Connect context to Qwen
* ⬜ Generate final answer
* ⬜ Test multiple questions
* ⬜ Test multiple documents
* ⬜ Improve answer grounding
* ⬜ Prevent unsupported claims
* ⬜ Add configurable Top-K
* ⬜ Add configurable retrieval threshold
* ⬜ Test complete end-to-end pipeline
* ⬜ Create Git commit
* ⬜ Create Git tag `v0.3-chromadb-llm`
* ⬜ Push to GitHub
* ⬜ Update README

---

# 📊 Current Progress

```text
v0.1 — Pickle + Local LLM
██████████████████████████████ 100%

v0.2 — ChromaDB Semantic Retrieval
██████████████████████████████ 100%

v0.3 — ChromaDB + Local LLM
░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%

Advanced RAG
░░░░░░░░░░░░░░░░░░░░░░░░░░░░   Planned
```

---

# 🎯 Long-Term Goal

The final goal is to build a complete, well-engineered local RAG application while understanding each major component instead of treating RAG as a black box.

```text
Document Ingestion
        ↓
Chunking
        ↓
Embedding Generation
        ↓
Vector Database
        ↓
Semantic Retrieval
        ↓
Context Management
        ↓
Prompt Engineering
        ↓
Local LLM
        ↓
Evaluation
        ↓
Application
```

The project will continue evolving through versioned Git milestones.

---

# 👨‍💻uthor / Credits

**Shanmukha Murthy**

GitHub: `shanmukha443`

This project is built as a hands-on learning project focused on:

* Retrieval-Augmented Generation
* Embeddings
* Semantic Search
* Vector Databases
* ChromaDB
* Local LLMs
* Prompt Engineering
* Python
* Git and GitHub
(.venv) Zoro@zoro:~/Code/rag-learning$ 
