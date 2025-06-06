# 🧠 RAG-Powered PDF Knowledge Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying PDF documents using LLMs, dense vector search, and cross-encoder re-ranking.

It supports:
- Loading and chunking PDFs
- Embedding using Sentence Transformers
- Retrieval using ChromaDB
- Query expansion via Ollama (Mistral)
- Re-ranking with HuggingFace Cross-Encoder
- Final response generation using Ollama LLM

---

## 🚀 Features

- 📄 PDF ingestion and intelligent chunking (sentence + character level)
- 🔍 Query expansion for semantic enrichment
- 💡 Fast & accurate document retrieval using ChromaDB
- 🎯 Cross-Encoder-based reranking for high-precision passage selection
- 🤖 Final response generated using local LLM via [Ollama](https://ollama.com/)

---


---

## 📦 Installation

### 1. Clone this repo

```bash
git clone https://github.com/MELVIN10101/Advance-Retrieval--For-AI-Chroma.git
cd Advance-Retrieval--For-AI-Chroma
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Mistral Model
```bash
ollama pull mistral
```

### 4. Run the pipeline:
```bash
python3 retrieval.py
```


## CONTACT ME:
portfolio (Desktop site) : https://melvin-cyberops-portfolio.vercel.app