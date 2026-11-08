# Hybrid Search RAG (Vector + BM25)

A RAG system over a research paper that combines **semantic (vector) search** and **keyword (BM25) search** with an ensemble retriever, then answers questions with a Groq-hosted LLM.

The document used is the original RAG paper: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)). Download the PDF and place it next to the notebook (it is not included in this repo).

## What it does
1. Loads the PDF with `PyPDFLoader` and splits it into 200-character chunks (30 overlap).
2. Embeds chunks with `BAAI/bge-base-en-v1.5` and indexes them in **Chroma**.
3. Builds a **BM25** keyword retriever over the same chunks.
4. Combines both with `EnsembleRetriever` (weights: 0.3 vector / 0.7 BM25).
5. Compares vector-only, BM25-only, and hybrid results for the same query (e.g. *"What is RAG-Token?"*).
6. Feeds the hybrid results into a retrieval chain to generate the final answer.

## Stack
`LangChain` · `Groq` · `Chroma` · `BM25` · `Weaviate client` · `BGE embeddings` · `PyTorch`

## Run it
Open `Hybrid_search.ipynb` in Google Colab and add these secrets (Colab → Secrets):

| Secret | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM answers |
| `HF_TOKEN` | HuggingFace access |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | Weaviate Cloud connection |

Never commit keys to the repo.
