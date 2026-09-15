# GOV-19 — Public Service RAG Evaluation System

GOV-19 is a measurable evaluation framework for assessing the quality, reliability, and trustworthiness of Retrieval-Augmented Generation (RAG) systems for government-service information.

Instead of only checking whether an AI system generates an answer, GOV-19 evaluates whether the answer is retrieved from relevant information, factually correct, grounded in source evidence, and supported by correct citations.

## Problem Statement

Government-service RAG systems can generate answers that appear correct but may contain incomplete, unsupported, or incorrectly cited information.

GOV-19 provides a measurable framework for evaluating these systems using four core metrics:

- **Retrieval Relevance** — Did the system retrieve the right information?
- **Answer Correctness** — Is the generated answer correct?
- **Groundedness** — Is the answer supported by the retrieved evidence?
- **Citation Correctness** — Do the cited document and page support the answer?

## Solution

The system processes government scheme documents, converts them into searchable semantic representations, retrieves relevant information for a question, generates a grounded response using an LLM, and evaluates the response using four reliability metrics.

```text
Government PDFs
       ↓
PDF Text Extraction
       ↓
Cleaning + Page Preservation
       ↓
Document Chunking
       ↓
Sentence Transformer Embeddings
       ↓
FAISS Vector Database
       ↓
Question Embedding
       ↓
Top-K Retrieval
       ↓
Grounded Gemini Response
       ↓
Answer + Source Metadata
       ↓
Evaluation Engine
       ↓
4 Evaluation Metrics
       ↓
Streamlit Dashboard

