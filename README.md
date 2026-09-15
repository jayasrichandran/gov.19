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



Key Features
Government document-based RAG
PDF text extraction with page preservation
Semantic document chunking
Sentence Transformer embeddings
FAISS vector similarity search
Gemini-powered grounded answer generation
Source and page-level citation tracking
40-question evaluation benchmark
Automated RAG evaluation
Scheme-wise performance comparison
Question-level analysis
Failure analysis
Interactive Streamlit dashboard
JSON-based evaluation dataset and results
Government Schemes

The current benchmark uses four government-service documents:

PMFBY — Pradhan Mantri Fasal Bima Yojana
PM SVANidhi — PM Street Vendor's AtmaNirbhar Nidhi
PMUY 2.0 — Pradhan Mantri Ujjwala Yojana 2.0
AYUSH Health & Wellness Centres

The evaluation benchmark contains 40 questions, with 10 questions for each scheme.

The questions cover:

Scheme information
Eligibility
Benefits
Required documents
Application process
Scheme-specific details
Unanswerable questions for testing unsupported-answer behavior
Evaluation Metrics
1. Retrieval Relevance

Measures whether the retrieved document chunks contain information relevant to the question.

2. Answer Correctness
Measures whether the generated answer correctly answers the question when compared with the reference answer.

3. Groundedness

Measures whether the generated answer is supported by the retrieved evidence.

4. Citation Correctness

Measures whether the cited government document and page actually support the generated answer.

Technology Stack
Component	Technology
Programming Language	Python
User Interface	Streamlit
PDF Processing	PyMuPDF
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Search	FAISS
LLM	Google Gemini API
Data Storage	JSON
Visualization	Plotly
RAG Pipeline	Custom Python Implementation
Evaluation	Custom Python Evaluation Engine
Project Structure
gov.19/
│
├── data/
│   ├── evaluation/
│   │   ├── evaluation_dataset.json
│   │   └── evaluation_results.json
│   │
│   ├── processed/
│   │   ├── PMFBY.txt
│   │   ├── PMSVAN.txt
│   │   ├── PMUY2.0.txt
│   │   └── AYUSHHWC.txt
│   │
│   └── vector_db/
│       ├── faiss_index.bin
│       └── chunks_metadata.json
│
├── src/
│   ├── process_pdfs.py
│   ├── chunk_documents.py
│   ├── build_vector_db.py
│   ├── rag_pipeline.py
│   ├── evaluation_engine.py
│   └── dashboard.py
│
├── requirements.txt
├── .gitignore
├── .env
└── README.md
Installation
1. Clone the Repository
git clone https://github.com/jayasrichandran/gov.19.git
cd gov.19
2. Install Dependencies

For Windows:

py -m pip install -r requirements.txt

For Linux/macOS:

python3 -m pip install -r requirements.txt
Gemini API Configuration

The project uses the Google Gemini API for grounded answer generation.

Create a .env file in the root directory of the project:

GEMINI_API_KEY=your_gemini_api_key_here

The structure should be:

gov.19/
├── .env
├── requirements.txt
├── README.md
└── src/
Security

Do not upload your .env file or Gemini API key to GitHub.

Make sure .gitignore contains:

.env
venv/
__pycache__/
*.pyc
Run the Application

Once the dependencies are installed and the .env file is configured, start the Streamlit dashboard:

py -m streamlit run src/dashboard.py

Streamlit will display a local URL similar to:

http://localhost:8501

Open the displayed URL in your browser.

Quick Start

If the processed documents, vector database, and evaluation results are already available in the repository:

git clone https://github.com/jayasrichandran/gov.19.git
cd gov.19
py -m pip install -r requirements.txt

Create the .env file:

GEMINI_API_KEY=your_gemini_api_key_here

Then run:

py -m streamlit run src/dashboard.py

Open:

http://localhost:8501
Complete Pipeline

To rebuild the complete RAG pipeline from the source documents, run the following commands in order.

Step 1 — Process PDFs
py src/process_pdfs.py

This extracts text from the government PDF documents while preserving page information.

Step 2 — Create Document Chunks
py src/chunk_documents.py

This divides the processed documents into smaller chunks suitable for semantic retrieval.

Output:

data/processed/chunks.json
Step 3 — Build the FAISS Vector Database
py src/build_vector_db.py

This creates semantic embeddings using:

sentence-transformers/all-MiniLM-L6-v2

and stores the embeddings in FAISS.

Outputs:

data/vector_db/faiss_index.bin
data/vector_db/chunks_metadata.json
Step 4 — Test the RAG Pipeline

Example:

py src/rag_pipeline.py --query "What is the maximum farmer premium for Kharif crops?"

The RAG pipeline:

Converts the question into an embedding
Searches the FAISS vector database
Retrieves the most relevant document chunks
Sends the retrieved evidence to Gemini
Generates a grounded answer
Returns source document and page information
Step 5 — Run Evaluation

Run the evaluation engine against the 40-question benchmark:

py src/evaluation_engine.py

The evaluation results are stored in:

data/evaluation/evaluation_results.json
Step 6 — Launch Dashboard
py -m streamlit run src/dashboard.py

Open:

http://localhost:8501
Evaluation Dataset

The evaluation dataset is stored at:

data/evaluation/evaluation_dataset.json

It contains 40 benchmark questions:

10 PMFBY questions
10 PM SVANidhi questions
10 PMUY 2.0 questions
10 AYUSH HWC questions

Each evaluation record contains:

id
scheme
question
reference_answer
supporting_evidence
source_document
source_page
question_type
answerability

Some questions are intentionally unanswerable. These test whether the RAG system avoids generating unsupported information when the required information is not present in the provided documents.

Dashboard

The Streamlit dashboard provides:

Overall Reliability

Displays the overall performance across the four evaluation metrics.

Evaluation Summary

Shows the total benchmark questions and evaluation status.

Scheme Comparison

Compares the performance of:

PMFBY
PM SVANidhi
PMUY 2.0
AYUSH HWC
Question-Level Results

Allows individual benchmark questions to be inspected.

Question Details

Displays:

Reference answer
Generated answer
Retrieved chunks
Evaluation scores
Source citations
Failure Analysis

Helps identify why the RAG system failed, including:

Retrieval failures
Incorrect answers
Unsupported answers
Citation issues
Current Benchmark Results

Based on the current 40-question benchmark:

Answer Correctness: 73.1%
AYUSH HWC Reliability: 94.9%
PMUY 2.0 Reliability: 85.9%
PMUY 2.0 Pass Rate: 70%

These results represent the current benchmark and RAG configuration.

Why GOV-19?

A RAG system should not be evaluated only by whether it can generate an answer.

GOV-19 evaluates whether the system:

Retrieved relevant information
          ↓
Generated a correct answer
          ↓
Stayed grounded in evidence
          ↓
Provided correct citations

This transforms RAG reliability from a subjective judgment into measurable evidence.

Future Improvements
Hybrid keyword + semantic retrieval
Retrieval reranking
Improved chunking strategies
Larger evaluation datasets
Additional government schemes
Comparison of multiple RAG configurations
Improved hallucination detection
Advanced retrieval failure diagnosis
More comprehensive citation verification
Project Repository

GitHub:

https://github.com/jayasrichandran/gov.19

see i have to exactly as it is in readme so thatr give full correct content include packge installation content also
GOV-19 — Public Service RAG Evaluation System

GOV-19 is a RAG evaluation system designed to measure the quality, reliability, and trustworthiness of Retrieval-Augmented Generation (RAG) systems for government-service information.

Instead of only checking whether an AI system generates an answer, GOV-19 evaluates whether the answer is retrieved from relevant information, factually correct, grounded in source evidence, and supported by correct citations.

Problem Statement

Government-service RAG systems can generate answers that appear correct but may contain incomplete, unsupported, or incorrectly cited information.

GOV-19 provides a measurable framework for evaluating these systems using four core metrics:

Retrieval Relevance — Did the system retrieve the right information?
Answer Correctness — Is the generated answer correct?
Groundedness — Is the answer supported by the retrieved evidence?
Citation Correctness — Do the cited document and page support the answer?
Key Features
Government scheme PDF processing
Page-aware document extraction
Semantic document chunking
Sentence Transformer embeddings
FAISS vector similarity search
Gemini-powered grounded answer generation
Source and page-level citation tracking
40-question evaluation benchmark
Automated RAG evaluation
Scheme-wise performance comparison
Question-level analysis
Failure analysis
Interactive Streamlit dashboard
JSON-based evaluation dataset and results
Government Schemes

The current benchmark uses four government-service documents:

PMFBY — Pradhan Mantri Fasal Bima Yojana
PM SVANidhi — PM Street Vendor's AtmaNirbhar Nidhi
PMUY 2.0 — Pradhan Mantri Ujjwala Yojana 2.0
AYUSH Health & Wellness Centres

The evaluation benchmark contains 40 questions, with 10 questions for each scheme.

The questions cover:

Scheme information
Eligibility
Benefits
Required documents
Application process
Scheme-specific details
Unanswerable questions for testing unsupported-answer behavior
RAG Pipeline
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
Evaluation Metrics
1. Retrieval Relevance

Measures whether the retrieved document chunks contain information relevant to the question.

2. Answer Correctness

Measures whether the generated answer correctly answers the question when compared with the reference answer.

3. Groundedness

Measures whether the generated answer is supported by the retrieved evidence.

4. Citation Correctness

Measures whether the cited government document and page actually support the generated answer.

Technology Stack
Component	Technology
Programming Language	Python
User Interface	Streamlit
PDF Processing	PyMuPDF
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Search	FAISS
LLM	Google Gemini API
Data Storage	JSON
Visualization	Plotly
RAG Pipeline	Custom Python Implementation
Evaluation	Custom Python Evaluation Engine
Project Structure
gov.19/
│
├── data/
│   ├── evaluation/
│   │   ├── evaluation_dataset.json
│   │   └── evaluation_results.json
│   │
│   ├── processed/
│   │   ├── PMFBY.txt
│   │   ├── PMSVAN.txt
│   │   ├── PMUY2.0.txt
│   │   └── AYUSHHWC.txt
│   │
│   └── vector_db/
│       ├── faiss_index.bin
│       └── chunks_metadata.json
│
├── src/
│   ├── process_pdfs.py
│   ├── chunk_documents.py
│   ├── build_vector_db.py
│   ├── rag_pipeline.py
│   ├── evaluation_engine.py
│   └── dashboard.py
│
├── requirements.txt
├── .gitignore
├── .env
└── README.md
Installation
Prerequisites

Make sure the following are installed:

Python 3.10 or later
Git
Internet connection for installing Python packages
Google Gemini API key

Check Python:

py --version

Check Git:

git --version
1. Clone the Repository
git clone https://github.com/jayasrichandran/gov.19.git

Move into the project directory:

cd gov.19
2. Create a Virtual Environment
Windows
py -m venv venv

Activate the environment:

venv\Scripts\activate
Linux / macOS
python3 -m venv venv

Activate the environment:

source venv/bin/activate
3. Upgrade pip
Windows
py -m pip install --upgrade pip
Linux / macOS
python3 -m pip install --upgrade pip
4. Install Required Packages

The project dependencies are listed in requirements.txt.

Install all required packages with:

Windows
py -m pip install -r requirements.txt
Linux / macOS
python3 -m pip install -r requirements.txt

The main packages used by the project are:

streamlit
plotly
PyMuPDF
sentence-transformers
faiss-cpu
torch
google-genai
python-dotenv
numpy
5. Configure the Gemini API Key

Create a file named:

.env

in the root directory of the project.

Add:

GEMINI_API_KEY=your_gemini_api_key_here

Project structure:

gov.19/
│
├── .env
├── requirements.txt
├── README.md
├── src/
└── data/
Important Security Note

Do not upload the .env file or your Gemini API key to GitHub.

Make sure .gitignore contains:

.env
venv/
__pycache__/
*.pyc
Running the Project
Quick Start

If the processed files, FAISS vector database, and evaluation results are already available:

git clone https://github.com/jayasrichandran/gov.19.git
cd gov.19
py -m pip install -r requirements.txt

Create and configure the .env file:

GEMINI_API_KEY=your_gemini_api_key_here

Then start the dashboard:

py -m streamlit run src/dashboard.py

Streamlit will display a local address such as:

Local URL: http://localhost:8501

Open the following URL in your browser:

http://localhost:8501

Start the Dashboard

Use:

py -m streamlit run src/dashboard.py

The dashboard provides:

Overall Reliability
Retrieval Relevance
Answer Correctness
Groundedness
Citation Correctness
Scheme Comparison
Question-Level Results
Reference Answers
Generated Answers
Retrieved Evidence
Source Citations
Failure Analysis
Complete RAG Pipeline

If you want to rebuild the complete pipeline from the source documents, run the following commands in order.

Step 1 — Process PDFs
py src/process_pdfs.py

This extracts text from the government PDF documents while preserving page information.

Step 2 — Create Document Chunks
py src/chunk_documents.py

This divides the processed documents into smaller chunks suitable for semantic retrieval.

Output:

data/processed/chunks.json
Step 3 — Build the FAISS Vector Database
py src/build_vector_db.py

This creates semantic embeddings using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings are stored in FAISS.

Outputs:

data/vector_db/faiss_index.bin
data/vector_db/chunks_metadata.json
Step 4 — Test the RAG Pipeline

Example:

py src/rag_pipeline.py --query "What is the maximum farmer premium for Kharif crops?"

The RAG pipeline:

Converts the question into an embedding.
Searches the FAISS vector database.
Retrieves the most relevant document chunks.
Sends the retrieved evidence to Gemini.
Generates a grounded answer.
Returns source document and page information.
Step 5 — Run Evaluation

Run the evaluation engine against the 40-question benchmark:

py src/evaluation_engine.py

The evaluation results are stored in:

data/evaluation/evaluation_results.json
Step 6 — Launch the Dashboard
py -m streamlit run src/dashboard.py

Open:

http://localhost:8501
Evaluation Dataset

The evaluation dataset is stored at:

data/evaluation/evaluation_dataset.json

It contains 40 benchmark questions:

10 PMFBY questions
10 PM SVANidhi questions
10 PMUY 2.0 questions
10 AYUSH HWC questions

Each evaluation record contains:

id
scheme
question
reference_answer
supporting_evidence
source_document
source_page
question_type
answerability

Some questions are intentionally unanswerable. These test whether the RAG system avoids generating unsupported information when the required information is not present in the provided documents.

Dashboard

The Streamlit dashboard contains the following sections.

Overall Reliability

Displays the overall evaluation performance across the four metrics.

Evaluation Summary

Shows the total benchmark questions and evaluation status.

Scheme Comparison

Compares the performance of:

PMFBY
PM SVANidhi
PMUY 2.0
AYUSH HWC
Question-Level Results

Allows individual benchmark questions to be inspected.

Question Details

Displays:

Reference answer
Generated answer
Retrieved chunks
Evaluation scores
Source citations
Failure Analysis

Helps identify why the RAG system failed, including:

Retrieval failures
Incorrect answers
Unsupported answers
Citation issues
Current Benchmark Results

Based on the current 40-question benchmark:

Answer Correctness: 73.1%
AYUSH HWC Reliability: 94.9%
PMUY 2.0 Reliability: 85.9%
PMUY 2.0 Pass Rate: 70%

These results represent the current benchmark and RAG configuration.

Why GOV-19?

A RAG system should not be evaluated only by whether it can generate an answer.

GOV-19 evaluates whether the system:

Retrieved relevant information
          ↓
Generated a correct answer
          ↓
Stayed grounded in evidence
          ↓
Provided correct citations

This transforms RAG reliability from a subjective judgment into measurable evidence.

Future Improvements
Hybrid keyword + semantic retrieval
Retrieval reranking
Improved chunking strategies
Larger evaluation datasets
Additional government schemes
Comparison of multiple RAG configurations
Improved hallucination detection
Advanced retrieval failure diagnosis
More comprehensive citation verification
Repository

GitHub Repository:

https://github.com/jayasrichandran/gov.19

License

This project was developed as part of the GOV-19 hackathon challenge for demonstrating RAG evaluation, reliability analysis, and grounded AI for government-service information.


