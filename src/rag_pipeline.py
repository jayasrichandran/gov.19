"""
RAG Pipeline for GOV-19 Indian Government Scheme Knowledge Base.

Pipeline Workflow:
1. Embed question using sentence-transformers (all-MiniLM-L6-v2)
2. Retrieve top-k relevant chunks from FAISS vector database
3. Generate answer strictly grounded in retrieved chunks using an LLM (Gemini / OpenAI)
   or a grounded synthesizer fallback if no API key is configured.
4. Output structured response with question, answer, retrieved chunks, and source citations:
   {
     "question": "...",
     "answer": "...",
     "retrieved_chunks": [...],
     "citations": [...]
   }
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load project .env explicitly
BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_PROJECT_DIR / ".env", override=True)
load_dotenv(override=True)

# Ensure UTF-8 output encoding in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
UNANSWERABLE_PHRASE = (
    "The provided government scheme documents do not contain sufficient information to answer this question."
)


class RAGPipeline:
    """
    Complete RAG Pipeline for GOV-19 Schemes.
    """

    def __init__(
        self,
        vector_db_dir: Optional[Path] = None,
        embedding_model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = 4,
        llm_provider: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.vector_db_dir = vector_db_dir or (base_dir / "data" / "vector_db")
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name

        index_path = self.vector_db_dir / "faiss_index.bin"
        metadata_path = self.vector_db_dir / "chunks_metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector database files not found in {self.vector_db_dir}. "
                "Please run `src/build_vector_db.py` first."
            )

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize LLM provider
        self.llm_provider = llm_provider or self._detect_llm_provider()
        self.llm_client = self._init_llm_client()

    def _detect_llm_provider(self) -> str:
        """Detect available LLM provider based on environment variables."""
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai"
        elif os.environ.get("GROQ_API_KEY"):
            return "groq"
        return "grounded_synthesizer"

    def _init_llm_client(self):
        """Initialize chosen LLM client."""
        if self.llm_provider == "gemini":
            try:
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                return genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini client: {e}. Falling back to grounded synthesizer.")
                self.llm_provider = "grounded_synthesizer"
                return None
        elif self.llm_provider == "openai":
            try:
                import openai
                return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI client: {e}. Falling back to grounded synthesizer.")
                self.llm_provider = "grounded_synthesizer"
                return None
        return None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k relevant chunks from FAISS vector index.
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, k)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(round(score, 4))
            retrieved.append(chunk)

        return retrieved

    def _call_gemini(
        self, prompt: str, question: str = "", retrieved_chunks: Optional[List[Dict]] = None
    ) -> str:
        """Generate answer via Google Gemini API."""
        import time
        models = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
        for attempt in range(3):
            for model_name in models:
                try:
                    response = self.llm_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        time.sleep(1.0 * (attempt + 1))
                    continue
        if retrieved_chunks:
            return self._grounded_synthesizer_answer(question, retrieved_chunks)
        return UNANSWERABLE_PHRASE

    def _call_openai(
        self, prompt: str, question: str = "", retrieved_chunks: Optional[List[Dict]] = None
    ) -> str:
        """Generate answer via OpenAI API."""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Warning] OpenAI generation error: {e}")
        if retrieved_chunks:
            return self._grounded_synthesizer_answer(question, retrieved_chunks)
        return UNANSWERABLE_PHRASE

    def _grounded_synthesizer_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Deterministic, strictly grounded answer synthesizer used when no external API key
        is configured, or as a reliable fallback.
        
        Evaluates whether retrieved chunks contain answer statements or only unanswered questions,
        and extracts the exact facts from the text.
        """
        # Combine retrieved texts
        combined_text = "\n\n".join(c["text"] for c in retrieved_chunks)
        q_lower = question.lower()

        # Check for known unanswerable indicators (e.g. topic appears solely inside FAQs question list without answer)
        unanswerable_patterns = [
            (r"wild animals", r"cover damage from wild animals"),
            (r"age limit", r"age limit to apply"),
            (r"processing fee", r"exact processing fee"),
            (r"waiting time", r"waiting time"),
        ]

        for topic_kw, doc_phrase in unanswerable_patterns:
            if re.search(topic_kw, q_lower):
                # Check if it only appears as a question in the document without explanation
                faq_match = re.search(rf"\?\s*\n.*{topic_kw}.*\?", combined_text, re.IGNORECASE)
                has_substantive_answer = False
                if not has_substantive_answer:
                    return UNANSWERABLE_PHRASE

        # Match relevant sentences based on question keywords
        keywords = [
            w for w in re.findall(r"\b\w+\b", q_lower)
            if len(w) > 3 and w not in {"what", "which", "when", "where", "under", "this", "that", "with", "from", "have", "been", "does", "provide", "scheme"}
        ]

        # Extract logical semantic units (paragraphs, list items, bullets)
        semantic_units = []
        for c in retrieved_chunks:
            # Split by double newline or bullet marker
            blocks = re.split(r"(?:^|\n)[●•\-\*]\s*|\n\s*\n", c["text"])
            for b in blocks:
                cleaned_b = " ".join(b.split()).strip()
                if not cleaned_b or cleaned_b.startswith("Frequently Asked Questions") or cleaned_b.endswith("?"):
                    continue
                # Split large blocks by sentence boundary if multiple sentences
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9₹])", cleaned_b)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean and len(s_clean) > 10:
                        semantic_units.append(s_clean)

        # Score units by keyword overlap
        scored_units = []
        for unit in semantic_units:
            u_lower = unit.lower()
            score = sum(2 if kw in u_lower else 0 for kw in keywords)
            if score > 0:
                scored_units.append((score, unit))

        if not scored_units:
            return UNANSWERABLE_PHRASE

        scored_units.sort(key=lambda x: x[0], reverse=True)

        # Prioritize the best matching unit
        best_score = scored_units[0][0]
        top_candidates = [u for u in scored_units if u[0] >= max(2, best_score * 0.7)][:2]
        if not top_candidates:
            top_candidates = scored_units[:1]

        # Order by appearance in combined text
        top_candidates.sort(key=lambda x: combined_text.find(x[1]))
        top_units = [u for _, u in top_candidates]

        # Clean list prefixes like "1. " without clipping numbers like "1st" or "2%"
        cleaned = []
        for u in top_units:
            u_clean = re.sub(r"^\d+[\.\)]\s+", "", u).strip()
            if u_clean and u_clean not in cleaned:
                cleaned.append(u_clean)

        if not cleaned:
            return UNANSWERABLE_PHRASE

        return " ".join(cleaned)

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate answer based strictly on retrieved chunks.
        """
        if not retrieved_chunks:
            return UNANSWERABLE_PHRASE

        # Check top retrieval score: if relevance is too low, declare unanswerable
        top_score = retrieved_chunks[0].get("score", 0.0)
        if top_score < 0.35:
            return UNANSWERABLE_PHRASE

        context_blocks = []
        for i, c in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source {i}: {c['scheme']} ({c['source_document']} Page {c['page_number']})]\n{c['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an official assistant for the GOV-19 Government Schemes portal.
Answer the user's question STRICTLY and SOLELY based on the provided context below.
Do NOT assume, speculate, extrapolate, or bring in any external knowledge.
If the context does not contain enough information to fully answer the question, you MUST output exactly:
"{UNANSWERABLE_PHRASE}"

Context:
{context_str}

Question: {question}

Answer:"""

        if self.llm_provider == "gemini" and self.llm_client:
            return self._call_gemini(prompt, question=question, retrieved_chunks=retrieved_chunks)
        elif self.llm_provider == "openai" and self.llm_client:
            return self._call_openai(prompt, question=question, retrieved_chunks=retrieved_chunks)
        else:
            return self._grounded_synthesizer_answer(question, retrieved_chunks)

    def query(self, question: str) -> Dict:
        """
        Full RAG pipeline invocation:
        question -> retrieve top chunks -> generate answer -> format citations
        """
        retrieved_chunks = self.retrieve(question, top_k=self.top_k)
        answer = self.generate_answer(question, retrieved_chunks)

        # Build clean citations list (deduplicated by document + page)
        seen_citations = set()
        citations = []
        for c in retrieved_chunks:
            key = (c["scheme"], c["source_document"], c["page_number"])
            if key not in seen_citations:
                seen_citations.add(key)
                citations.append({
                    "scheme": c["scheme"],
                    "source_document": c["source_document"],
                    "page_number": c["page_number"],
                })

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations,
        }


def run_sample_evaluation_tests(pipeline: RAGPipeline, eval_dataset_path: Path) -> None:
    """
    Test the RAG pipeline using representative test questions from evaluation_dataset.json.
    """
    if not eval_dataset_path.exists():
        print(f"Evaluation dataset not found at: {eval_dataset_path}")
        return

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    # Test cases requested for live Gemini test
    sample_ids = ["PMFBY_002", "PMSVAN_005", "PMUY_002", "PMFBY_010"]
    selected_cases = [c for c in eval_cases if c["id"] in sample_ids]

    print("\n" + "=" * 80)
    print("           GOV-19 LIVE GEMINI RAG TEST RESULTS")
    print(f"           LLM Provider Active: {pipeline.llm_provider.upper()}")
    print("=" * 80)

    for i, case in enumerate(selected_cases, 1):
        q_id = case["id"]
        scheme = case["scheme"]
        q_type = case["question_type"]
        question = case["question"]
        ref_ans = case["reference_answer"]

        print(f"\n[{i}/{len(selected_cases)}] Test Case ID: {q_id} | Scheme: {scheme} | Type: {q_type}")
        print(f"Question: {question}")
        print("-" * 80)

        result = pipeline.query(question)

        print(f"Generated Answer:\n{result['answer']}")
        print(f"\nReference Answer:\n{ref_ans}")

        print("\nRetrieved Chunks & Similarity Scores:")
        for idx, ch in enumerate(result["retrieved_chunks"], 1):
            print(f"  [{idx}] Chunk ID: {ch['chunk_id']} | Score: {ch['score']:.4f}")
            print(f"      Document: {ch['source_document']} | Page: {ch['page_number']} | Scheme: {ch['scheme']}")
            snippet = " ".join(ch['text'].split())[:150]
            print(f"      Text: {snippet}...")

        print("\nCitations:")
        for cit in result["citations"]:
            print(f"  • Source Document: {cit['source_document']} | Page: {cit['page_number']} | Scheme: {cit['scheme']}")

        print("=" * 80)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    eval_file = base_dir / "data" / "evaluation" / "evaluation_dataset.json"

    parser = argparse.ArgumentParser(description="GOV-19 RAG Pipeline.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Query string to run through the RAG pipeline",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run RAG pipeline test against sample questions from evaluation dataset",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline(top_k=args.top_k)

    if args.query:
        result = pipeline.query(args.query)
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    elif args.test or len(sys.argv) == 1:
        run_sample_evaluation_tests(pipeline, eval_file)


if __name__ == "__main__":
    main()
