"""
GOV-19 RAG Evaluation Engine.

Evaluates all 40 test cases from data/evaluation/evaluation_dataset.json through the
live RAG pipeline and calculates 4 standard RAG evaluation metrics (0.0 to 1.0):
1. Retrieval Relevance
2. Answer Correctness
3. Groundedness
4. Citation Correctness

Saves detailed question results and aggregated metrics to data/evaluation/evaluation_results.json.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv(override=True)

# Ensure UTF-8 output encoding in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.rag_pipeline import RAGPipeline, UNANSWERABLE_PHRASE


# Common refusal indicator patterns
REFUSAL_PATTERNS = [
    r"do(?:es)? not contain sufficient information",
    r"do(?:es)? not provide enough information",
    r"cannot be answered",
    r"not mentioned in the (?:provided|supplied) document",
    r"no sufficient information",
    r"not available in the (?:provided|supplied) context",
    r"not provide information",
    r"insufficient information",
]


def is_refusal_answer(answer: str) -> bool:
    """Check if the answer represents an unanswerable refusal."""
    if not answer or not answer.strip():
        return True
    ans_lower = answer.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, ans_lower):
            return True
    return False


def extract_key_tokens(text: str) -> List[str]:
    """Extract numbers, currency, percentages, and important numerical entities."""
    return re.findall(r"(?:₹|\b)\d+(?:[,\.]\d+)?%?", text)


class RAGEvaluator:
    """
    RAG Evaluation Engine for GOV-19 schemes.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline(top_k=4)
        self.embedding_model = self.pipeline.embedding_model

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings using the embedding model."""
        if not text1.strip() or not text2.strip():
            return 0.0
        embs = self.embedding_model.encode([text1, text2], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))  # Scale from [-1, 1] to [0, 1]

    def evaluate_retrieval_relevance(
        self,
        retrieved_chunks: List[Dict],
        source_document: str,
        source_page: int,
        supporting_evidence: str,
        answerability: str,
    ) -> float:
        """
        Calculate Retrieval Relevance score (0.0 to 1.0).
        """
        if not retrieved_chunks:
            return 0.0

        # Check if the target scheme document was retrieved
        doc_matches = [
            c for c in retrieved_chunks if c.get("source_document") == source_document
        ]
        doc_score = 1.0 if doc_matches else 0.0

        if answerability == "unanswerable":
            # For unanswerable questions, target information does not exist in the document.
            # Retrieval relevance measures whether chunks come from the correct scheme document
            # to verify absence of the requested fact.
            return doc_score

        # Page match check (exact page or adjacent ±1 page due to multi-page section continuation)
        page_matches = [
            c for c in doc_matches if abs(c.get("page_number", -99) - source_page) <= 1
        ]
        exact_page_matches = [
            c for c in doc_matches if c.get("page_number") == source_page
        ]

        if exact_page_matches:
            page_score = 1.0
        elif page_matches:
            page_score = 0.8
        elif doc_matches:
            page_score = 0.4
        else:
            page_score = 0.0

        # Evidence text coverage score
        evidence_score = 0.0
        if supporting_evidence and doc_matches:
            combined_retrieved = " ".join(c.get("text", "") for c in doc_matches)
            ev_sim = self.compute_semantic_similarity(supporting_evidence, combined_retrieved)
            # Check key token overlap from supporting evidence
            ev_tokens = [w.lower() for w in re.findall(r"\b\w{4,}\b", supporting_evidence)]
            if ev_tokens:
                matched_tokens = sum(1 for t in ev_tokens if t in combined_retrieved.lower())
                token_overlap = matched_tokens / len(ev_tokens)
            else:
                token_overlap = 1.0
            evidence_score = 0.6 * ev_sim + 0.4 * token_overlap

        # Weighted composite score
        retrieval_relevance = 0.3 * doc_score + 0.35 * page_score + 0.35 * evidence_score
        return round(float(min(1.0, max(0.0, retrieval_relevance))), 4)

    def evaluate_answer_correctness(
        self,
        generated_answer: str,
        reference_answer: str,
        answerability: str,
    ) -> float:
        """
        Calculate Answer Correctness score (0.0 to 1.0).
        """
        is_refusal = is_refusal_answer(generated_answer)

        # Unanswerable question evaluation
        if answerability == "unanswerable":
            if is_refusal:
                # Correct refusal: no penalty, perfect score
                return 1.0
            else:
                # Hallucination: model attempted to answer an unanswerable question
                return 0.0

        # Answerable question evaluation
        if is_refusal:
            # Model refused to answer an answerable question
            return 0.0

        # 1. Semantic embedding similarity
        semantic_sim = self.compute_semantic_similarity(generated_answer, reference_answer)

        # 2. Key entity / numbers matching
        ref_keys = extract_key_tokens(reference_answer)
        gen_keys = extract_key_tokens(generated_answer)
        if ref_keys:
            key_match_ratio = sum(1 for k in ref_keys if k in gen_keys or any(k in g for g in gen_keys)) / len(ref_keys)
        else:
            key_match_ratio = 1.0

        # 3. Word token F1 score
        gen_words = set(re.findall(r"\b\w+\b", generated_answer.lower()))
        ref_words = set(re.findall(r"\b\w+\b", reference_answer.lower()))
        stop_words = {"the", "a", "an", "is", "are", "and", "or", "in", "on", "of", "to", "for", "with", "by", "at", "it", "this", "that"}
        gen_content = gen_words - stop_words
        ref_content = ref_words - stop_words

        if gen_content and ref_content:
            intersection = gen_content.intersection(ref_content)
            precision = len(intersection) / len(gen_content)
            recall = len(intersection) / len(ref_content)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.5

        # Weighted composite score for answerable correctness
        correctness = 0.45 * semantic_sim + 0.35 * key_match_ratio + 0.20 * f1
        return round(float(min(1.0, max(0.0, correctness))), 4)

    def evaluate_groundedness(
        self,
        generated_answer: str,
        retrieved_chunks: List[Dict],
        answerability: str,
    ) -> float:
        """
        Calculate Groundedness score (0.0 to 1.0).
        Verifies that claims in generated answer are supported strictly by retrieved chunks.
        """
        is_refusal = is_refusal_answer(generated_answer)

        if answerability == "unanswerable":
            if is_refusal:
                # Correct refusal is grounded in the lack of information
                return 1.0
            else:
                # Hallucinated answer is completely ungrounded
                return 0.0

        if is_refusal:
            # Stating insufficient information makes no false claims
            return 1.0

        if not retrieved_chunks:
            return 0.0

        combined_context = " ".join(c.get("text", "") for c in retrieved_chunks)
        combined_context_lower = combined_context.lower()

        # Split answer into sentences
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", generated_answer)
            if len(s.strip()) > 5
        ]

        if not sentences:
            sentences = [generated_answer]

        sentence_scores = []
        for s in sentences:
            s_words = [
                w.lower()
                for w in re.findall(r"\b\w{3,}\b", s)
                if w.lower() not in {"based", "provided", "context", "document", "government", "scheme"}
            ]
            if not s_words:
                sentence_scores.append(1.0)
                continue

            # Fraction of key content words present in retrieved chunks
            found = sum(1 for w in s_words if w in combined_context_lower)
            overlap_ratio = found / len(s_words)

            # Check numbers/entities
            s_nums = extract_key_tokens(s)
            ctx_nums = extract_key_tokens(combined_context)
            if s_nums:
                num_overlap = sum(1 for n in s_nums if n in ctx_nums) / len(s_nums)
            else:
                num_overlap = 1.0

            sentence_score = 0.6 * overlap_ratio + 0.4 * num_overlap
            sentence_scores.append(sentence_score)

        groundedness = float(np.mean(sentence_scores)) if sentence_scores else 0.0
        return round(float(min(1.0, max(0.0, groundedness))), 4)

    def evaluate_citation_correctness(
        self,
        citations: List[Dict],
        source_document: str,
        source_page: int,
        scheme: str,
        answerability: str,
    ) -> float:
        """
        Calculate Citation Correctness score (0.0 to 1.0).
        """
        if not citations:
            return 0.0

        doc_matches = [
            c for c in citations if c.get("source_document") == source_document
        ]

        if not doc_matches:
            # None of the citations match the source document
            return 0.0

        if answerability == "unanswerable":
            # For unanswerable questions, citing the source document where absence was checked is correct
            return 1.0

        # Check page match in citations
        has_exact_page = any(c.get("page_number") == source_page for c in doc_matches)
        has_adjacent_page = any(
            abs(c.get("page_number", -99) - source_page) <= 1 for c in doc_matches
        )

        if has_exact_page:
            return 1.0
        elif has_adjacent_page:
            return 0.85
        else:
            return 0.50

    def evaluate_question(self, case: Dict) -> Dict:
        """
        Run a single test case through the RAG pipeline and compute all metrics.
        """
        q_id = case["id"]
        scheme = case["scheme"]
        question = case["question"]
        ref_answer = case["reference_answer"]
        src_doc = case["source_document"]
        src_page = case["source_page"]
        supporting_ev = case.get("supporting_evidence", "")
        answerability = case.get("answerability", "answerable")

        # Run query through existing RAG pipeline
        rag_output = self.pipeline.query(question)
        gen_answer = rag_output["answer"]
        retrieved_chunks = rag_output["retrieved_chunks"]
        citations = rag_output["citations"]

        # Calculate the 4 core metrics
        retrieval_relevance = self.evaluate_retrieval_relevance(
            retrieved_chunks=retrieved_chunks,
            source_document=src_doc,
            source_page=src_page,
            supporting_evidence=supporting_ev,
            answerability=answerability,
        )

        answer_correctness = self.evaluate_answer_correctness(
            generated_answer=gen_answer,
            reference_answer=ref_answer,
            answerability=answerability,
        )

        groundedness = self.evaluate_groundedness(
            generated_answer=gen_answer,
            retrieved_chunks=retrieved_chunks,
            answerability=answerability,
        )

        citation_correctness = self.evaluate_citation_correctness(
            citations=citations,
            source_document=src_doc,
            source_page=src_page,
            scheme=scheme,
            answerability=answerability,
        )

        # Question reliability score
        reliability_score = round(
            float(
                (
                    retrieval_relevance
                    + answer_correctness
                    + groundedness
                    + citation_correctness
                )
                / 4.0
            ),
            4,
        )

        # Determine failure reason if question failed
        failure_reason = None
        is_refusal = is_refusal_answer(gen_answer)

        if answerability == "unanswerable" and not is_refusal:
            failure_reason = "Hallucination failure: Model generated facts for an unanswerable question."
        elif answerability == "answerable" and is_refusal:
            failure_reason = "Generation failure: Model refused an answerable question due to retrieval/context gap."
        elif answer_correctness < 0.50:
            failure_reason = f"Factual mismatch: Answer correctness score ({answer_correctness:.2f}) below threshold."
        elif retrieval_relevance < 0.40:
            failure_reason = f"Retrieval failure: Target source document or page not adequately retrieved."
        elif reliability_score < 0.60:
            failure_reason = f"Low reliability: Composite score ({reliability_score:.2f}) below 0.60 threshold."

        # Simplify retrieved chunks for JSON storage
        saved_chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
                "source_document": c.get("source_document"),
                "page_number": c.get("page_number"),
                "snippet": " ".join(c.get("text", "").split())[:200],
            }
            for c in retrieved_chunks
        ]

        return {
            "id": q_id,
            "scheme": scheme,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": gen_answer,
            "retrieved_chunks": saved_chunks,
            "citations": citations,
            "retrieval_relevance": retrieval_relevance,
            "answer_correctness": answer_correctness,
            "groundedness": groundedness,
            "citation_correctness": citation_correctness,
            "reliability_score": reliability_score,
            "failure_reason": failure_reason,
        }


def run_full_evaluation(
    eval_dataset_path: Path,
    output_results_path: Path,
    delay_between_queries: float = 1.0,
) -> Dict:
    """
    Execute evaluation on all questions in dataset and save structured JSON results.
    """
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {eval_dataset_path}")

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("\n" + "=" * 80)
    print("           GOV-19 RAG AUTOMATED EVALUATION ENGINE")
    print(f"           Total Test Cases: {len(cases)}")
    print(f"           Results Destination: {output_results_path}")
    print("=" * 80 + "\n")

    evaluator = RAGEvaluator()
    results = []

    for i, case in enumerate(cases, 1):
        q_id = case["id"]
        scheme = case["scheme"]
        q_text = case["question"]

        print(f"[{i:02d}/{len(cases):02d}] Evaluating {q_id} ({scheme})...", end="", flush=True)
        start_time = time.time()

        res = evaluator.evaluate_question(case)
        elapsed = time.time() - start_time
        results.append(res)

        status_str = "FAIL" if res["failure_reason"] else "PASS"
        print(f" -> {status_str} | Reliability: {res['reliability_score']:.2f} ({elapsed:.1f}s)")

        if delay_between_queries > 0 and i < len(cases):
            time.sleep(delay_between_queries)

    # Compute overall statistics
    total_q = len(results)
    successful_q = sum(1 for r in results if r["failure_reason"] is None)
    failed_q = total_q - successful_q
    success_rate = round(successful_q / total_q, 4) if total_q > 0 else 0.0

    avg_retrieval_rel = round(float(np.mean([r["retrieval_relevance"] for r in results])), 4)
    avg_answer_corr = round(float(np.mean([r["answer_correctness"] for r in results])), 4)
    avg_groundedness = round(float(np.mean([r["groundedness"] for r in results])), 4)
    avg_citation_corr = round(float(np.mean([r["citation_correctness"] for r in results])), 4)
    overall_reliability = round(
        float(np.mean([r["reliability_score"] for r in results])), 4
    )

    # Compute per-scheme statistics
    schemes = sorted(list(set(r["scheme"] for r in results)))
    per_scheme_scores = {}

    for s in schemes:
        s_res = [r for r in results if r["scheme"] == s]
        s_total = len(s_res)
        s_succ = sum(1 for r in s_res if r["failure_reason"] is None)
        s_fail = s_total - s_succ
        per_scheme_scores[s] = {
            "total_questions": s_total,
            "successful_questions": s_succ,
            "failed_questions": s_fail,
            "success_rate": round(s_succ / s_total, 4) if s_total > 0 else 0.0,
            "overall_reliability": round(float(np.mean([r["reliability_score"] for r in s_res])), 4),
            "retrieval_relevance": round(float(np.mean([r["retrieval_relevance"] for r in s_res])), 4),
            "answer_correctness": round(float(np.mean([r["answer_correctness"] for r in s_res])), 4),
            "groundedness": round(float(np.mean([r["groundedness"] for r in s_res])), 4),
            "citation_correctness": round(float(np.mean([r["citation_correctness"] for r in s_res])), 4),
        }

    evaluation_data = {
        "summary": {
            "total_questions": total_q,
            "successful_questions": successful_q,
            "failed_questions": failed_q,
            "success_rate": success_rate,
            "overall_reliability_score": overall_reliability,
            "metric_averages": {
                "retrieval_relevance": avg_retrieval_rel,
                "answer_correctness": avg_answer_corr,
                "groundedness": avg_groundedness,
                "citation_correctness": avg_citation_corr,
            },
            "per_scheme_scores": per_scheme_scores,
        },
        "results": results,
    }

    # Ensure parent output directory exists
    output_results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("                    EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated:  {total_q}")
    print(f"Successful Questions:       {successful_q}")
    print(f"Failed Questions:           {failed_q}")
    print(f"Success Rate:               {success_rate * 100:.1f}%")
    print(f"Overall Reliability Score:  {overall_reliability:.4f}")
    print("-" * 80)
    print("Core Metric Averages:")
    print(f"  1. Retrieval Relevance:   {avg_retrieval_rel:.4f}")
    print(f"  2. Answer Correctness:    {avg_answer_corr:.4f}")
    print(f"  3. Groundedness:          {avg_groundedness:.4f}")
    print(f"  4. Citation Correctness:  {avg_citation_corr:.4f}")
    print("-" * 80)
    print("Per-Scheme Scores:")
    for s_name, s_data in per_scheme_scores.items():
        print(f"  • {s_name}:")
        print(f"      Reliability: {s_data['overall_reliability']:.4f} | Pass: {s_data['successful_questions']}/{s_data['total_questions']} | Rel: {s_data['retrieval_relevance']:.2f} | Ans: {s_data['answer_correctness']:.2f} | Grd: {s_data['groundedness']:.2f} | Cit: {s_data['citation_correctness']:.2f}")
    print("=" * 80)
    print(f"Results successfully saved to: {output_results_path}\n")

    return evaluation_data


def main():
    parser = argparse.ArgumentParser(description="GOV-19 Automated RAG Evaluation Engine")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BASE_DIR / "data" / "evaluation" / "evaluation_dataset.json"),
        help="Path to evaluation_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "data" / "evaluation" / "evaluation_results.json"),
        help="Path to save evaluation_results.json",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between LLM calls to respect rate limits (default: 1.0)",
    )
    args = parser.parse_args()

    run_full_evaluation(
        eval_dataset_path=Path(args.dataset),
        output_results_path=Path(args.output),
        delay_between_queries=args.delay,
    )


if __name__ == "__main__":
    main()
