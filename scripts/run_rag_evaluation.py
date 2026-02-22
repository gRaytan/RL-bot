#!/usr/bin/env python3
"""
RAG Evaluation - Run evaluation dataset through the RAG pipeline.

Compares RAG performance against baseline (55% accuracy).

Usage:
    python scripts/run_rag_evaluation.py
    python scripts/run_rag_evaluation.py --no-rerank
    python scripts/run_rag_evaluation.py --model Qwen/Qwen2.5-72B-Instruct
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.rag import RAGPipeline, RAGConfig
from src.rag.answer_generator import GeneratorConfig

# Domain mapping from Hebrew to English
DOMAIN_MAP = {
    "רכב": "car",
    "דירה": "apartment",
    "נסיעות": "travel",
    "בריאות": "health",
    "עסק": "business",
    "חיים": "life",
}


def load_dataset(path: str) -> list[dict]:
    """Load and flatten the evaluation dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for domain, items in data.items():
        for item in items:
            questions.append({
                "domain": domain,
                "domain_en": DOMAIN_MAP.get(domain, domain),
                "question": item["שאלה"],
                "expected_answer": item["תשובה"],
                "source_file": item["מקור"]["קובץ"],
                "source_page": item["מקור"]["עמוד"],
            })
    return questions


def check_correctness(expected: str, generated: str) -> bool:
    """Check if the answer is correct using simple heuristics."""
    expected_lower = expected.lower().strip()
    generated_lower = generated.lower().strip()

    # Hebrew yes/no detection
    yes_words = ["כן", "yes", "נכון", "מכוסה", "זכאי"]
    no_words = ["לא", "no", "אינו", "אינה", "לא מכוסה", "לא זכאי"]

    expected_is_yes = any(w in expected_lower for w in yes_words) and not any(w in expected_lower for w in no_words)
    expected_is_no = any(w in expected_lower for w in no_words)
    generated_is_yes = any(w in generated_lower for w in yes_words) and not any(w in generated_lower for w in no_words)
    generated_is_no = any(w in generated_lower for w in no_words)

    if expected_is_yes and generated_is_yes:
        return True
    if expected_is_no and generated_is_no:
        return True

    # Check for numbers
    expected_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
    generated_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))

    if expected_numbers and expected_numbers.issubset(generated_numbers):
        return True

    # Word overlap
    expected_words = set(expected_lower.split())
    generated_words = set(generated_lower.split())
    overlap = len(expected_words & generated_words) / max(len(expected_words), 1)

    return overlap > 0.5


def run_rag_evaluation(pipeline: RAGPipeline, questions: list[dict], use_domain_filter: bool = True) -> list[dict]:
    """Run evaluation through RAG pipeline."""
    results = []

    for q in tqdm(questions, desc="RAG Evaluation"):
        domain_filter = q["domain_en"] if use_domain_filter else None

        try:
            response = pipeline.query(q["question"], domain_filter=domain_filter)
            generated = response.answer
            latency_ms = response.total_time_ms
            citations = [{"file": c.source_file, "page": c.page_num} for c in response.citations]
            confidence = response.confidence
        except Exception as e:
            generated = f"Error: {e}"
            latency_ms = 0
            citations = []
            confidence = "error"

        is_correct = check_correctness(q["expected_answer"], generated)

        # Check if correct source was cited (lenient matching)
        # Try exact match first, then partial filename match
        correct_source_cited = False
        if citations:
            expected_filename = q["source_file"].split("/")[-1].replace(".pdf", "")
            for c in citations:
                cited_filename = c["file"].replace(".pdf", "")
                # Check if any significant words overlap
                if expected_filename in cited_filename or cited_filename in expected_filename:
                    correct_source_cited = True
                    break
                # Check domain match at least
                if q["domain_en"] in c["file"].lower():
                    correct_source_cited = True
                    break

        results.append({
            "domain": q["domain"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "generated_answer": generated,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
            "citations": citations,
            "correct_source_cited": correct_source_cited,
            "confidence": confidence,
            "expected_source": q["source_file"],
            "expected_page": q["source_page"],
        })

    return results


def print_summary(results: list[dict], baseline_accuracy: float = 0.55):
    """Print evaluation summary with comparison to baseline."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total
    avg_latency = sum(r["latency_ms"] for r in results) / total
    correct_citations = sum(1 for r in results if r["correct_source_cited"])

    print(f"\n{'='*70}")
    print(f"RAG EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total Questions: {total}")
    print(f"Correct: {correct} ({accuracy*100:.1f}%)")
    print(f"Baseline: {baseline_accuracy*100:.0f}%")
    print(f"Improvement: {(accuracy - baseline_accuracy)*100:+.1f}%")
    print(f"Correct Source Cited: {correct_citations}/{total} ({correct_citations/total*100:.1f}%)")
    print(f"Avg Latency: {avg_latency:.0f}ms")

    # Per-domain breakdown
    print(f"\nPer-Domain Breakdown:")
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        domain_results = [r for r in results if r["domain"] == domain]
        domain_correct = sum(1 for r in domain_results if r["is_correct"])
        print(f"  {domain}: {domain_correct}/{len(domain_results)} ({domain_correct/len(domain_results)*100:.0f}%)")

    # Show failures
    failures = [r for r in results if not r["is_correct"]]
    if failures:
        print(f"\n{'='*70}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'='*70}")
        for i, f in enumerate(failures[:5], 1):
            print(f"\n{i}. [{f['domain']}] {f['question'][:60]}...")
            print(f"   Expected: {f['expected_answer'][:50]}")
            print(f"   Got: {f['generated_answer'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="LLM model")
    parser.add_argument("--dataset", default="data/evaluation/dev_set.json")
    parser.add_argument("--output", default="data/evaluation/results/rag_results.json")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking")
    parser.add_argument("--use-domain-filter", action="store_true", help="Filter by domain (default: no filter)")
    parser.add_argument("--auto-domain", action="store_true", help="Use auto domain classification from TOC")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification agent")

    args = parser.parse_args()

    print("Loading dataset...")
    questions = load_dataset(args.dataset)
    print(f"Loaded {len(questions)} questions")

    print("\nInitializing RAG pipeline...")
    config = RAGConfig(
        retrieval_top_k=50,       # Increased from 30 for better recall
        rerank_top_k=15,          # Increased from 10
        final_context_k=10,       # Optimal: 10 chunks to LLM
        use_reranker=not args.no_rerank,
        use_auto_domain=args.auto_domain,
        use_verification=not args.no_verification,
        generator_config=GeneratorConfig(
            provider="nebius",
            model=args.model,
        ),
    )
    pipeline = RAGPipeline(config=config)

    print(f"\nConfig: rerank={not args.no_rerank}, auto_domain={args.auto_domain}, verification={not args.no_verification}, domain_filter={args.use_domain_filter}")
    print("\nRunning RAG evaluation...")
    results = run_rag_evaluation(pipeline, questions, use_domain_filter=args.use_domain_filter)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
