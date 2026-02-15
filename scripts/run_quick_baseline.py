#!/usr/bin/env python3
"""
Quick baseline evaluation without RAGAS (faster, uses custom metrics only).
Good for initial testing and iteration.

Supports both OpenAI and Nebius Token Factory (Qwen models).

Usage:
    python scripts/run_quick_baseline.py
    python scripts/run_quick_baseline.py --model gpt-4o --provider openai
    python scripts/run_quick_baseline.py --model Qwen/Qwen2.5-72B-Instruct --provider nebius
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


def get_client(provider: str) -> OpenAI:
    """Get the appropriate OpenAI-compatible client."""
    if provider == "nebius":
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.studio.nebius.ai/v1")
        if not api_key:
            raise ValueError("LLM_API_KEY not set for Nebius")
        return OpenAI(api_key=api_key, base_url=base_url)
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)


def load_dataset(path: str) -> list[dict]:
    """Load and flatten the evaluation dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = []
    for domain, items in data.items():
        for item in items:
            questions.append({
                "domain": domain,
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
    import re
    expected_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
    generated_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))
    
    if expected_numbers and expected_numbers.issubset(generated_numbers):
        return True
    
    # Word overlap
    expected_words = set(expected_lower.split())
    generated_words = set(generated_lower.split())
    overlap = len(expected_words & generated_words) / max(len(expected_words), 1)
    
    return overlap > 0.5


def run_evaluation(model: str, questions: list[dict], strategy: str = "basic", provider: str = "nebius") -> list[dict]:
    """Run evaluation on all questions."""
    client = get_client(provider)
    results = []
    
    # Define prompt strategies
    strategies = {
        "basic": {
            "system": "אתה נציג שירות לקוחות של חברת הביטוח הראל. ענה על שאלות הלקוחות בצורה מקצועית ומדויקת.",
            "template": "{question}",
        },
        "strict": {
            "system": """אתה נציג שירות לקוחות של חברת הביטוח הראל.
כללים חשובים:
1. ענה רק על סמך מידע שאתה בטוח בו
2. אם אינך בטוח, אמור "אני לא בטוח"
3. אל תמציא מידע
4. תשובות קצרות וממוקדות""",
            "template": "תחום: {domain}\nשאלה: {question}\nתשובה:",
        },
    }
    
    strat = strategies.get(strategy, strategies["basic"])
    
    for q in tqdm(questions, desc=f"Evaluating {model}"):
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": strat["system"]},
                    {"role": "user", "content": strat["template"].format(
                        question=q["question"],
                        domain=q["domain"]
                    )},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            generated = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            generated = f"Error: {e}"
            latency_ms = 0
        
        is_correct = check_correctness(q["expected_answer"], generated)
        
        results.append({
            "domain": q["domain"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "generated_answer": generated,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
            "model": model,
            "strategy": strategy,
            "source_file": q["source_file"],
            "source_page": q["source_page"],
        })
    
    return results


def print_summary(results: list[dict], model: str):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    avg_latency = sum(r["latency_ms"] for r in results) / total
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model}")
    print(f"{'='*60}")
    print(f"Total Questions: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {total - correct} ({(total-correct)/total*100:.1f}%)")
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
        print(f"\n{'='*60}")
        print("FAILURES (where RAG is needed):")
        print(f"{'='*60}")
        for i, f in enumerate(failures[:5], 1):
            print(f"\n{i}. [{f['domain']}] {f['question'][:60]}...")
            print(f"   Expected: {f['expected_answer'][:50]}")
            print(f"   Got: {f['generated_answer'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Quick baseline evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct", help="Model to test")
    parser.add_argument("--provider", default="nebius", choices=["nebius", "openai"], help="API provider")
    parser.add_argument("--strategy", default="basic", choices=["basic", "strict"])
    parser.add_argument("--dataset", default="data/evaluation/dev_set.json")
    parser.add_argument("--output", default="data/evaluation/results/quick_results.json")

    args = parser.parse_args()

    # Validate API key based on provider
    if args.provider == "nebius":
        if not os.getenv("LLM_API_KEY"):
            print("Error: LLM_API_KEY not set for Nebius")
            sys.exit(1)
    else:
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)

    print(f"Loading dataset from {args.dataset}...")
    questions = load_dataset(args.dataset)
    print(f"Loaded {len(questions)} questions")

    print(f"\nRunning evaluation with {args.model} via {args.provider} ({args.strategy} strategy)...")
    results = run_evaluation(args.model, questions, args.strategy, args.provider)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print_summary(results, args.model)


if __name__ == "__main__":
    main()

