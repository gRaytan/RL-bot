#!/usr/bin/env python3
"""
Run baseline evaluation on GPT models without RAG.
Establishes baseline performance and identifies where domain grounding is needed.

Usage:
    python scripts/run_baseline_evaluation.py
    python scripts/run_baseline_evaluation.py --models gpt-4o --strategies basic,strict_grounding
    python scripts/run_baseline_evaluation.py --no-ragas  # Skip RAGAS metrics (faster)
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on GPT models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/dev_set.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-4o",
        help="Comma-separated list of models to test (e.g., gpt-4o,gpt-4-turbo)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="basic,domain_aware,strict_grounding,citation_required",
        help="Comma-separated list of prompt strategies to test",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS metrics (faster but less accurate)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results",
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not args.report_only:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in .env file or export it")
        sys.exit(1)
    
    # Parse models and strategies
    models = [m.strip() for m in args.models.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    
    from src.evaluation.baseline_runner import BaselineRunner
    from src.evaluation.report_generator import ReportGenerator
    
    if not args.report_only:
        print("=" * 60)
        print("BASELINE EVALUATION - Harel Insurance Chatbot")
        print("=" * 60)
        print(f"\nDataset: {args.dataset}")
        print(f"Models: {models}")
        print(f"Strategies: {strategies}")
        print(f"Use RAGAS: {not args.no_ragas}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)
        
        # Run evaluation
        runner = BaselineRunner(
            models=models,
            strategies=strategies,
        )
        
        results = runner.run_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            use_ragas=not args.no_ragas,
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
    
    # Generate report
    print("\nGenerating baseline report...")
    report_generator = ReportGenerator(results_dir=args.output_dir)
    report_path = Path(args.output_dir) / "BASELINE_REPORT.md"
    
    try:
        report = report_generator.generate_report(output_path=report_path)
        print(f"\nReport saved to: {report_path}")
        print("\n" + "=" * 60)
        print("REPORT PREVIEW")
        print("=" * 60)
        # Print first 50 lines of report
        lines = report.split("\n")[:50]
        print("\n".join(lines))
        if len(report.split("\n")) > 50:
            print("\n... (see full report in file)")
    except FileNotFoundError:
        print("Error: No results found. Run evaluation first (without --report-only)")
        sys.exit(1)


if __name__ == "__main__":
    main()

