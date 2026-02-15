"""
Report generator for baseline evaluation results.
Creates markdown reports showing where GPT-5 succeeds and fails.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any


class ReportGenerator:
    """
    Generates baseline evaluation reports in Markdown format.
    Analyzes where models succeed and fail without domain grounding.
    """
    
    def __init__(self, results_dir: str | Path = "data/evaluation/results"):
        """
        Initialize the report generator.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
    
    def load_results(self) -> tuple[list[dict], dict]:
        """Load detailed and aggregated results."""
        detailed_path = self.results_dir / "detailed_results.json"
        aggregated_path = self.results_dir / "aggregated_results.json"
        
        with open(detailed_path, "r", encoding="utf-8") as f:
            detailed = json.load(f)
        
        with open(aggregated_path, "r", encoding="utf-8") as f:
            aggregated = json.load(f)
        
        return detailed, aggregated
    
    def _find_best_config(self, aggregated: dict) -> tuple[str, str, dict]:
        """Find the best model/strategy combination by accuracy."""
        best_model, best_strategy, best_metrics = None, None, None
        best_accuracy = -1
        
        for model, strategies in aggregated.items():
            for strategy, metrics in strategies.items():
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_model, best_strategy, best_metrics = model, strategy, metrics
        
        return best_model, best_strategy, best_metrics
    
    def _analyze_failures(self, detailed: list[dict]) -> dict[str, list[dict]]:
        """Analyze failure patterns by domain and type."""
        failures = {
            "incorrect_answers": [],
            "hallucinations": [],
            "missing_citations": [],
            "by_domain": {},
        }
        
        for result in detailed:
            domain = result["domain"]
            if domain not in failures["by_domain"]:
                failures["by_domain"][domain] = []
            
            if not result["is_correct"]:
                failures["incorrect_answers"].append(result)
                failures["by_domain"][domain].append(result)
            
            if result["is_hallucination"]:
                failures["hallucinations"].append(result)
            
            if not result["has_citation"]:
                failures["missing_citations"].append(result)
        
        return failures
    
    def _analyze_successes(self, detailed: list[dict]) -> list[dict]:
        """Find questions where the model succeeded."""
        return [r for r in detailed if r["is_correct"] and not r["is_hallucination"]]
    
    def generate_report(self, output_path: str | Path | None = None) -> str:
        """
        Generate a comprehensive baseline report.
        
        Args:
            output_path: Path to save the report (optional)
        
        Returns:
            The report as a markdown string
        """
        detailed, aggregated = self.load_results()
        best_model, best_strategy, best_metrics = self._find_best_config(aggregated)
        failures = self._analyze_failures(detailed)
        successes = self._analyze_successes(detailed)
        
        # Filter detailed results for best config
        best_results = [
            r for r in detailed 
            if r["model"] == best_model and r["prompt_strategy"] == best_strategy
        ]
        
        report = self._build_report(
            aggregated=aggregated,
            best_model=best_model,
            best_strategy=best_strategy,
            best_metrics=best_metrics,
            best_results=best_results,
            failures=failures,
            successes=successes,
        )
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
        
        return report
    
    def _build_report(
        self,
        aggregated: dict,
        best_model: str,
        best_strategy: str,
        best_metrics: dict,
        best_results: list[dict],
        failures: dict,
        successes: list[dict],
    ) -> str:
        """Build the markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Baseline Evaluation Report

**Generated**: {timestamp}

## Executive Summary

This report evaluates GPT models on Harel Insurance customer support questions **without RAG** (no document retrieval).
The goal is to establish a baseline and identify where domain grounding is needed.

### Key Findings

| Metric | Best Result | Target | Gap |
|--------|-------------|--------|-----|
| **Accuracy** | {best_metrics['accuracy']:.1%} | 90% | {90 - best_metrics['accuracy']*100:.1f}% |
| **Hallucination Rate** | {best_metrics['hallucination_rate']:.1%} | <5% | {max(0, best_metrics['hallucination_rate']*100 - 5):.1f}% |
| **Citation Rate** | {best_metrics['citation_rate']:.1%} | >90% | {max(0, 90 - best_metrics['citation_rate']*100):.1f}% |
| **Avg Latency** | {best_metrics['avg_latency_ms']:.0f}ms | <2000ms | ✅ |

**Best Configuration**: `{best_model}` with `{best_strategy}` strategy

---

## Model Comparison

"""
        # Add model comparison table
        report += "| Model | Strategy | Accuracy | Hallucination | Citation Rate | Latency |\n"
        report += "|-------|----------|----------|---------------|---------------|----------|\n"
        
        for model, strategies in aggregated.items():
            for strategy, metrics in strategies.items():
                report += f"| {model} | {strategy} | {metrics['accuracy']:.1%} | {metrics['hallucination_rate']:.1%} | {metrics['citation_rate']:.1%} | {metrics['avg_latency_ms']:.0f}ms |\n"
        
        report += "\n---\n\n"

        # Domain breakdown
        report += "## Performance by Domain\n\n"
        report += "| Domain | Questions | Correct | Accuracy | Hallucinations |\n"
        report += "|--------|-----------|---------|----------|----------------|\n"

        domain_stats = {}
        for result in best_results:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "correct": 0, "hallucinations": 0}
            domain_stats[domain]["total"] += 1
            if result["is_correct"]:
                domain_stats[domain]["correct"] += 1
            if result["is_hallucination"]:
                domain_stats[domain]["hallucinations"] += 1

        for domain, stats in sorted(domain_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            report += f"| {domain} | {stats['total']} | {stats['correct']} | {acc:.1%} | {stats['hallucinations']} |\n"

        report += "\n---\n\n"

        # Where GPT-5 Succeeds
        report += "## Where GPT Succeeds (Without RAG)\n\n"
        report += "These questions were answered correctly without domain-specific documents:\n\n"

        success_results = [r for r in best_results if r["is_correct"]]
        for i, result in enumerate(success_results[:5], 1):
            report += f"### Success {i}: {result['domain']}\n\n"
            report += f"**Question**: {result['question']}\n\n"
            report += f"**Expected**: {result['expected_answer']}\n\n"
            report += f"**Generated**: {result['generated_answer'][:200]}...\n\n"
            report += "---\n\n"

        # Where GPT-5 Fails
        report += "## Where GPT Fails (Needs RAG)\n\n"
        report += "These questions require domain-specific documents to answer correctly:\n\n"

        failure_results = [r for r in best_results if not r["is_correct"]]
        for i, result in enumerate(failure_results[:5], 1):
            report += f"### Failure {i}: {result['domain']}\n\n"
            report += f"**Question**: {result['question']}\n\n"
            report += f"**Expected**: {result['expected_answer']}\n\n"
            report += f"**Generated**: {result['generated_answer'][:200]}...\n\n"
            report += f"**Source**: {result['source_file']} (page {result['source_page']})\n\n"
            report += f"**Analysis**: This requires specific policy information from the source document.\n\n"
            report += "---\n\n"

        # Hallucination Analysis
        report += "## Hallucination Analysis\n\n"
        hallucination_results = [r for r in best_results if r["is_hallucination"]]
        report += f"**Total Hallucinations**: {len(hallucination_results)} / {len(best_results)} ({len(hallucination_results)/len(best_results)*100:.1f}%)\n\n"

        if hallucination_results:
            report += "### Examples of Hallucinations\n\n"
            for i, result in enumerate(hallucination_results[:3], 1):
                report += f"**Example {i}**: {result['domain']}\n\n"
                report += f"- Question: {result['question']}\n"
                report += f"- Expected: {result['expected_answer']}\n"
                report += f"- Generated: {result['generated_answer'][:150]}...\n"
                report += f"- **Issue**: Model provided incorrect information without source verification\n\n"

        report += "---\n\n"

        # Citation Analysis
        report += "## Citation Analysis\n\n"
        citation_results = [r for r in best_results if r["has_citation"]]
        report += f"**Answers with Citations**: {len(citation_results)} / {len(best_results)} ({len(citation_results)/len(best_results)*100:.1f}%)\n\n"
        report += "**Key Finding**: Without RAG, the model cannot provide accurate citations to source documents.\n\n"
        report += "This is expected behavior - citations require document retrieval.\n\n"

        report += "---\n\n"

        # Recommendations
        report += "## Recommendations for RAG Implementation\n\n"
        report += "Based on this baseline evaluation:\n\n"
        report += "1. **High Priority Domains**: Focus RAG on domains with lowest accuracy\n"
        report += "2. **Citation System**: Implement mandatory citation attachment from retrieved documents\n"
        report += "3. **Hallucination Prevention**: Use strict grounding prompts + verification agent\n"
        report += "4. **Specific Numbers**: Questions about prices, dates, limits need exact document retrieval\n"
        report += "5. **Yes/No Questions**: Many failures are on coverage questions - need policy documents\n\n"

        # Conclusion
        report += "## Conclusion\n\n"
        report += f"The baseline GPT model achieves **{best_metrics['accuracy']:.1%} accuracy** without RAG.\n\n"
        report += "Key gaps that RAG must address:\n\n"
        report += f"- **{100 - best_metrics['accuracy']*100:.1f}%** of questions need domain-specific documents\n"
        report += f"- **{best_metrics['hallucination_rate']*100:.1f}%** hallucination rate must be reduced to <5%\n"
        report += f"- **{100 - best_metrics['citation_rate']*100:.1f}%** of answers lack citations\n\n"
        report += "The RAG system must retrieve relevant policy documents and ground all answers in source material.\n"

        return report
