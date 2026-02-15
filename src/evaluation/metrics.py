"""
Evaluation metrics using RAGAS framework.
Measures answer relevance, faithfulness, and hallucination detection.

Compatible with RAGAS 0.4.x API.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Any

# RAGAS 0.4.x imports
try:
    from ragas.metrics import AnswerRelevancy, AnswerCorrectness, Faithfulness
    from ragas.llms import llm_factory
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not available. Install with: pip install ragas")


@dataclass
class EvaluationResult:
    """Single evaluation result for a question-answer pair."""
    question: str
    expected_answer: str
    generated_answer: str
    domain: str
    source_file: str
    source_page: int
    
    # RAGAS metrics
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0
    faithfulness: float = 0.0
    
    # Custom metrics
    is_correct: bool = False
    has_citation: bool = False
    citation_accurate: bool = False
    is_hallucination: bool = False
    
    # Metadata
    latency_ms: float = 0.0
    model: str = ""
    prompt_strategy: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "domain": self.domain,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "answer_relevancy": self.answer_relevancy,
            "answer_correctness": self.answer_correctness,
            "faithfulness": self.faithfulness,
            "is_correct": self.is_correct,
            "has_citation": self.has_citation,
            "citation_accurate": self.citation_accurate,
            "is_hallucination": self.is_hallucination,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all evaluation results."""
    total_questions: int = 0
    correct_answers: int = 0
    accuracy: float = 0.0
    
    avg_answer_relevancy: float = 0.0
    avg_answer_correctness: float = 0.0
    avg_faithfulness: float = 0.0
    
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    
    citation_count: int = 0
    citation_rate: float = 0.0
    citation_accuracy: float = 0.0
    
    avg_latency_ms: float = 0.0
    
    # Per-domain breakdown
    domain_metrics: dict[str, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "avg_answer_relevancy": self.avg_answer_relevancy,
            "avg_answer_correctness": self.avg_answer_correctness,
            "avg_faithfulness": self.avg_faithfulness,
            "hallucination_count": self.hallucination_count,
            "hallucination_rate": self.hallucination_rate,
            "citation_count": self.citation_count,
            "citation_rate": self.citation_rate,
            "citation_accuracy": self.citation_accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "domain_metrics": self.domain_metrics,
        }


class EvaluationMetrics:
    """
    Evaluation metrics calculator using RAGAS.
    Provides answer relevancy, correctness, faithfulness, and custom metrics.
    """

    def __init__(self, llm_model: str = "gpt-4o"):
        """
        Initialize evaluation metrics.

        Args:
            llm_model: LLM model to use for RAGAS evaluation
        """
        self.llm_model = llm_model
        self._llm = None
        self._metrics_initialized = False

    def _init_ragas_metrics(self):
        """Initialize RAGAS metrics lazily."""
        if not RAGAS_AVAILABLE:
            return

        if not self._metrics_initialized:
            try:
                self._llm = llm_factory(self.llm_model)
                self._answer_relevancy = AnswerRelevancy(llm=self._llm)
                self._answer_correctness = AnswerCorrectness(llm=self._llm)
                self._faithfulness = Faithfulness(llm=self._llm)
                self._metrics_initialized = True
            except Exception as e:
                print(f"Failed to initialize RAGAS metrics: {e}")

    def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        context: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a single question-answer pair using RAGAS.

        Args:
            question: The input question
            expected_answer: Ground truth answer
            generated_answer: Model-generated answer
            context: Retrieved context (if any)

        Returns:
            Dictionary of metric scores
        """
        if not RAGAS_AVAILABLE:
            return {
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
            }

        self._init_ragas_metrics()

        if not self._metrics_initialized:
            return {
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
            }

        # Run RAGAS evaluation using new API
        try:
            # Create async event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Evaluate each metric
            relevancy_score = loop.run_until_complete(
                self._answer_relevancy.ascore(
                    user_input=question,
                    response=generated_answer,
                )
            )

            correctness_score = loop.run_until_complete(
                self._answer_correctness.ascore(
                    user_input=question,
                    response=generated_answer,
                    reference=expected_answer,
                )
            )

            # Faithfulness requires context
            if context:
                faithfulness_score = loop.run_until_complete(
                    self._faithfulness.ascore(
                        user_input=question,
                        response=generated_answer,
                        retrieved_contexts=context,
                    )
                )
            else:
                # Without context, faithfulness is not applicable
                faithfulness_score = type('obj', (object,), {'value': 0.5})()

            return {
                "answer_relevancy": getattr(relevancy_score, 'value', relevancy_score) if relevancy_score else 0.0,
                "answer_correctness": getattr(correctness_score, 'value', correctness_score) if correctness_score else 0.0,
                "faithfulness": getattr(faithfulness_score, 'value', faithfulness_score) if faithfulness_score else 0.0,
            }
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
            }

    def check_correctness(
        self,
        expected: str,
        generated: str,
    ) -> bool:
        """
        Check if the generated answer is correct.
        Uses simple heuristics for yes/no questions and keyword matching.
        """
        expected_lower = expected.lower().strip()
        generated_lower = generated.lower().strip()

        # Handle yes/no questions (Hebrew)
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

        # For non-yes/no, check if key parts of expected are in generated
        # Extract numbers and key terms
        import re
        expected_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
        generated_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))

        if expected_numbers and expected_numbers.issubset(generated_numbers):
            return True

        # Check for significant overlap
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())
        overlap = len(expected_words & generated_words) / max(len(expected_words), 1)

        return overlap > 0.5

    def check_citation(self, generated: str) -> tuple[bool, bool]:
        """
        Check if the answer contains a citation and if it's accurate.

        Returns:
            Tuple of (has_citation, citation_accurate)
        """
        import re

        # Look for citation patterns
        citation_patterns = [
            r'\[.*?\]',  # [source]
            r'מקור:',    # Hebrew: source:
            r'עמוד \d+', # Hebrew: page X
            r'page \d+',
            r'source:',
            r'reference:',
            r'לפי',      # Hebrew: according to
        ]

        has_citation = any(re.search(p, generated, re.IGNORECASE) for p in citation_patterns)

        # For baseline (no RAG), citation accuracy is always False
        # This will be updated when we have actual document retrieval
        citation_accurate = False

        return has_citation, citation_accurate

    def detect_hallucination(
        self,
        generated: str,
        expected: str,
        faithfulness_score: float,
    ) -> bool:
        """
        Detect if the answer is a hallucination.

        A hallucination is when:
        1. The answer contradicts the expected answer
        2. The faithfulness score is low (< 0.5)
        3. The answer contains specific claims not in the expected answer
        """
        # Low faithfulness indicates potential hallucination
        if faithfulness_score < 0.5:
            return True

        # Check for contradiction in yes/no
        expected_lower = expected.lower()
        generated_lower = generated.lower()

        yes_words = ["כן", "yes", "נכון", "מכוסה"]
        no_words = ["לא", "no", "אינו", "לא מכוסה"]

        expected_is_yes = any(w in expected_lower for w in yes_words) and not any(w in expected_lower for w in no_words)
        expected_is_no = any(w in expected_lower for w in no_words)
        generated_is_yes = any(w in generated_lower for w in yes_words) and not any(w in generated_lower for w in no_words)
        generated_is_no = any(w in generated_lower for w in no_words)

        # Contradiction = hallucination
        if (expected_is_yes and generated_is_no) or (expected_is_no and generated_is_yes):
            return True

        return False

    def aggregate_results(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedMetrics:
        """
        Aggregate evaluation results into summary metrics.
        """
        if not results:
            return AggregatedMetrics()

        metrics = AggregatedMetrics()
        metrics.total_questions = len(results)

        # Calculate averages
        metrics.correct_answers = sum(1 for r in results if r.is_correct)
        metrics.accuracy = metrics.correct_answers / metrics.total_questions

        metrics.avg_answer_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        metrics.avg_answer_correctness = sum(r.answer_correctness for r in results) / len(results)
        metrics.avg_faithfulness = sum(r.faithfulness for r in results) / len(results)

        metrics.hallucination_count = sum(1 for r in results if r.is_hallucination)
        metrics.hallucination_rate = metrics.hallucination_count / metrics.total_questions

        metrics.citation_count = sum(1 for r in results if r.has_citation)
        metrics.citation_rate = metrics.citation_count / metrics.total_questions

        citations_with_accuracy = [r for r in results if r.has_citation]
        if citations_with_accuracy:
            metrics.citation_accuracy = sum(1 for r in citations_with_accuracy if r.citation_accurate) / len(citations_with_accuracy)

        metrics.avg_latency_ms = sum(r.latency_ms for r in results) / len(results)

        # Per-domain breakdown
        domains = set(r.domain for r in results)
        for domain in domains:
            domain_results = [r for r in results if r.domain == domain]
            metrics.domain_metrics[domain] = {
                "total": len(domain_results),
                "correct": sum(1 for r in domain_results if r.is_correct),
                "accuracy": sum(1 for r in domain_results if r.is_correct) / len(domain_results),
                "avg_relevancy": sum(r.answer_relevancy for r in domain_results) / len(domain_results),
                "hallucination_rate": sum(1 for r in domain_results if r.is_hallucination) / len(domain_results),
            }

        return metrics

