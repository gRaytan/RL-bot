"""
Baseline runner for evaluating GPT-4o and GPT-5.2 on the dev set.
Runs models without RAG to establish baseline performance.
"""

import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from .metrics import EvaluationMetrics, EvaluationResult, AggregatedMetrics


@dataclass
class PromptStrategy:
    """Configuration for a prompt strategy."""
    name: str
    system_prompt: str
    user_template: str
    
    def format_user_prompt(self, question: str, domain: str) -> str:
        """Format the user prompt with the question and domain."""
        return self.user_template.format(question=question, domain=domain)


# Define prompt strategies to test
PROMPT_STRATEGIES = {
    "basic": PromptStrategy(
        name="basic",
        system_prompt="אתה נציג שירות לקוחות של חברת הביטוח הראל. ענה על שאלות הלקוחות בצורה מקצועית ומדויקת.",
        user_template="{question}",
    ),
    "domain_aware": PromptStrategy(
        name="domain_aware",
        system_prompt="""אתה נציג שירות לקוחות של חברת הביטוח הראל. 
ענה על שאלות הלקוחות בצורה מקצועית ומדויקת.
אם אינך בטוח בתשובה, ציין זאת במפורש.
תמיד נסה לספק מקור או הפניה לפוליסה הרלוונטית.""",
        user_template="תחום: {domain}\n\nשאלה: {question}",
    ),
    "strict_grounding": PromptStrategy(
        name="strict_grounding",
        system_prompt="""אתה נציג שירות לקוחות של חברת הביטוח הראל.
כללים חשובים:
1. ענה רק על סמך מידע שאתה בטוח בו לגבי פוליסות הראל
2. אם אינך בטוח, אמור "אני לא בטוח" או "יש לבדוק בפוליסה"
3. אל תמציא מידע
4. ציין תמיד את המקור אם ידוע לך
5. תשובות קצרות וממוקדות""",
        user_template="תחום ביטוח: {domain}\n\nשאלת לקוח: {question}\n\nתשובה:",
    ),
    "citation_required": PromptStrategy(
        name="citation_required",
        system_prompt="""אתה נציג שירות לקוחות של חברת הביטוח הראל.
חובה לציין מקור לכל תשובה.
אם אין לך מקור מדויק, ציין "מקור: לא ידוע - יש לאמת מול הפוליסה".
פורמט תשובה:
תשובה: [התשובה]
מקור: [שם המסמך/עמוד]""",
        user_template="תחום: {domain}\nשאלה: {question}",
    ),
}


class BaselineRunner:
    """
    Runs baseline evaluation on GPT models without RAG.
    Tests multiple prompt strategies and collects metrics.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
        strategies: list[str] | None = None,
    ):
        """
        Initialize the baseline runner.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            models: List of models to test (default: gpt-4o, gpt-4-turbo)
            strategies: List of prompt strategies to test
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.models = models or ["gpt-4o", "gpt-4-turbo"]
        self.strategies = strategies or list(PROMPT_STRATEGIES.keys())
        self.metrics = EvaluationMetrics()
    
    def load_dataset(self, dataset_path: str | Path) -> dict[str, list[dict]]:
        """Load the evaluation dataset from JSON."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _flatten_dataset(self, dataset: dict[str, list[dict]]) -> list[dict]:
        """Flatten the domain-grouped dataset into a list of questions."""
        questions = []
        for domain, items in dataset.items():
            for item in items:
                questions.append({
                    "domain": domain,
                    "question": item["שאלה"],
                    "expected_answer": item["תשובה"],
                    "source_file": item["מקור"]["קובץ"],
                    "source_page": item["מקור"]["עמוד"],
                })
        return questions
    
    def _call_model(
        self,
        model: str,
        strategy: PromptStrategy,
        question: str,
        domain: str,
    ) -> tuple[str, float]:
        """
        Call the model and return the response with latency.
        
        Returns:
            Tuple of (response_text, latency_ms)
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": strategy.system_prompt},
                    {"role": "user", "content": strategy.format_user_prompt(question, domain)},
                ],
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=500,
            )
            latency_ms = (time.time() - start_time) * 1000
            return response.choices[0].message.content, latency_ms
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return f"Error: {str(e)}", latency_ms

    def evaluate_single_question(
        self,
        model: str,
        strategy: PromptStrategy,
        question_data: dict,
        use_ragas: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single question with a specific model and strategy.
        """
        question = question_data["question"]
        expected = question_data["expected_answer"]
        domain = question_data["domain"]

        # Get model response
        generated, latency_ms = self._call_model(model, strategy, question, domain)

        # Create result object
        result = EvaluationResult(
            question=question,
            expected_answer=expected,
            generated_answer=generated,
            domain=domain,
            source_file=question_data["source_file"],
            source_page=question_data["source_page"],
            latency_ms=latency_ms,
            model=model,
            prompt_strategy=strategy.name,
        )

        # Calculate custom metrics
        result.is_correct = self.metrics.check_correctness(expected, generated)
        result.has_citation, result.citation_accurate = self.metrics.check_citation(generated)

        # RAGAS evaluation (optional, can be slow)
        if use_ragas:
            try:
                ragas_scores = self.metrics.evaluate_single(
                    question=question,
                    expected_answer=expected,
                    generated_answer=generated,
                    context=[],  # No context for baseline
                )
                result.answer_relevancy = ragas_scores.get("answer_relevancy", 0.0)
                result.answer_correctness = ragas_scores.get("answer_correctness", 0.0)
                result.faithfulness = ragas_scores.get("faithfulness", 0.0)
            except Exception as e:
                print(f"RAGAS error for question: {e}")

        # Detect hallucination
        result.is_hallucination = self.metrics.detect_hallucination(
            generated, expected, result.faithfulness
        )

        return result

    def run_evaluation(
        self,
        dataset_path: str | Path,
        output_dir: str | Path = "data/evaluation/results",
        use_ragas: bool = True,
        verbose: bool = True,
    ) -> dict[str, dict[str, AggregatedMetrics]]:
        """
        Run full baseline evaluation across all models and strategies.

        Args:
            dataset_path: Path to the evaluation dataset JSON
            output_dir: Directory to save results
            use_ragas: Whether to use RAGAS metrics (slower but more accurate)
            verbose: Print progress

        Returns:
            Nested dict: {model: {strategy: AggregatedMetrics}}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and flatten dataset
        dataset = self.load_dataset(dataset_path)
        questions = self._flatten_dataset(dataset)

        if verbose:
            print(f"Loaded {len(questions)} questions from {len(dataset)} domains")
            print(f"Models: {self.models}")
            print(f"Strategies: {self.strategies}")
            print("-" * 50)

        all_results: dict[str, dict[str, AggregatedMetrics]] = {}
        all_detailed_results: list[dict] = []

        for model in self.models:
            all_results[model] = {}

            for strategy_name in self.strategies:
                strategy = PROMPT_STRATEGIES[strategy_name]
                results: list[EvaluationResult] = []

                if verbose:
                    print(f"\nEvaluating: {model} with {strategy_name} strategy")

                # Evaluate each question
                iterator = tqdm(questions, desc=f"{model}/{strategy_name}") if verbose else questions
                for q in iterator:
                    result = self.evaluate_single_question(
                        model=model,
                        strategy=strategy,
                        question_data=q,
                        use_ragas=use_ragas,
                    )
                    results.append(result)
                    all_detailed_results.append(result.to_dict())

                # Aggregate results
                aggregated = self.metrics.aggregate_results(results)
                all_results[model][strategy_name] = aggregated

                if verbose:
                    print(f"  Accuracy: {aggregated.accuracy:.2%}")
                    print(f"  Hallucination Rate: {aggregated.hallucination_rate:.2%}")
                    print(f"  Citation Rate: {aggregated.citation_rate:.2%}")
                    print(f"  Avg Latency: {aggregated.avg_latency_ms:.0f}ms")

        # Save detailed results
        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(all_detailed_results, f, ensure_ascii=False, indent=2)

        # Save aggregated results
        aggregated_path = output_dir / "aggregated_results.json"
        aggregated_data = {
            model: {
                strategy: metrics.to_dict()
                for strategy, metrics in strategies.items()
            }
            for model, strategies in all_results.items()
        }
        with open(aggregated_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_data, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\nResults saved to {output_dir}")

        return all_results

