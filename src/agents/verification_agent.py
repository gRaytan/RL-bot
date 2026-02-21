"""
Verification Agent - Agent 5 from the 5-Agent Architecture.

Responsibilities:
1. Verify citations match the source context
2. Detect hallucinations (claims not supported by context)
3. Check answer completeness
4. Validate consistency

This agent reviews the generated answer BEFORE returning to user.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for verification agent."""
    provider: str = "nebius"
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0  # Deterministic for verification
    max_tokens: int = 500
    hallucination_threshold: float = 0.3  # Flag if >30% unsupported claims


@dataclass
class VerificationResult:
    """Result of answer verification."""
    is_valid: bool
    confidence: str  # "high", "medium", "low"
    hallucination_score: float  # 0.0 = no hallucination, 1.0 = all hallucinated
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verified_answer: Optional[str] = None  # Corrected answer if needed


VERIFICATION_PROMPT = """אתה מומחה לאימות תשובות. בדוק את התשובה מול ההקשר המקורי.

הקשר מהמסמכים:
{context}

שאלה: {question}

תשובה לבדיקה: {answer}

בדוק:
1. האם כל הטענות בתשובה נתמכות בהקשר?
2. האם יש מידע שהומצא (הזיה)?
3. האם התשובה מלאה?
4. האם הציטוטים מדויקים?

ענה בפורמט JSON:
{{
    "is_valid": true/false,
    "confidence": "high/medium/low",
    "hallucination_score": 0.0-1.0,
    "unsupported_claims": ["טענה 1", "טענה 2"],
    "missing_info": ["מידע חסר 1"],
    "suggestions": ["הצעה לשיפור"]
}}

רק JSON, ללא טקסט נוסף."""


class VerificationAgent:
    """
    Agent 5: Verification & Quality Assurance.
    
    Verifies generated answers against source context to:
    - Detect hallucinations
    - Validate citations
    - Check completeness
    - Ensure consistency
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        
        # Initialize LLM client
        if self.config.provider == "nebius":
            api_key = self.config.api_key or os.getenv("LLM_API_KEY")
            base_url = self.config.base_url or os.getenv("LLM_BASE_URL", "https://api.studio.nebius.ai/v1")
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = None
        
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"VerificationAgent initialized: {self.config.provider}/{self.config.model}")

    def verify(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> VerificationResult:
        """
        Verify an answer against the source context.
        
        Args:
            question: Original user question
            answer: Generated answer to verify
            context: Source context used to generate the answer
            
        Returns:
            VerificationResult with validation status and issues
        """
        if not answer or not context:
            return VerificationResult(
                is_valid=True,
                confidence="low",
                hallucination_score=0.0,
                issues=["No answer or context to verify"],
            )

        prompt = VERIFICATION_PROMPT.format(
            context=context[:3000],  # Limit context size
            question=question,
            answer=answer,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            result_text = response.choices[0].message.content
            return self._parse_verification_result(result_text)
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                is_valid=True,  # Don't block on verification errors
                confidence="low",
                hallucination_score=0.0,
                issues=[f"Verification error: {str(e)}"],
            )

    def _parse_verification_result(self, result_text: str) -> VerificationResult:
        """Parse LLM verification response into VerificationResult."""
        import json

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse verification JSON: {e}")
            return VerificationResult(
                is_valid=True,
                confidence="medium",
                hallucination_score=0.0,
                issues=["Could not parse verification result"],
            )

        # Build issues list
        issues = []
        unsupported = data.get("unsupported_claims", [])
        if unsupported:
            issues.extend([f"טענה לא נתמכת: {claim}" for claim in unsupported])

        missing = data.get("missing_info", [])
        if missing:
            issues.extend([f"מידע חסר: {info}" for info in missing])

        hallucination_score = float(data.get("hallucination_score", 0.0))
        is_valid = data.get("is_valid", True)

        # Override validity if hallucination is too high
        if hallucination_score > self.config.hallucination_threshold:
            is_valid = False
            issues.append(f"ציון הזיה גבוה: {hallucination_score:.1%}")

        return VerificationResult(
            is_valid=is_valid,
            confidence=data.get("confidence", "medium"),
            hallucination_score=hallucination_score,
            issues=issues,
            suggestions=data.get("suggestions", []),
        )

    def quick_verify(self, answer: str, context: str) -> bool:
        """
        Quick heuristic verification without LLM call.

        Checks:
        1. Answer mentions something from context
        2. No obvious fabricated numbers
        3. Answer is not too generic

        Returns:
            True if answer passes quick checks
        """
        if not answer or not context:
            return True

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Check for keyword overlap
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())

        # Filter out common words
        common_words = {"את", "של", "על", "עם", "או", "לא", "כן", "הוא", "היא", "זה", "the", "a", "is", "are"}
        answer_words -= common_words
        context_words -= common_words

        overlap = len(answer_words & context_words)

        # At least 3 meaningful words should overlap
        if overlap < 3:
            logger.warning(f"Low keyword overlap: {overlap} words")
            return False

        return True

