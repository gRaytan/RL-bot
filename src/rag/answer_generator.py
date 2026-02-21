"""
Answer Generator - Generate answers with citations from retrieved context.

Uses LLM to synthesize answers from retrieved documents with source citations.
"""

import logging
import os
from typing import Optional
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for answer generator."""
    provider: str = "nebius"  # "nebius" or "openai"
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    temperature: float = 0.0
    max_tokens: int = 1000
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class Citation:
    """A citation to a source document."""
    source_file: str
    page_num: Optional[int]
    chunk_id: str
    text_snippet: str


@dataclass
class GeneratedAnswer:
    """Generated answer with citations."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "medium"  # "high", "medium", "low"
    context_used: int = 0  # Number of context chunks used


SYSTEM_PROMPT = """אתה נציג שירות לקוחות מקצועי של חברת הביטוח הראל.

כללים חשובים:
1. ענה על סמך המידע שמופיע בהקשר (Context) שניתן לך
2. אם השאלה היא כן/לא - ענה בצורה ישירה: "כן" או "לא" ואז הסבר קצר
3. אם השאלה מבקשת מספר או סכום - ציין את המספר בתחילת התשובה
4. אם יש מידע רלוונטי בהקשר - השתמש בו גם אם הוא לא מדויק ב-100%
5. רק אם באמת אין שום מידע רלוונטי בהקשר - ציין זאת
6. תשובות קצרות, ברורות ומקצועיות בעברית

פורמט התשובה:
תשובה: [התשובה שלך - התחל עם כן/לא או המספר המבוקש]

מקורות:
- [שם קובץ], עמוד [מספר]
"""

USER_TEMPLATE = """הקשר (Context):
{context}

---

שאלה: {question}

ענה על השאלה בהתבסס על ההקשר שלמעלה בלבד."""


class AnswerGenerator:
    """
    Generate answers with citations from retrieved context.
    
    Usage:
        generator = AnswerGenerator()
        answer = generator.generate(question, ranked_results)
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        
        # Initialize client
        if self.config.provider == "nebius":
            api_key = self.config.api_key or os.getenv("LLM_API_KEY")
            base_url = self.config.base_url or os.getenv("LLM_BASE_URL", "https://api.studio.nebius.ai/v1")
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = None
        
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"AnswerGenerator initialized: {self.config.provider}/{self.config.model}")

    def generate(
        self,
        question: str,
        context_results: list,  # RankedResult or RetrievalResult
        max_context_chunks: int = 5,
        conversation_history: Optional[list[dict]] = None,
    ) -> GeneratedAnswer:
        """
        Generate an answer from retrieved context.

        Args:
            question: User's question
            context_results: Retrieved and reranked results
            max_context_chunks: Maximum chunks to include in context
            conversation_history: Previous conversation turns

        Returns:
            GeneratedAnswer with answer text and citations
        """
        # Build context string
        context_chunks = context_results[:max_context_chunks]
        context_parts = []
        citations = []
        
        for i, result in enumerate(context_chunks, 1):
            source_file = result.metadata.get("source_filename", "unknown")
            page_num = result.metadata.get("page_num")
            
            context_parts.append(f"[מקור {i}: {source_file}, עמוד {page_num}]\n{result.text}")
            
            citations.append(Citation(
                source_file=source_file,
                page_num=page_num,
                chunk_id=result.id,
                text_snippet=result.text[:100] + "...",
            ))
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        user_message = USER_TEMPLATE.format(context=context_str, question=question)

        # Build messages with conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 3 turns max
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        answer_text = response.choices[0].message.content
        
        # Determine confidence based on context quality
        confidence = self._assess_confidence(context_chunks)
        
        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            context_used=len(context_chunks),
        )

    def _assess_confidence(self, context_chunks: list) -> str:
        """Assess confidence based on context quality."""
        if not context_chunks:
            return "low"
        
        # Check rerank scores if available
        if hasattr(context_chunks[0], "rerank_score"):
            top_score = context_chunks[0].rerank_score
            if top_score > -1.0:
                return "high"
            elif top_score > -3.0:
                return "medium"
            else:
                return "low"
        
        return "medium"

