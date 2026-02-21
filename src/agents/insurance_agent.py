"""Insurance Agent with tool calling capabilities."""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from src.rag import RAGPipeline, RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the insurance agent."""
    provider: str = "nebius"
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000


# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search insurance policy documents for information. Use this when you need to find specific policy details, coverage information, deductibles, or any insurance-related facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in Hebrew. Be specific about what information you need."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """אתה נציג שירות לקוחות מקצועי של חברת הביטוח הראל.

יש לך גישה לכלי חיפוש במסמכי הפוליסות. השתמש בו כדי למצוא מידע מדויק.

כללים:
1. תמיד השתמש בכלי search_policy כדי לחפש מידע בפוליסות
2. ענה בעברית בצורה ברורה ומקצועית
3. אם השאלה היא כן/לא - ענה ישירות ואז הסבר
4. ציין את המקורות בסוף התשובה
5. אם אין מידע - אמור זאת בבירור
"""


class InsuranceAgent:
    """
    Insurance agent with tool calling capabilities.

    The agent can:
    1. Understand user questions
    2. Decide when to search policy documents
    3. Generate answers based on retrieved context
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
    ):
        self.config = config or AgentConfig()

        # Initialize LLM client
        if self.config.provider == "nebius":
            api_key = self.config.api_key or os.getenv("LLM_API_KEY")
            base_url = self.config.base_url or os.getenv("LLM_BASE_URL", "https://api.studio.nebius.ai/v1")
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = None

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        # Initialize RAG pipeline
        self.rag = rag_pipeline or RAGPipeline(RAGConfig())

        logger.info(f"InsuranceAgent initialized: {self.config.provider}/{self.config.model}")

    def _search_policy(self, query: str) -> str:
        """Execute policy search using RAG pipeline."""
        result = self.rag.query(query)

        # Format results for the agent
        if not result.citations:
            return "לא נמצא מידע רלוונטי במסמכי הפוליסות."

        context_parts = []
        for i, citation in enumerate(result.citations[:5], 1):
            context_parts.append(f"[מקור {i}: {citation.source_file}, עמוד {citation.page_num}]")

        return f"{result.answer}\n\nמקורות:\n" + "\n".join(context_parts)

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name == "search_policy":
            return self._search_policy(arguments.get("query", ""))
        else:
            return f"Unknown tool: {tool_name}"

    def chat(
        self,
        message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Process a user message and generate a response.

        Args:
            message: User's message
            conversation_history: Previous conversation turns

        Returns:
            dict with 'answer', 'tool_calls', 'citations'
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append({"role": "user", "content": message})

        # First call - may include tool calls
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message
        tool_calls_made = []

        # Process tool calls if any
        if assistant_message.tool_calls:
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                tool_calls_made.append({
                    "tool": tool_name,
                    "arguments": arguments,
                })

                # Execute tool
                tool_result = self._execute_tool(tool_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            # Second call - generate final response with tool results
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_answer = response.choices[0].message.content
        else:
            final_answer = assistant_message.content

        return {
            "answer": final_answer,
            "tool_calls": tool_calls_made,
        }

