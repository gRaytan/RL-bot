import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional

from src.rag import RAGPipeline, RAGConfig
from src.rag.answer_generator import GeneratorConfig

app = FastAPI(title="Completions-Compatible REST API")

# Global RAG pipeline (lazy loaded)
_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton with optimal config."""
    global _pipeline
    if _pipeline is None:
        print("Initializing RAG pipeline...")
        config = RAGConfig(
            retrieval_top_k=50,
            rerank_top_k=15,
            final_context_k=10,
            use_reranker=True,
            use_verification=False,  # Faster without verification
            use_auto_domain=False,   # No domain filtering for best accuracy
            generator_config=GeneratorConfig(
                provider="nebius",
                model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            ),
        )
        _pipeline = RAGPipeline(config=config)
        print("RAG pipeline ready!")
    return _pipeline

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class Source(BaseModel):
    link: str
    page: int
    
class Choice(BaseModel):
    index: int
    text: str
    sources: List[Source] = []
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: float
    model: str
    choices: List[Choice]

# ----------------------
# Endpoints
# ----------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

async def process_completions_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Get the user's question from messages
    question = ""
    for msg in request.messages:
        if msg.role == "user":
            question = msg.content

    print(f"Processing question: {question[:100]}...")

    # Call our RAG pipeline
    pipeline = get_pipeline()
    response = pipeline.query(question, domain_filter=None)

    answer_text = response.answer

    # Build source list from citations
    source_list = []
    for citation in response.citations:
        # Convert source filename to URL format
        source_file = citation.source_file
        # Extract meaningful part of the filename for the link
        if source_file.startswith("data/"):
            source_file = source_file.split("/")[-1]
        link = f"https://www.harel-group.co.il/documents/{source_file}"
        page = citation.page_num if citation.page_num else 1
        source_list.append(Source(link=link, page=page))

    print(f"Answer generated with {len(source_list)} sources")

    return ChatCompletionResponse(
        id="question-1",
        object="chat.completion",
        created=time.time(),
        model=request.model,
        choices=[
            Choice(
                index=0,
                text=answer_text,
                sources=source_list,
                finish_reason="stop"
            )
        ]
    )

# ----------------------
# Run with:
# uvicorn myapp:app --reload
# ----------------------
