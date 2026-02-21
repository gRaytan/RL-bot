"""FastAPI application for Harel Insurance Chatbot."""

import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ChatRequest, 
    ChatResponse, 
    Citation,
    HealthResponse,
    SessionInfo,
    ErrorResponse,
)
from .session_store import session_store
from .guardrails import validate_input, validate_output, is_insurance_related, get_off_topic_response
from src.rag import RAGPipeline, RAGConfig
from src.agents import InsuranceAgent


# Global singletons (lazy loaded)
_rag_pipeline: Optional[RAGPipeline] = None
_agent: Optional[InsuranceAgent] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        print("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline(RAGConfig())
        print("RAG pipeline ready.")
    return _rag_pipeline


def get_agent() -> InsuranceAgent:
    """Get or create agent singleton."""
    global _agent
    if _agent is None:
        print("Initializing Insurance Agent...")
        _agent = InsuranceAgent(rag_pipeline=get_rag_pipeline())
        print("Agent ready.")
    return _agent


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Harel Insurance Chatbot API",
        description="Customer support chatbot for Harel Insurance powered by RAG",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check API and component health."""
        components = {
            "api": "healthy",
            "sessions": session_store.session_count,
        }
        
        # Check RAG pipeline
        try:
            pipeline = get_rag_pipeline()
            components["rag_pipeline"] = "healthy"
            components["vector_store"] = "healthy"
        except Exception as e:
            components["rag_pipeline"] = f"unhealthy: {str(e)}"
        
        status = "healthy" if components.get("rag_pipeline") == "healthy" else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            timestamp=datetime.utcnow(),
            components=components,
        )
    
    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """
        Send a message and get a response.

        - Creates new session if session_id not provided
        - Maintains conversation history within session
        - Returns answer with citations from insurance documents
        """
        start_time = time.time()

        # Validate input
        input_validation = validate_input(request.message)
        if not input_validation.is_valid:
            raise HTTPException(status_code=400, detail=input_validation.error_message)

        message = input_validation.sanitized_text

        # Get or create session
        session = session_store.get_or_create_session(request.session_id)

        # Add user message to history
        session.add_message("user", message)

        try:
            # Get RAG pipeline
            pipeline = get_rag_pipeline()
            
            # Get conversation history for context
            history = session.get_history(max_turns=3)
            
            # Query RAG pipeline
            result = pipeline.query(
                question=message,
                conversation_history=history[:-1],  # Exclude current message
            )

            # Parse citations from RAGResponse dataclass
            citations = [
                Citation(
                    source=c.source_file,
                    page=c.page_num,
                    relevance_score=None,
                )
                for c in result.citations
            ]

            answer = result.answer or "מצטער, לא הצלחתי למצוא תשובה."
            confidence = result.confidence

            # Validate and sanitize output
            output_validation = validate_output(answer)
            answer = output_validation.sanitized_text

            # Add assistant response to history
            session.add_message("assistant", answer, [c.dict() for c in citations])
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                session_id=session.session_id,
                answer=answer,
                citations=citations,
                confidence=confidence,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )
    
    @app.get("/sessions", response_model=list[SessionInfo], tags=["Sessions"])
    async def list_sessions():
        """List all active sessions."""
        sessions = session_store.get_all_sessions()
        return [
            SessionInfo(
                session_id=s.session_id,
                created_at=s.created_at,
                last_activity=s.last_activity,
                message_count=s.message_count,
            )
            for s in sessions
        ]
    
    @app.get("/sessions/{session_id}", tags=["Sessions"])
    async def get_session(session_id: str):
        """Get session details and conversation history."""
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "message_count": session.message_count,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
                for m in session.messages
            ]
        }

    @app.post("/agent/chat", tags=["Agent"])
    async def agent_chat(request: ChatRequest):
        """
        Chat with the insurance agent (with tool calling).

        The agent can decide when to search policy documents.
        """
        start_time = time.time()

        # Validate input
        input_validation = validate_input(request.message)
        if not input_validation.is_valid:
            raise HTTPException(status_code=400, detail=input_validation.error_message)

        message = input_validation.sanitized_text

        # Get or create session
        session = session_store.get_or_create_session(request.session_id)
        session.add_message("user", message)

        try:
            agent = get_agent()
            history = session.get_history(max_turns=3)

            result = agent.chat(
                message=message,
                conversation_history=history[:-1],
            )

            answer = result.get("answer", "מצטער, לא הצלחתי לענות.")

            # Validate and sanitize output
            output_validation = validate_output(answer)
            answer = output_validation.sanitized_text

            session.add_message("assistant", answer)

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "session_id": session.session_id,
                "answer": answer,
                "tool_calls": result.get("tool_calls", []),
                "latency_ms": latency_ms,
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Agent error: {str(e)}"
            )

    return app


# Create app instance
app = create_app()

