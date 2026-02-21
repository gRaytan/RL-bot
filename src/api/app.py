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
from src.rag import RAGPipeline, RAGConfig


# Global RAG pipeline (lazy loaded)
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        print("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline(RAGConfig())
        print("RAG pipeline ready.")
    return _rag_pipeline


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
        
        # Get or create session
        session = session_store.get_or_create_session(request.session_id)
        
        # Add user message to history
        session.add_message("user", request.message)
        
        try:
            # Get RAG pipeline
            pipeline = get_rag_pipeline()
            
            # Get conversation history for context
            history = session.get_history(max_turns=3)
            
            # Query RAG pipeline
            result = pipeline.query(
                question=request.message,
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
    
    return app


# Create app instance
app = create_app()

