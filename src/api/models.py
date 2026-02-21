"""Pydantic models for API request/response."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity. Creates new session if not provided."
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question in Hebrew"
    )


class Citation(BaseModel):
    """Citation reference to source document."""
    
    source: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(default=None, description="Page number if available")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score 0-1")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    session_id: str = Field(..., description="Session ID for this conversation")
    answer: str = Field(..., description="Generated answer in Hebrew")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    confidence: Optional[str] = Field(default=None, description="Confidence level: high/medium/low")
    latency_ms: int = Field(..., description="Response time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status: healthy/unhealthy")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict = Field(default_factory=dict, description="Component health status")


class SessionInfo(BaseModel):
    """Information about a conversation session."""
    
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional details")

