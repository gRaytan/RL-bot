"""In-memory session store for conversation history."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import uuid
import threading


@dataclass
class Message:
    """Single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    citations: list = field(default_factory=list)


@dataclass 
class Session:
    """Conversation session with history."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages: list[Message] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, citations: list = None):
        """Add a message to the conversation."""
        self.messages.append(Message(
            role=role,
            content=content,
            citations=citations or []
        ))
        self.last_activity = datetime.utcnow()
    
    def get_history(self, max_turns: int = 5) -> list[dict]:
        """Get recent conversation history for context."""
        recent = self.messages[-(max_turns * 2):]  # Last N turns (user + assistant)
        return [{"role": m.role, "content": m.content} for m in recent]
    
    @property
    def message_count(self) -> int:
        return len(self.messages)


class SessionStore:
    """Thread-safe in-memory session store."""
    
    def __init__(self, ttl_minutes: int = 60, max_sessions: int = 1000):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_sessions = max_sessions
    
    def create_session(self) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        
        with self._lock:
            # Cleanup old sessions if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired()
            
            self._sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get existing session or None if not found/expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                return None
            
            # Check if expired
            if datetime.utcnow() - session.last_activity > self._ttl:
                del self._sessions[session_id]
                return None
            
            return session
    
    def get_or_create_session(self, session_id: Optional[str]) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session()
    
    def _cleanup_expired(self):
        """Remove expired sessions (called with lock held)."""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
    
    def get_all_sessions(self) -> list[Session]:
        """Get all active sessions."""
        with self._lock:
            self._cleanup_expired()
            return list(self._sessions.values())
    
    @property
    def session_count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Global session store instance
session_store = SessionStore()

