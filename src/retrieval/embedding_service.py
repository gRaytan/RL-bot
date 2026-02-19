"""
Embedding Service - Generate embeddings for text using OpenAI or local models.

Supports:
- OpenAI text-embedding-3-large (3072 dims) - best quality
- OpenAI text-embedding-3-small (1536 dims) - faster/cheaper
- Local models via sentence-transformers (for offline/privacy)
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    model: str = "text-embedding-3-large"
    dimension: int = 3072
    batch_size: int = 100
    api_key: Optional[str] = None


class EmbeddingService:
    """
    Generate embeddings for text chunks.
    
    Usage:
        service = EmbeddingService()
        embedding = service.embed("מה הכיסוי לגניבת רכב?")
        embeddings = service.embed_batch(["text1", "text2", ...])
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set - embedding service unavailable")
                return
            
            self._client = OpenAI(api_key=api_key)
            logger.info(f"EmbeddingService initialized: model={self.config.model}, dim={self.config.dimension}")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    @property
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self._client is not None

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        if not self._client:
            raise RuntimeError("Embedding service not available - check OPENAI_API_KEY")
        
        # Clean text
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * self.config.dimension
        
        try:
            response = self._client.embeddings.create(
                model=self.config.model,
                input=text,
                dimensions=self.config.dimension,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_batch(
        self, 
        texts: list[str], 
        show_progress: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not self._client:
            raise RuntimeError("Embedding service not available - check OPENAI_API_KEY")
        
        embeddings = []
        batch_size = self.config.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Clean texts
            cleaned = [t.replace("\n", " ").strip() or " " for t in batch]
            
            try:
                response = self._client.embeddings.create(
                    model=self.config.model,
                    input=cleaned,
                    dimensions=self.config.dimension,
                )
                
                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
                
                if show_progress:
                    logger.info(f"Embedded batch {batch_num}/{total_batches} ({len(embeddings)}/{len(texts)} texts)")
                    
            except Exception as e:
                logger.error(f"Batch embedding failed at batch {batch_num}: {e}")
                raise
        
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension

