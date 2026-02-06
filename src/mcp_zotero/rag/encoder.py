"""Sentence transformer embedding encoder."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Wrapper for sentence-transformers embedding models."""

    MODELS = {
        "all-MiniLM-L6-v2": {
            "size_mb": 80,
            "dimensions": 384,
            "description": "Fast, good quality (default)",
        },
        "all-mpnet-base-v2": {
            "size_mb": 420,
            "dimensions": 768,
            "description": "Best quality, slower",
        },
        "paraphrase-MiniLM-L6-v2": {
            "size_mb": 80,
            "dimensions": 384,
            "description": "Good for paraphrase detection",
        },
    }

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimensions: Optional[int] = None

    @property
    def model(self):
        """Get the sentence transformer model (lazy-loaded)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded: {self.model_name} "
                f"(dim={self._dimensions}, device={self._model.device})"
            )
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            _ = self.model
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def get_model_info(self) -> dict:
        info = {"model_name": self.model_name, "is_loaded": self.is_loaded}
        if self.model_name in self.MODELS:
            info.update(self.MODELS[self.model_name])
        if self.is_loaded:
            info["dimensions"] = self._dimensions
            info["device"] = str(self._model.device)
        return info
