from typing import List, Optional
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class CustomEmbeddingds(Embeddings):
    def __init__(
            self,
            model_name: str = "all-mpnet-base-v2",
            device: Optional[str] = None,
            normalize: bool = True,
            batch_size: int = 32,
    ):
        # Auto-detect device if not set
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching and optional normalization."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            batch_size=1
        )
        return embedding[0].tolist()
