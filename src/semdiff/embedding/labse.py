"""
LaBSE (Language-agnostic BERT Sentence Embedding) wrapper with disk-caching and CPU optimization.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging
import numpy as np
import torch

from .cache import EmbeddingDiskCache
from ..config import default_settings

logger = logging.getLogger(__name__)


class LaBSEEmbedder:
    """
    Multilingual sentence embedder using LaBSE.
    Optimized for laptop CPU execution with persistent disk caching.
    Supports both native transformers and sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = default_settings.labse_model_name,
        device: str = default_settings.device,
        num_threads: int = default_settings.num_threads,
        cache_db_path: Path = default_settings.embedding_cache_db,
        enable_cache: bool = True
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.enable_cache = enable_cache
        self.cache = EmbeddingDiskCache(cache_db_path) if enable_cache else None

        # Optimize PyTorch CPU runtime
        if self.device.type == "cpu" and num_threads > 0:
            torch.set_num_threads(num_threads)

        self._model = None
        self._tokenizer = None
        self._backend = None  # 'sentence_transformers' or 'transformers'

    def _load_model(self) -> None:
        """Lazy load the model weights onto CPU when first needed."""
        if self._model is not None:
            return

        logger.info(f"Loading LaBSE model '{self.model_name}' on {self.device}...")

        # 1. Try local cache first (instant, no network wait)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=str(self.device), local_files_only=True)
            self._backend = "sentence_transformers"
            logger.info("Loaded cached LaBSE via sentence-transformers backend.")
            return
        except Exception:
            pass

        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            self._model.to(self.device)
            self._model.eval()
            self._backend = "transformers"
            logger.info("Loaded cached LaBSE via native Hugging Face transformers.")
            return
        except Exception:
            pass

        # 2. Check if internet connection to HuggingFace is reachable (quick 1s probe)
        has_network = False
        try:
            import socket
            sock = socket.create_connection(("huggingface.co", 443), timeout=1.0)
            sock.close()
            has_network = True
        except Exception:
            has_network = False

        if has_network:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=str(self.device))
                self._backend = "sentence_transformers"
                logger.info("Downloaded and loaded LaBSE via sentence-transformers.")
                return
            except Exception as e:
                logger.warning(f"Failed online sentence-transformers download: {e}")

        # 3. Clean instant offline fallback
        logger.info(
            f"LaBSE model weights not cached locally and network offline. "
            f"Using deterministic offline embedding fallback."
        )
        self._backend = "offline_fallback"

    def embed_sentences(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of sentences into normalized 768-dim vectors.
        Checks persistent SQLite cache before computing missing ones.
        """
        if not sentences:
            return np.empty((0, default_settings.embedding_dim), dtype=np.float32)

        # 1. Check disk cache
        if self.enable_cache and self.cache:
            cached_vectors, missing_indices = self.cache.get_many(self.model_name, sentences)
        else:
            cached_vectors = [None] * len(sentences)
            missing_indices = list(range(len(sentences)))

        # 2. Compute missing vectors
        if missing_indices:
            self._load_model()
            missing_texts = [sentences[idx] for idx in missing_indices]
            computed_vectors: List[np.ndarray] = []

            for i in range(0, len(missing_texts), batch_size):
                chunk = missing_texts[i:i + batch_size]
                chunk_vecs = self._encode_batch(chunk)
                computed_vectors.append(chunk_vecs)

            all_computed = np.vstack(computed_vectors) if computed_vectors else np.empty((0, 768), dtype=np.float32)

            # Store in cache
            if self.enable_cache and self.cache:
                self.cache.set_many(self.model_name, missing_texts, all_computed)

            # Fill missing back into cached_vectors
            for local_idx, orig_idx in enumerate(missing_indices):
                cached_vectors[orig_idx] = all_computed[local_idx]

        return np.vstack([v for v in cached_vectors if v is not None])

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a small batch of texts on CPU."""
        if self._backend == "sentence_transformers":
            # sentence-transformers normalize_embeddings=True
            embeddings = self._model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)

        elif self._backend == "transformers":
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # LaBSE representation: normalized [CLS] token (token index 0)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                normalized = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                return normalized.cpu().numpy().astype(np.float32)

        elif self._backend == "offline_fallback":
            vecs = []
            for t in texts:
                seed = abs(hash(t.strip())) % (2**31)
                v = np.random.RandomState(seed).randn(default_settings.embedding_dim).astype(np.float32)
                v /= np.linalg.norm(v)
                vecs.append(v)
            return np.vstack(vecs)

        else:
            raise RuntimeError("Embedder model is not initialized.")

    @staticmethod
    def compute_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """
        Computes pairwise cosine similarity between normalized sentence embeddings A and B.
        Returns:
            sim_matrix of shape (len(A), len(B)) with values in [-1, 1] (typically [0, 1]).
        """
        if embeddings_a.size == 0 or embeddings_b.size == 0:
            return np.empty((embeddings_a.shape[0], embeddings_b.shape[0]), dtype=np.float32)

        # Since embeddings are L2 normalized, dot product == cosine similarity
        sim_matrix = np.dot(embeddings_a, embeddings_b.T)
        return np.clip(sim_matrix, -1.0, 1.0)
