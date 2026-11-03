"""
Disk-backed SQLite embedding cache for semdiff.
"""

import sqlite3
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class EmbeddingDiskCache:
    """
    Zero-dependency, persistent SQLite cache for sentence embeddings.
    Uses SHA-256 hash of (model_name + text) to store normalized vector blobs.
    """

    def __init__(self, db_path: Path = Path(".cache/embeddings.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentence_embeddings (
                    hash_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON sentence_embeddings(model_name)")

    def _make_key(self, model_name: str, text: str) -> str:
        payload = f"{model_name}:{text.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model_name: str, text: str) -> Optional[np.ndarray]:
        """Fetch single vector from cache, or None if not found."""
        key = self._make_key(model_name, text)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector, dim FROM sentence_embeddings WHERE hash_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                blob, dim = row
                return np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return None

    def get_many(self, model_name: str, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Batch fetch embeddings.
        Returns:
            cached_vectors: list of np.ndarray or None for each input text
            missing_indices: list of indices for texts not found in cache
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: List[int] = []

        if not texts:
            return results, missing_indices

        keys = [self._make_key(model_name, t) for t in texts]
        key_to_idx = {k: i for i, k in enumerate(keys)}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Query in batches to stay well within SQLite variable limits
            batch_size = 500
            for i in range(0, len(keys), batch_size):
                chunk = keys[i:i + batch_size]
                placeholders = ",".join("?" * len(chunk))
                query = f"SELECT hash_key, vector, dim FROM sentence_embeddings WHERE hash_key IN ({placeholders})"
                cursor.execute(query, chunk)
                for h_key, blob, dim in cursor.fetchall():
                    original_idx = key_to_idx[h_key]
                    vec = np.frombuffer(blob, dtype=np.float32).reshape(dim)
                    results[original_idx] = vec

        for idx, vec in enumerate(results):
            if vec is None:
                missing_indices.append(idx)

        return results, missing_indices

    def set(self, model_name: str, text: str, vector: np.ndarray) -> None:
        """Store single embedding into cache."""
        key = self._make_key(model_name, text)
        vec_f32 = np.ascontiguousarray(vector, dtype=np.float32)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sentence_embeddings (hash_key, model_name, text, dim, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, model_name, text.strip(), vec_f32.shape[0], vec_f32.tobytes())
            )

    def set_many(self, model_name: str, texts: List[str], vectors: np.ndarray) -> None:
        """Batch store embeddings into cache."""
        if len(texts) == 0:
            return
        records = []
        for text, vec in zip(texts, vectors):
            key = self._make_key(model_name, text)
            vec_f32 = np.ascontiguousarray(vec, dtype=np.float32)
            records.append((key, model_name, text.strip(), vec_f32.shape[0], vec_f32.tobytes()))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO sentence_embeddings (hash_key, model_name, text, dim, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                records
            )

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sentence_embeddings")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sentence_embeddings")
