"""
Unit tests for disk-backed embedding cache and similarity matrix calculations.
"""

import pytest
import numpy as np
from pathlib import Path

from semdiff.embedding.cache import EmbeddingDiskCache
from semdiff.embedding.labse import LaBSEEmbedder


def test_sqlite_embedding_cache(tmp_path: Path):
    db_file = tmp_path / "test_embeddings.db"
    cache = EmbeddingDiskCache(db_path=db_file)

    model_name = "test-model"
    text1 = "Every reporting entity shall maintain records for 5 years."
    text2 = "Banks must verify customer identity."

    vec1 = np.random.randn(768).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    vec2 = np.random.randn(768).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)

    # Test single set/get
    cache.set(model_name, text1, vec1)
    retrieved1 = cache.get(model_name, text1)
    assert retrieved1 is not None
    assert np.allclose(retrieved1, vec1, atol=1e-5)

    # Test batch get with missing element
    cached_vecs, missing_idx = cache.get_many(model_name, [text1, text2])
    assert cached_vecs[0] is not None
    assert cached_vecs[1] is None
    assert missing_idx == [1]

    # Test batch set
    cache.set_many(model_name, [text2], np.array([vec2]))
    cached_vecs_after, missing_after = cache.get_many(model_name, [text1, text2])
    assert missing_after == []
    assert np.allclose(cached_vecs_after[1], vec2, atol=1e-5)
    assert cache.count() == 2


def test_similarity_matrix_computation():
    # Construct 3 normalized vectors
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=np.float32)

    emb_a = np.vstack([v1, v2])
    emb_b = np.vstack([v1, v3])

    sim_matrix = LaBSEEmbedder.compute_similarity_matrix(emb_a, emb_b)

    assert sim_matrix.shape == (2, 2)
    # v1 with v1 -> 1.0
    assert pytest.approx(sim_matrix[0, 0], 1e-5) == 1.0
    # v1 with v3 -> 1/sqrt(2) approx 0.7071
    assert pytest.approx(sim_matrix[0, 1], 1e-4) == 0.7071
    # v2 with v1 -> 0.0 (orthogonal)
    assert pytest.approx(sim_matrix[1, 0], 1e-5) == 0.0
    # v2 with v3 -> 1/sqrt(2) approx 0.7071
    assert pytest.approx(sim_matrix[1, 1], 1e-4) == 0.7071
