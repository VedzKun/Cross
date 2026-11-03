"""
Configuration settings for semdiff.
"""

from pathlib import Path
from pydantic import BaseModel, Field
import torch


class Settings(BaseModel):
    # Device configuration - optimized for laptop CPU
    device: str = "cpu"
    num_threads: int = 4

    # Embedding model
    labse_model_name: str = "sentence-transformers/LaBSE"
    embedding_dim: int = 768
    batch_size: int = 32

    # DistilBERT Classification model
    classifier_model_name: str = "distilbert-base-multilingual-cased"
    max_seq_length: int = 256
    num_classes: int = 3  # cosmetic, substantive, critical

    # Alignment DP parameters
    sim_threshold: float = 0.50
    gap_open_penalty: float = 0.35
    gap_extend_penalty: float = 0.15
    split_merge_threshold: float = 0.58

    # Cache paths
    cache_dir: Path = Path(".cache")
    embedding_cache_db: Path = Path(".cache/embeddings.db")

    # Output paths
    reports_dir: Path = Path("reports")
    data_dir: Path = Path("data")

    def ensure_directories(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


default_settings = Settings()
