"""
DistilBERT multilingual classifier for sentence-pair severity classification.

Labels: cosmetic (0), substantive (1), critical (2)
Input: "[CLS] sentence_a [SEP] sentence_b"
Backbone: distilbert-base-multilingual-cased
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

LABEL2ID = {"cosmetic": 0, "substantive": 1, "critical": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


class PairDataset(Dataset):
    """PyTorch Dataset for sentence pair classification."""

    def __init__(
        self,
        texts_a: List[str],
        texts_b: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 256,
    ):
        assert len(texts_a) == len(texts_b), "texts_a and texts_b must have equal length"
        if labels is not None:
            assert len(labels) == len(texts_a)

        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class SeverityClassifier:
    """
    Fine-tunable DistilBERT-based sentence-pair classifier.

    Features:
    - Lazy model loading (only when train/predict is called)
    - Class-weighted cross-entropy loss for imbalanced data
    - Save/load checkpoint
    - CPU-optimized inference
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 256,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models/classifier")
        self._model = None
        self._tokenizer = None

    def _load_backbone(self) -> None:
        if self._model is not None:
            return
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        logger.info(f"Loading DistilBERT backbone: {self.model_name}")
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self._model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self._model.to(self.device)

    def train(
        self,
        texts_a: List[str],
        texts_b: List[str],
        labels: List[int],
        num_epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        class_weights: Optional[List[float]] = None,
        val_texts_a: Optional[List[str]] = None,
        val_texts_b: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> List[Dict[str, float]]:
        """Fine-tune on training pairs. Returns per-epoch loss history."""
        self._load_backbone()
        self._model.train()

        dataset = PairDataset(texts_a, texts_b, labels, self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Class weights for imbalanced data
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(batch_labels)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

            scheduler.step()
            avg_loss = total_loss / total
            acc = correct / total
            epoch_stats = {"epoch": epoch + 1, "train_loss": avg_loss, "train_acc": acc}

            # Validation
            if val_texts_a and val_labels:
                val_acc, val_loss = self._evaluate(
                    val_texts_a, val_texts_b or val_texts_a,
                    val_labels, batch_size, loss_fn
                )
                epoch_stats["val_loss"] = val_loss
                epoch_stats["val_acc"] = val_acc

            logger.info(f"Epoch {epoch+1}/{num_epochs} — {epoch_stats}")
            history.append(epoch_stats)

        return history

    def _evaluate(
        self,
        texts_a: List[str],
        texts_b: List[str],
        labels: List[int],
        batch_size: int,
        loss_fn,
    ) -> Tuple[float, float]:
        self._model.eval()
        dataset = PairDataset(texts_a, texts_b, labels, self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, batch_labels)
                total_loss += loss.item() * len(batch_labels)
                correct += (outputs.logits.argmax(-1) == batch_labels).sum().item()
                total += len(batch_labels)
        self._model.train()
        return correct / total, total_loss / total

    def predict(
        self,
        texts_a: List[str],
        texts_b: List[str],
        batch_size: int = 32,
    ) -> Tuple[List[str], List[float]]:
        """
        Predict severity labels and confidence scores.
        Returns (label_names, confidences).
        """
        self._load_backbone()
        self._model.eval()

        dataset = PairDataset(texts_a, texts_b, None, self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_labels: List[str] = []
        all_confs: List[float] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                confs, pred_ids = probs.max(dim=-1)
                for pid, conf in zip(pred_ids.tolist(), confs.tolist()):
                    all_labels.append(ID2LABEL[pid])
                    all_confs.append(round(conf, 4))

        return all_labels, all_confs

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model and tokenizer to disk."""
        save_dir = Path(path) if path else self.checkpoint_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        logger.info(f"Classifier saved to {save_dir}")
        return save_dir

    def load(self, path: Optional[Path] = None) -> None:
        """Load fine-tuned model from disk."""
        load_dir = Path(path) if path else self.checkpoint_dir
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        logger.info(f"Loading classifier from {load_dir}")
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(load_dir))
        self._model = DistilBertForSequenceClassification.from_pretrained(str(load_dir))
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def compute_class_weights(labels: List[int]) -> List[float]:
        """Compute inverse-frequency class weights to handle imbalance."""
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        weights = []
        for cls_id in sorted(LABEL2ID.values()):
            cnt = counts.get(cls_id, 1)
            weights.append(total / (NUM_LABELS * cnt))
        return weights

