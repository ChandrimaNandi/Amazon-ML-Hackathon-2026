"""
PyTorch Dataset and DataLoader Utilities for ColBERT-Ditto Entity Resolution.
Supports triplet training (Anchor, Positive, Hard Negative) and pair evaluation with dynamic padding.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ColBERTTripletDataset(Dataset if HAS_TORCH else object):
    """
    Dataset yielding triplets: (query_text, pos_s1_text, neg_s1_text).
    Trained via pairwise margin ranking loss.
    """
    def __init__(
        self,
        triplets: List[Tuple[str, str, str]]
    ):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.triplets[idx]


class ColBERTPairDataset(Dataset if HAS_TORCH else object):
    """
    Dataset yielding candidate pairs: (query_id, s1_id, query_text, s1_text, label).
    Used for validation scoring and threshold optimization.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str, str, str, int]]
    ):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str, int]:
        return self.pairs[idx]


class ColBERTBatchCollator:
    """
    Dynamic tokenization and padding collator for PyTorch DataLoader.
    Pads sequences dynamically to the batch's max sequence length to minimize memory.
    """
    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_len: int = 72
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers required.")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len

    def collate_triplets(self, batch: List[Tuple[str, str, str]]) -> Dict[str, "torch.Tensor"]:
        queries = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        negatives = [item[2] for item in batch]
        
        q_tok = self.tokenizer(
            queries, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
        )
        pos_tok = self.tokenizer(
            positives, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
        )
        neg_tok = self.tokenizer(
            negatives, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
        )
        
        return {
            "q_ids": q_tok["input_ids"],
            "q_mask": q_tok["attention_mask"],
            "pos_ids": pos_tok["input_ids"],
            "pos_mask": pos_tok["attention_mask"],
            "neg_ids": neg_tok["input_ids"],
            "neg_mask": neg_tok["attention_mask"],
        }

    def collate_pairs(self, batch: List[Tuple[str, str, str, str, int]]) -> Dict[str, Any]:
        q_ids = [item[0] for item in batch]
        s1_ids = [item[1] for item in batch]
        queries = [item[2] for item in batch]
        references = [item[3] for item in batch]
        labels = [item[4] for item in batch]
        
        q_tok = self.tokenizer(
            queries, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
        )
        d_tok = self.tokenizer(
            references, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
        )
        
        return {
            "query_ids": q_ids,
            "s1_ids": s1_ids,
            "q_ids": q_tok["input_ids"],
            "q_mask": q_tok["attention_mask"],
            "d_ids": d_tok["input_ids"],
            "d_mask": d_tok["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        }
