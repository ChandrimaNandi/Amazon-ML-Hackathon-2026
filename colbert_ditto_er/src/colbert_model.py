"""
Late-Interaction ColBERT-Ditto Model Architecture for Entity Resolution.
Combines contextualized transformer token embeddings with the MaxSim late-interaction operator.
Optimized for NVIDIA Tesla T4 GPUs (FP16 Tensor Cores, DataParallel) and CPU fallback.
References:
- "Deep Entity Matching with Pre-Trained Language Models" (Li et al., EMNLP 2020)
- "ColBERT: Contextualized Late Interaction over BERT" (Khattab & Zaharia, SIGIR 2020)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("ColBERT_Ditto.Model")


class ColBERTTokenEncoder(nn.Module if HAS_TORCH else object):
    """
    ColBERT Token Encoder:
    Encodes serialized Ditto records into a sequence of L2-normalized d-dimensional token vectors.
    """
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 64,
        max_seq_len: int = 72,
        use_fallback_if_offline: bool = True
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers are required for ColBERTTokenEncoder.")
        super().__init__()
        self.base_model_name = base_model_name
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Load transformer backbone with offline fallback protection
        try:
            self.transformer = AutoModel.from_pretrained(base_model_name)
            hidden_dim = self.transformer.config.hidden_size
            logger.info(f"[MODEL] Loaded pre-trained transformer: {base_model_name} (hidden_dim={hidden_dim})")
        except Exception as e:
            if use_fallback_if_offline:
                logger.warning(f"[MODEL] Could not download {base_model_name} ({e}). Creating model from config...")
                config = AutoConfig.from_pretrained(base_model_name)
                self.transformer = AutoModel.from_config(config)
                hidden_dim = config.hidden_size
            else:
                raise e
                
        # ColBERT Linear Projection Layer: maps hidden_dim (e.g. 384) to embed_dim (e.g. 64)
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
        # Learned calibration parameters for probability output: Score = sigmoid((MaxSim - bias) / tau)
        self.tau = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def encode_tokens(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Passes input through transformer, projects to embed_dim, and L2-normalizes each token vector.
        Returns: (batch_size, seq_len, embed_dim)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Token sequence hidden states: (batch_size, seq_len, hidden_dim)
        sequence_output = outputs[0]
        # Project to lower dimension (e.g., 64-d)
        projected = self.proj(sequence_output)
        # L2-normalize token embeddings along embedding dimension
        normalized = F.normalize(projected, p=2, dim=-1)
        # Zero out masked padding tokens
        mask = attention_mask.unsqueeze(-1).to(normalized.dtype)
        return normalized * mask

    def forward(
        self,
        q_ids: "torch.Tensor",
        q_mask: "torch.Tensor",
        d_ids: "torch.Tensor",
        d_mask: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Computes the pairwise MaxSim late-interaction score between query Q and document/reference D.
        Returns: (raw_maxsim, calibrated_probability)
        """
        # Encode tokens
        q_emb = self.encode_tokens(q_ids, q_mask)  # (B, L_q, d)
        d_emb = self.encode_tokens(d_ids, d_mask)  # (B, L_d, d)
        
        maxsim = compute_maxsim_score(q_emb, q_mask, d_emb, d_mask)
        prob = torch.sigmoid((maxsim - self.bias) / torch.clamp(self.tau, min=0.01))
        return maxsim, prob


def compute_maxsim_score(
    q_emb: "torch.Tensor",
    q_mask: "torch.Tensor",
    d_emb: "torch.Tensor",
    d_mask: "torch.Tensor"
) -> "torch.Tensor":
    """
    Batched MaxSim Late-Interaction Operator.
    
    Given:
      q_emb: (B, L_q, d)
      q_mask: (B, L_q)
      d_emb: (B, L_d, d)
      d_mask: (B, L_d)
      
    Computes:
      S = Q * D^T  (B, L_q, L_d)
      max_over_D = max_j S[:, :, j]
      MaxSim = mean(max_over_D over valid query tokens)
    """
    # 1. Pairwise token cosine similarity matrix: (B, L_q, L_d)
    similarity = torch.bmm(q_emb, d_emb.transpose(1, 2))
    
    # 2. Mask out candidate padding tokens so they are never selected as max
    mask_d = (1.0 - d_mask.unsqueeze(1).to(similarity.dtype)) * -1e4
    similarity_masked = similarity + mask_d
    
    # 3. MaxSim: Find best matching reference token for each query token
    max_scores, _ = torch.max(similarity_masked, dim=2)  # (B, L_q)
    
    # 4. Length-normalized sum over unmasked query tokens
    q_mask_dtype = q_mask.to(max_scores.dtype)
    valid_query_tokens = torch.clamp(q_mask_dtype.sum(dim=1), min=1.0)
    score = torch.sum(max_scores * q_mask_dtype, dim=1) / valid_query_tokens
    
    return score


def compute_cross_maxsim_batch(
    q_emb: "torch.Tensor",
    q_mask: "torch.Tensor",
    d_emb_list: List["torch.Tensor"],
    d_mask_list: List["torch.Tensor"]
) -> "torch.Tensor":
    """
    Efficient multi-candidate evaluation for a single query against K candidate references.
    q_emb: (1, L_q, d)
    d_emb_list: K tensors of (1, L_d, d)
    Returns: (K,) tensor of MaxSim scores
    """
    K = len(d_emb_list)
    if K == 0:
        return torch.empty(0, device=q_emb.device)
    
    # Stack candidate tensors: (K, L_d, d)
    stacked_d_emb = torch.cat(d_emb_list, dim=0)
    stacked_d_mask = torch.cat(d_mask_list, dim=0)
    
    # Repeat query K times: (K, L_q, d)
    expanded_q_emb = q_emb.expand(K, -1, -1)
    expanded_q_mask = q_mask.expand(K, -1)
    
    return compute_maxsim_score(expanded_q_emb, expanded_q_mask, stacked_d_emb, stacked_d_mask)
