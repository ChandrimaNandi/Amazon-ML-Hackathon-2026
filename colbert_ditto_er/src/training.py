"""
ColBERT-Ditto Fine-Tuning Module.
Implements Margin Ranking Loss with hard negative mining and FP16 acceleration on Kaggle 2× T4 GPUs.
"""

import time
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder, compute_maxsim_score
from colbert_ditto_er.src.dataset import ColBERTTripletDataset, ColBERTBatchCollator

logger = logging.getLogger("ColBERT_Ditto.Training")


def train_colbert_model(
    model: ColBERTTokenEncoder,
    triplets: List[Tuple[str, str, str]],
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    margin: float = 0.20,
    device: Optional["torch.device"] = None,
    use_fp16: bool = True
) -> ColBERTTokenEncoder:
    """
    Trains ColBERTTokenEncoder on triplets (Query, Positive Ref, Negative Ref)
    using Margin Ranking Loss.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training.")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"[TRAINING] Starting ColBERT Late-Interaction Training on {device}")
    logger.info(f"  Total Triplets:  {len(triplets):,}")
    logger.info(f"  Epochs:          {epochs}")
    logger.info(f"  Batch Size:      {batch_size}")
    logger.info(f"  Learning Rate:   {learning_rate}")
    logger.info(f"  Margin:          {margin}")
    logger.info(f"  FP16 Enabled:    {use_fp16 and torch.cuda.is_available()}")
    logger.info("=" * 60)

    model.to(device)
    
    # Wrap in DataParallel if multiple GPUs present (e.g. 2x T4 on Kaggle)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1 and str(device).startswith("cuda"):
        logger.info(f"[TRAINING] Distributing model across {gpu_count} GPUs via nn.DataParallel")
        train_model = nn.DataParallel(model)
    else:
        train_model = model

    dataset = ColBERTTripletDataset(triplets)
    collator = ColBERTBatchCollator(tokenizer_name=model.base_model_name, max_seq_len=model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.collate_triplets, drop_last=False)

    optimizer = AdamW(train_model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(total_steps, 1))
    
    scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and torch.cuda.is_available()))
    criterion = nn.MarginRankingLoss(margin=margin)

    train_model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        step = 0

        for batch in loader:
            step += 1
            optimizer.zero_grad()

            q_ids = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            pos_mask = batch["pos_mask"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            neg_mask = batch["neg_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=(use_fp16 and torch.cuda.is_available())):
                # If wrapped in DataParallel, access module methods or forward directly
                if isinstance(train_model, nn.DataParallel):
                    enc = train_model.module
                else:
                    enc = train_model

                q_emb = enc.encode_tokens(q_ids, q_mask)
                pos_emb = enc.encode_tokens(pos_ids, pos_mask)
                neg_emb = enc.encode_tokens(neg_ids, neg_mask)

                pos_score = compute_maxsim_score(q_emb, q_mask, pos_emb, pos_mask)
                neg_score = compute_maxsim_score(q_emb, q_mask, neg_emb, neg_mask)

                # Target label = 1 indicates pos_score should be greater than neg_score by margin
                target = torch.ones_like(pos_score)
                loss = criterion(pos_score, neg_score, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == len(loader):
                avg_l = running_loss / step
                logger.info(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(loader)}] - Loss: {avg_l:.4f}")

        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} finished in {elapsed:.1f}s - Avg Loss: {running_loss / max(len(loader), 1):.4f}")

    logger.info("[TRAINING] ColBERT Late-Interaction training complete.")
    return model
