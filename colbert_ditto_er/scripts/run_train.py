"""
ColBERT-Ditto Entity Resolution: Model Training & Threshold Optimization Script.
Runs on Kaggle 2× NVIDIA Tesla T4 GPUs (or CPU fallback) with strict entity-disjoint splitting.
"""

import os
import sys
import time
import json
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from colbert_ditto_er.configs.default_config import (
    TRAIN_DIR, RESULTS_DIR, print_gpu_info,
    DEFAULT_BASE_MODEL, DEFAULT_EMBED_DIM, DEFAULT_MAX_SEQ_LEN
)
from colbert_ditto_er.src.normalization import create_normalized_dataframe
from colbert_ditto_er.src.serialization import serialize_dataframe
from colbert_ditto_er.src.blocking import SparseBlockingEngine
from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder
from colbert_ditto_er.src.training import train_colbert_model
from colbert_ditto_er.src.evaluation import optimize_thresholds_grid, evaluate_macro_f05
from colbert_ditto_er.src.assignment import apply_assignment_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ColBERT_Ditto.RunTrain")


def main():
    parser = argparse.ArgumentParser(description="ColBERT-Ditto Model Training")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--num-train-entities", type=int, default=50000, help="Number of S1 entities for training")
    parser.add_argument("--num-val-entities", type=int, default=10000, help="Number of S1 entities for validation")
    parser.add_argument("--model-name", type=str, default=DEFAULT_BASE_MODEL)
    args = parser.parse_args()

    print_gpu_info()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Ground Truth
    gt_path = args.train_dir / "train_ground_truth.tsv"
    logger.info(f"Loading ground truth from {gt_path}...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    # Parse ground truth mapping: {s1_id: set(matched_query_ids)}
    gt_map = {}
    query_to_s1 = {}
    for row in gt_df.itertuples():
        s1 = row.source1_entity_id
        matched = str(row.matched_entity_ids).split(",") if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip() else []
        matched_set = set(m.strip() for m in matched if m.strip())
        gt_map[s1] = matched_set
        for q in matched_set:
            query_to_s1[q] = s1

    # 2. Strict Entity-Level Disjoint Split
    logger.info("[SPLIT] Performing strict entity-disjoint train/val split on S1 entities...")
    all_s1_list = list(gt_map.keys())
    random.seed(42)
    random.shuffle(all_s1_list)

    val_s1_set = set(all_s1_list[:args.num_val_entities])
    train_s1_set = set(all_s1_list[args.num_val_entities:args.num_val_entities + args.num_train_entities])

    assert len(train_s1_set.intersection(val_s1_set)) == 0, "Data leakage detected between train and val!"
    logger.info(f"Train S1 Entities: {len(train_s1_set):,} | Val S1 Entities: {len(val_s1_set):,}")

    # 3. Load S1 reference and S2/S3 queries
    logger.info("Loading reference and query data...")
    s1_df = pd.read_csv(args.train_dir / "train_source1.tsv", sep="\t", dtype=str)
    s2_df = pd.read_csv(args.train_dir / "train_source2.tsv", sep="\t", nrows=200000, dtype=str)
    s3_df = pd.read_csv(args.train_dir / "train_source3.tsv", sep="\t", nrows=200000, dtype=str)
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Preprocess text
    s1_df = create_normalized_dataframe(s1_df)
    s1_df["serialized"] = serialize_dataframe(s1_df)
    s1_text_map = dict(zip(s1_df["entity_id"], s1_df["serialized"]))

    query_df = create_normalized_dataframe(query_df)
    query_df["serialized"] = serialize_dataframe(query_df)
    query_text_map = dict(zip(query_df["entity_id"], query_df["serialized"]))

    # 4. Mine Hard Negatives with Blocker for Training
    logger.info("[MINING] Mining hard retrieval negatives from BM25 blocker...")
    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].copy()
    blocker_train = SparseBlockingEngine(top_k_candidates=10)
    blocker_train.fit(train_s1_df)

    # Collect training queries linked to train_s1
    train_queries = [q for q, s1 in query_to_s1.items() if s1 in train_s1_set and q in query_text_map][:30000]
    train_query_df = query_df[query_df["entity_id"].isin(set(train_queries))].copy()
    cand_train = blocker_train.generate_candidates_for_chunk(train_query_df)

    triplets = []
    for qid in train_queries:
        pos_s1 = query_to_s1[qid]
        cands = cand_train.get(qid, [])
        # Hard negatives: candidates retrieved that are NOT the true positive
        neg_cands = [c for c in cands if c != pos_s1 and c in s1_text_map]
        if neg_cands and pos_s1 in s1_text_map:
            # Pick a hard negative
            neg_s1 = random.choice(neg_cands)
            triplets.append((query_text_map[qid], s1_text_map[pos_s1], s1_text_map[neg_s1]))

    logger.info(f"Generated {len(triplets):,} high-quality training triplets.")

    # 5. Initialize and Train ColBERT Model
    model = ColBERTTokenEncoder(
        base_model_name=args.model_name,
        embed_dim=DEFAULT_EMBED_DIM,
        max_seq_len=DEFAULT_MAX_SEQ_LEN
    )

    trained_model = train_colbert_model(
        model=model,
        triplets=triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        device=device,
        use_fp16=torch.cuda.is_available()
    )

    # 6. Validation Evaluation & Threshold Optimization
    logger.info("[VALIDATION] Evaluating ColBERT on held-out unseen validation entities...")
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].copy()
    blocker_val = SparseBlockingEngine(top_k_candidates=10)
    blocker_val.fit(val_s1_df)

    val_queries = [q for q, s1 in query_to_s1.items() if s1 in val_s1_set and q in query_text_map][:10000]
    val_query_df = query_df[query_df["entity_id"].isin(set(val_queries))].copy()
    cand_val = blocker_val.generate_candidates_for_chunk(val_query_df)

    # Score validation pairs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    val_pairs_to_score = []
    for qid in val_queries:
        for s1_cand in cand_val.get(qid, []):
            if s1_cand in s1_text_map:
                val_pairs_to_score.append((qid, s1_cand, query_text_map[qid], s1_text_map[s1_cand]))

    logger.info(f"Scoring {len(val_pairs_to_score):,} validation candidate pairs...")
    scored_val = []
    trained_model.eval()
    batch_size = 256

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for b_i in range(0, len(val_pairs_to_score), batch_size):
                b_slice = val_pairs_to_score[b_i:b_i + batch_size]
                b_q = [x[2] for x in b_slice]
                b_r = [x[3] for x in b_slice]

                tok_q = tokenizer(b_q, padding=True, truncation=True, max_length=DEFAULT_MAX_SEQ_LEN, return_tensors="pt")
                tok_r = tokenizer(b_r, padding=True, truncation=True, max_length=DEFAULT_MAX_SEQ_LEN, return_tensors="pt")

                q_emb = trained_model.encode_tokens(tok_q["input_ids"].to(device), tok_q["attention_mask"].to(device))
                d_emb = trained_model.encode_tokens(tok_r["input_ids"].to(device), tok_r["attention_mask"].to(device))

                from colbert_ditto_er.src.colbert_model import compute_maxsim_score
                maxsim = compute_maxsim_score(q_emb, tok_q["attention_mask"].to(device), d_emb, tok_r["attention_mask"].to(device))
                probs = torch.sigmoid((maxsim - trained_model.bias) / torch.clamp(trained_model.tau, min=0.01))

                probs_np = probs.cpu().numpy().tolist()
                for (qid, s1_cand, _, _), sc in zip(b_slice, probs_np):
                    scored_val.append({"query_id": qid, "s1_id": s1_cand, "score": sc})

    # Ground truth mapping restricted to validation S1 entities
    val_gt_map = {s1: gt_map[s1] for s1 in val_s1_set}

    best_abs, best_margin, best_f05, metrics = optimize_thresholds_grid(
        scored_candidates=scored_val,
        ground_truth=val_gt_map,
        all_s1_ids=list(val_s1_set)
    )

    logger.info("=" * 60)
    logger.info("FINAL VALIDATION BENCHMARK RESULTS (ColBERT Late-Interaction)")
    logger.info(f"  Macro F0.5:        {best_f05:.4f}")
    logger.info(f"  Macro Precision:   {metrics.get('macro_precision', 0):.4f}")
    logger.info(f"  Macro Recall:      {metrics.get('macro_recall', 0):.4f}")
    logger.info(f"  Optimal Threshold: {best_abs:.2f}")
    logger.info(f"  Optimal Margin:    {best_margin:.2f}")
    logger.info("=" * 60)

    # 7. Save model checkpoint and threshold configuration
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = RESULTS_DIR / "colbert_model.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"Saved model weights to {model_save_path}")

    threshold_config = {
        "abs_threshold": best_abs,
        "margin_threshold": best_margin,
        "macro_f0.5": best_f05,
        "metrics": metrics,
        "model_name": args.model_name
    }
    config_save_path = RESULTS_DIR / "threshold_config.json"
    with open(config_save_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved threshold configuration to {config_save_path}")


if __name__ == "__main__":
    main()
