"""
Entity-Disjoint Training & Threshold Optimization Pipeline for Fast Hybrid ER.
Evaluates Macro F0.5 on held-out validation entities and persists optimal model artifacts.
"""

import sys
import os
import json
import time
import random
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fast_hybrid_er.configs.default_config import (
    TRAIN_DIR, TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH,
    RESULTS_DIR, RANDOM_SEED, print_hardware_summary
)
from fast_hybrid_er.src.data_loader import load_coherent_training_sample
from fast_hybrid_er.src.normalization import create_normalized_features
from fast_hybrid_er.src.lsh_blocking import MultiChannelBlocker
from fast_hybrid_er.src.features import HybridFeatureExtractor
from fast_hybrid_er.src.classifier import AsymmetricEntityRanker
from fast_hybrid_er.src.assignment import optimize_thresholds_grid, apply_query_exclusivity, evaluate_macro_f05


def run_training_pipeline(sample_s1_count: int = 15000, random_seed: int = RANDOM_SEED):
    """Executes end-to-end training and threshold optimization."""
    print_hardware_summary()
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("\n[1/6] Loading Coherent Training Dataset & Ground Truth...")
    s1_df, active_queries_df, gt_map = load_coherent_training_sample(
        s1_path=TRAIN_S1_PATH,
        gt_path=TRAIN_GROUND_TRUTH_PATH,
        s2_path=TRAIN_S2_PATH,
        s3_path=TRAIN_S3_PATH,
        sample_s1_rows=sample_s1_count,
        max_active_queries=sample_s1_count,
        num_unmatched_queries=max(50, sample_s1_count // 10),
        random_seed=random_seed
    )

    query_to_s1 = {}
    for s1, q_set in gt_map.items():
        for q in q_set:
            query_to_s1[q] = s1

    print("\n[2/6] Normalizing Text Fields...")
    s1_df = create_normalized_features(s1_df)
    active_queries_df = create_normalized_features(active_queries_df)

    # Disjoint Entity Split
    print("\n[3/6] Splitting Reference Entities (80% Train, 20% Val Disjoint Split)...")
    all_s1_ids = s1_df["entity_id"].tolist()
    random.shuffle(all_s1_ids)
    split_idx = int(0.8 * len(all_s1_ids))
    train_s1_ids = set(all_s1_ids[:split_idx])
    val_s1_ids = all_s1_ids[split_idx:]

    s1_train_df = s1_df[s1_df["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    s1_val_df = s1_df[s1_df["entity_id"].isin(set(val_s1_ids))].reset_index(drop=True)

    # Fast Lookups
    s1_lookup = {r.entity_id: r._asdict() for r in s1_df.itertuples()}
    q_lookup = {r.entity_id: r._asdict() for r in active_queries_df.itertuples()}

    print("\n[4/6] Fitting Multi-Channel Blocking on Training S1 & Generating Training Pairs...")
    blocker = MultiChannelBlocker()
    blocker.fit(s1_train_df)

    # Retrieve candidates on train queries
    train_queries = active_queries_df[active_queries_df["entity_id"].map(lambda q: query_to_s1.get(q, "")) .isin(train_s1_ids)]
    train_candidates = blocker.retrieve_candidates(train_queries, top_k=10)

    # Ensure ground truth positive pairs exist in training set
    cand_pairs_set = {(c["query_id"], c["s1_id"]) for c in train_candidates}
    for qid in train_queries["entity_id"]:
        true_s1 = query_to_s1.get(qid)
        if true_s1 and true_s1 in train_s1_ids and (qid, true_s1) not in cand_pairs_set:
            train_candidates.append({
                "query_id": qid,
                "s1_id": true_s1,
                "by_exact_name": 0, "by_exact_address": 0, "by_exact_combined": 0,
                "by_phonetic": 0, "by_lsh": 0, "by_token": 0, "channel_agreement": 1
            })

    # Extract features for train candidates
    feature_extractor = HybridFeatureExtractor()
    X_train_df = feature_extractor.extract_features(train_candidates, s1_lookup, q_lookup)
    y_train = np.array([1 if query_to_s1.get(c["query_id"]) == c["s1_id"] else 0 for c in train_candidates], dtype=np.int32)
    print(f"Training dataset: {len(X_train_df):,} pairs ({np.sum(y_train):,} positives, {len(y_train) - np.sum(y_train):,} negatives).")

    print("\n[5/6] Training Asymmetric Precision-Weighted LightGBM Matcher...")
    ranker = AsymmetricEntityRanker()
    ranker.train(X_train_df, y_train)
    model_path = ranker.save()

    print("\n[6/6] Validation Benchmarking & 2D Threshold Optimization...")
    # Evaluate on validation split
    val_blocker = MultiChannelBlocker()
    val_blocker.fit(s1_val_df)
    val_queries = active_queries_df[active_queries_df["entity_id"].map(lambda q: query_to_s1.get(q, "")).isin(set(val_s1_ids))]
    val_candidates = val_blocker.retrieve_candidates(val_queries, top_k=10)

    X_val_df = feature_extractor.extract_features(val_candidates, s1_lookup, q_lookup)
    val_scores = ranker.predict_proba(X_val_df)
    X_val_df["score"] = val_scores

    best_abs, best_margin, best_f05, metrics = optimize_thresholds_grid(
        X_val_df[["query_id", "s1_id", "score"]],
        val_s1_ids,
        gt_map
    )

    print("\n" + "=" * 60)
    print(f"Validation Benchmark Results (Target Metric: Macro F0.5):")
    print(f"  Optimal Absolute Threshold: {best_abs:.2f}")
    print(f"  Optimal Margin Threshold:   {best_margin:.2f}")
    print(f"  Validation Macro F0.5:      {metrics.get('macro_f0.5', 0.0):.4f}")
    print(f"  Validation Macro Precision: {metrics.get('macro_precision', 0.0):.4f}")
    print(f"  Validation Macro Recall:    {metrics.get('macro_recall', 0.0):.4f}")
    print("=" * 60)

    # Persist threshold configuration
    thresh_path = RESULTS_DIR / "threshold_config.json"
    payload = {
        "abs_threshold": float(best_abs),
        "margin_threshold": float(best_margin),
        "macro_f0.5": float(best_f05),
        "metrics": metrics
    }
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold configuration to {thresh_path}")

    return {
        "model_path": str(model_path),
        "threshold_path": str(thresh_path),
        "metrics": metrics
    }


if __name__ == "__main__":
    run_training_pipeline()
