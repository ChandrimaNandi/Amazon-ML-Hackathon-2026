"""
End-to-End Training & Validation Script for Precision-Sieve ER (Target: 99+ Macro F0.5).
Trains Asymmetric 4x False-Positive GBDT Ensemble and optimizes high-precision thresholds.
"""

import sys
import os
import time
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import pandas as pd
import numpy as np

# Ensure root workspace is in sys.path
sys_path_root = str(Path(__file__).resolve().parents[2])
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from precision_sieve_er.configs.default_config import (
    discover_dataset_paths, RESULTS_DIR, print_gpu_info,
    FP_PENALTY_WEIGHT, DEFAULT_ABS_THRESHOLD, DEFAULT_MARGIN_THRESHOLD
)
from precision_sieve_er.src.normalization import create_normalized_dataframe
from precision_sieve_er.src.blocking import PrecisionSieveBlocker
from precision_sieve_er.src.features import extract_candidate_features_dataframe
from precision_sieve_er.src.classifier import AsymmetricPrecisionEnsemble
from precision_sieve_er.src.evaluation import evaluate_macro_f05, optimize_thresholds_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PrecisionSieve.Train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Precision-Sieve ER GBDT Ensemble")
    parser.add_argument("--train_dir", type=str, default=None, help="Path to train directory containing source1-3 and ground_truth TSVs")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR), help="Output directory for model weights and configs")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio for S1 entity-level split")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of trees per GBDT model")
    parser.add_argument("--fp_weight", type=float, default=FP_PENALTY_WEIGHT, help="Asymmetric false-positive loss penalty weight")
    parser.add_argument("--max_train_queries", type=int, default=None, help="Optional subset limit for fast debugging")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print("=" * 70)
    print("      PRECISION-SIEVE ER: 99+ MACRO F0.5 TRAINING PIPELINE")
    print("=" * 70)
    print_gpu_info()

    # Discover paths
    if args.train_dir:
        train_dir = Path(args.train_dir).resolve()
    else:
        train_dir, _ = discover_dataset_paths()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using Training Directory: {train_dir}")
    logger.info(f"Model Artifacts Directory: {output_dir}")

    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    assert s1_path.exists(), f"Missing {s1_path}"
    assert gt_path.exists(), f"Missing {gt_path}"

    # 1. Load Data
    logger.info("Loading TSV datasets...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str) if s2_path.exists() else pd.DataFrame()
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str) if s3_path.exists() else pd.DataFrame()
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)

    queries_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"Loaded: S1={len(s1_df):,} | S2={len(s2_df):,} | S3={len(s3_df):,} | Queries={len(queries_df):,} | GT={len(gt_df):,}")

    # Build ground truth dictionary {s1_id: set(qids)}
    gt_dict: Dict[str, Set[str]] = {s1: set() for s1 in s1_df["entity_id"]}
    q_to_true_s1: Dict[str, str] = {}
    for row in gt_df.itertuples():
        s1 = getattr(row, "source1_entity_id", "")
        matches_str = getattr(row, "matched_entity_ids", "")
        if s1 in gt_dict and pd.notna(matches_str) and str(matches_str).strip():
            for m in str(matches_str).split(","):
                m_clean = m.strip()
                if m_clean:
                    gt_dict[s1].add(m_clean)
                    q_to_true_s1[m_clean] = s1

    # 2. Normalize Text
    logger.info("Applying NFKC normalization and phonetic transliteration...")
    s1_df = create_normalized_dataframe(s1_df)
    queries_df = create_normalized_dataframe(queries_df)

    s1_lookup = {
        row.entity_id: {
            "name_norm": getattr(row, "name_norm", ""),
            "addr_norm": getattr(row, "addr_norm", ""),
            "country": getattr(row, "country_clean", "")
        }
        for row in s1_df.itertuples()
    }
    query_lookup = {
        row.entity_id: {
            "name_norm": getattr(row, "name_norm", ""),
            "addr_norm": getattr(row, "addr_norm", ""),
            "country": getattr(row, "country_clean", "")
        }
        for row in queries_df.itertuples()
    }

    # 3. Entity-Level Train / Validation Split
    all_s1_ids = s1_df["entity_id"].tolist()
    np.random.shuffle(all_s1_ids)
    num_val = int(len(all_s1_ids) * args.val_ratio)
    val_s1_set = set(all_s1_ids[:num_val])
    train_s1_set = set(all_s1_ids[num_val:])

    logger.info(f"Split: {len(train_s1_set):,} Train S1 entities | {len(val_s1_set):,} Validation S1 entities")

    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].copy()
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].copy()

    # Identify train queries vs val queries
    train_queries_list = [q for q, s1 in q_to_true_s1.items() if s1 in train_s1_set]
    val_queries_list = [q for q, s1 in q_to_true_s1.items() if s1 in val_s1_set]

    # Add unlinked/singleton queries if any
    all_matched_qids = set(q_to_true_s1.keys())
    unmatched_qids = [qid for qid in queries_df["entity_id"] if qid not in all_matched_qids]
    val_unmatched = unmatched_qids[:len(unmatched_qids) // 2]
    train_unmatched = unmatched_qids[len(unmatched_qids) // 2:]

    logger.info(f"Train queries: {len(train_queries_list):,} linked + {len(train_unmatched):,} unlinked")
    logger.info(f"Val queries:   {len(val_queries_list):,} linked + {len(val_unmatched):,} unlinked")

    # 4. Fit Precision Sieve Blocker on Train S1
    logger.info("Fitting PrecisionSieveBlocker on Train S1...")
    train_blocker = PrecisionSieveBlocker(top_k_bm25=8)
    train_blocker.fit(train_s1_df)

    # 5. Mine Training Pairs (Positives + Hard BM25 Negatives)
    logger.info("Mining targeted training candidate pairs...")
    train_q_set = set(train_queries_list + train_unmatched)
    sub_train_q_df = queries_df[queries_df["entity_id"].isin(train_q_set)]
    if args.max_train_queries:
        sub_train_q_df = sub_train_q_df.iloc[:args.max_train_queries]

    train_cands_map = train_blocker.generate_candidates(sub_train_q_df)

    training_pairs: List[Dict[str, Any]] = []
    labels: List[int] = []

    for row in sub_train_q_df.itertuples():
        qid = row.entity_id
        true_s1 = q_to_true_s1.get(qid, None)
        cands = train_cands_map.get(qid, [])
        cand_s1s = {c[0] for c in cands}

        # Ensure ground-truth positive is included if in train_s1_set
        if true_s1 and true_s1 in train_s1_set:
            training_pairs.append({
                "query_id": qid,
                "s1_id": true_s1,
                "bm25_score": 15.0 if true_s1 not in cand_s1s else [c[1] for c in cands if c[0] == true_s1][0],
                "rank": 1
            })
            labels.append(1)

        # Hard negatives from BM25 retrieval
        for rank, (cand_s1, score) in enumerate(cands, start=1):
            if cand_s1 != true_s1:
                training_pairs.append({
                    "query_id": qid,
                    "s1_id": cand_s1,
                    "bm25_score": score,
                    "rank": rank
                })
                labels.append(0)

    logger.info(f"Extracted {len(training_pairs):,} training pairs ({sum(labels):,} positive, {len(labels) - sum(labels):,} hard negatives)")

    # 6. Extract Features & Train Asymmetric Ensemble
    logger.info("Extracting candidate pairwise features...")
    X_train_df = extract_candidate_features_dataframe(training_pairs, s1_lookup, query_lookup)
    y_train = np.array(labels, dtype=int)

    logger.info(f"Initializing Asymmetric GBDT Ensemble (FP Penalty Weight = {args.fp_weight}x)...")
    ensemble = AsymmetricPrecisionEnsemble(
        fp_weight=args.fp_weight,
        n_estimators=args.n_estimators,
        learning_rate=0.04
    )
    ensemble.fit(X_train_df, y_train)

    # 7. Evaluate on Unseen Validation S1 Entities
    logger.info("=" * 60)
    logger.info("[VALIDATION] Evaluating on unseen S1 validation partition...")
    val_blocker = PrecisionSieveBlocker(top_k_bm25=8)
    val_blocker.fit(val_s1_df)

    val_q_set = set(val_queries_list + val_unmatched)
    sub_val_q_df = queries_df[queries_df["entity_id"].isin(val_q_set)]
    val_cands_map = val_blocker.generate_candidates(sub_val_q_df)

    val_pairs: List[Dict[str, Any]] = []
    for row in sub_val_q_df.itertuples():
        qid = row.entity_id
        for rank, (cand_s1, score) in enumerate(val_cands_map.get(qid, []), start=1):
            val_pairs.append({
                "query_id": qid,
                "s1_id": cand_s1,
                "bm25_score": score,
                "rank": rank
            })

    logger.info(f"Scoring {len(val_pairs):,} validation candidate pairs...")
    X_val_df = extract_candidate_features_dataframe(val_pairs, s1_lookup, query_lookup)
    val_probs = ensemble.predict_proba(
        X_val_df,
        apply_vetoes=True,
        s1_lookup=s1_lookup,
        query_lookup=query_lookup
    )

    scored_val_candidates = [
        {"query_id": p["query_id"], "s1_id": p["s1_id"], "score": float(prob)}
        for p, prob in zip(val_pairs, val_probs)
    ]

    val_gt_dict = {s1: gt_dict.get(s1, set()) for s1 in val_s1_set}

    # 8. 2D High-Precision Grid Search
    best_abs, best_margin, best_f05, best_metrics = optimize_thresholds_grid(
        scored_candidates=scored_val_candidates,
        ground_truth=val_gt_dict,
        all_s1_ids=list(val_s1_set),
        abs_range=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        margin_range=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    )

    print("\n" + "=" * 60)
    print("            VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Macro F0.5:      {best_metrics.get('macro_f0.5', 0.0):.4f}")
    print(f"  Macro Precision: {best_metrics.get('macro_precision', 0.0):.4f}")
    print(f"  Macro Recall:    {best_metrics.get('macro_recall', 0.0):.4f}")
    print(f"  Optimal Abs Th:  {best_abs:.2f}")
    print(f"  Optimal Margin:  {best_margin:.2f}")
    print("=" * 60 + "\n")

    # 9. Save Artifacts
    model_path = output_dir / "precision_ensemble.pkl"
    ensemble.save(str(model_path))
    logger.info(f"Saved ensemble weights to: {model_path}")

    config_path = output_dir / "best_thresholds.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "abs_threshold": best_abs,
            "margin_threshold": best_margin,
            "val_macro_f0.5": best_f05,
            "val_precision": best_metrics.get("macro_precision", 0.0),
            "val_recall": best_metrics.get("macro_recall", 0.0),
            "fp_weight": args.fp_weight,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    logger.info(f"Saved threshold configs to: {config_path}")


if __name__ == "__main__":
    main()
