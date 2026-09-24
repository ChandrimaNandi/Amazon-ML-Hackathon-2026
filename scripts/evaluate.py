"""
Script to evaluate matching pipeline and calculate Macro F0.5.
Run from project root: python3 scripts/evaluate.py
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH, RESULTS_DIR
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.normalization import create_normalized_features
from src.candidate_generation import generate_candidate_union
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel
from src.thresholding import apply_decision_rules
from src.singleton import analyze_singleton_performance
from src.evaluation import evaluate_macro_metrics


def main():
    print("=" * 60)
    print("[EVALUATE] Running Validation Framework on Train Dataset")
    print("=" * 60)
    
    s1_df = load_source_tsv(TRAIN_S1_PATH)
    s2_df = load_source_tsv(TRAIN_S2_PATH)
    s3_df = load_source_tsv(TRAIN_S3_PATH)
    gt_df, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    # Use a representative sample for fast validation evaluation
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    if len(query_df) > 30000:
        query_df = query_df.sample(n=30000, random_state=42).reset_index(drop=True)
        print(f"Evaluated on sample of {len(query_df):,} query records.")
        
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    cand_df, stats = generate_candidate_union(s1_df, query_df, k_name=20, k_address=15)
    feat_df = extract_candidate_features(cand_df, s1_df, query_df, s1_to_matches)
    
    model = EntityMatcherModel()
    model.fit(feat_df)
    
    probs = model.predict_proba(feat_df)
    feat_df["pred_score"] = probs
    
    s1_to_pred_matches = apply_decision_rules(feat_df, abs_threshold=0.50, margin_threshold=0.05)
    
    all_s1_ids = set(s1_df["entity_id"])
    metrics = evaluate_macro_metrics(all_s1_ids, s1_to_matches, s1_to_pred_matches)
    singleton_stats = analyze_singleton_performance(all_s1_ids, s1_to_matches, s1_to_pred_matches)
    
    print("\n" + "=" * 60)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nSingleton & Match Breakdown:")
    for k, v in singleton_stats.items():
        print(f"  {k}: {v}")
        
    # Log to results/metrics.csv and results/experiments.csv
    m_csv = RESULTS_DIR / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(m_csv, index=False)
    
    exp_csv = RESULTS_DIR / "experiments.csv"
    exp_row = {
        "experiment": "LightGBM Baseline + BM25/Char-TFIDF Union",
        "candidate_method": "BM25+CharTFIDF",
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f0_5": metrics["macro_f0.5"],
        "zero_match_accuracy_pct": singleton_stats["zero_match_accuracy_pct"],
    }
    exp_df = pd.DataFrame([exp_row])
    if exp_csv.exists():
        exp_df.to_csv(exp_csv, mode="a", header=False, index=False)
    else:
        exp_df.to_csv(exp_csv, index=False)
        
    print(f"\nSaved metrics to {m_csv} and logged experiment to {exp_csv}")
    print("\n[DONE] Evaluation Complete.")


if __name__ == "__main__":
    main()
