"""
Script to evaluate matching pipeline, run threshold optimization, ablation suite, and error analysis.
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
from src.data_loader import load_source_tsv, load_ground_truth, load_coherent_training_sample
from src.normalization import create_normalized_features
from src.candidate_generation import CandidateGenerator
from src.features import extract_candidate_features
from src.negative_sampling import build_controlled_training_pairs
from src.ranking import EntityMatcherModel
from src.thresholding import optimize_threshold_grid, apply_decision_rules
from src.evaluation import evaluate_macro_metrics, evaluate_candidate_recall_diagnostics, generate_error_analysis
from src.experiments import create_entity_level_split, run_ablation_experiments
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Pipeline")
    parser.add_argument("--sample-s1", type=int, default=25000, help="S1 reference entities sample size")
    parser.add_argument("--sample-queries", type=int, default=20000, help="Query records sample size")
    args = parser.parse_args()

    print("=" * 60)
    print(f"[EVALUATE] Running Leakage-Free Validation & Ablation Pipeline (S1: {args.sample_s1:,}, Queries: {args.sample_queries:,})")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load coherent data
    s1_df, query_df, active_gt = load_coherent_training_sample(
        s1_path=TRAIN_S1_PATH,
        gt_path=TRAIN_GROUND_TRUTH_PATH,
        s2_path=TRAIN_S2_PATH,
        s3_path=TRAIN_S3_PATH,
        sample_s1_rows=args.sample_s1,
        max_active_queries=args.sample_queries,
        num_unmatched_queries=2000,
        random_seed=42
    )
    
    # 2. Normalize
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    # 3. Entity-level split
    train_s1, val_s1, train_q, val_q, train_gt, val_gt = create_entity_level_split(
        s1_df, query_df, active_gt, val_ratio=0.20, random_seed=42
    )
    
    # 4. Candidate generation on train and val separately
    gen_train = CandidateGenerator()
    gen_train.fit(train_s1)
    train_cands, _ = gen_train.generate_candidates(train_q)
    
    gen_val = CandidateGenerator()
    gen_val.fit(val_s1)
    val_cands, _ = gen_val.generate_candidates(val_q)
    
    # Recall diagnostics
    val_recall_stats = evaluate_candidate_recall_diagnostics(val_cands, val_gt)
    
    # 5. Feature extraction
    train_feat = extract_candidate_features(train_cands, train_s1, train_q, train_gt)
    val_feat = extract_candidate_features(val_cands, val_s1, val_q, val_gt)
    
    # 6. Negative sampling on train
    balanced_train, _ = build_controlled_training_pairs(train_feat, max_negatives_per_positive=8)
    
    # 7. Model training
    model = EntityMatcherModel()
    model.fit(balanced_train, val_df=val_feat)
    
    # 8. Score validation candidates
    val_feat["pred_score"] = model.predict_proba(val_feat)
    
    # 9. Threshold optimization sweep strictly on validation set
    val_s1_ids = set(val_s1["entity_id"])
    opt_results = optimize_threshold_grid(
        val_cand_df_with_probs=val_feat,
        val_s1_ids=val_s1_ids,
        s1_to_true_matches=val_gt
    )
    
    best_thresh = opt_results["best_threshold"]
    best_margin = opt_results["best_margin"]
    best_f05 = opt_results["best_macro_f0.5"]
    
    print("\n" + "=" * 60)
    print("OPTIMAL VALIDATION THRESHOLD & METRICS")
    print("=" * 60)
    print(f"  Best Threshold: {best_thresh:.2f}")
    print(f"  Best Margin:    {best_margin:.2f}")
    print(f"  Macro F0.5:     {best_f05:.4f}")
    print(f"  Macro Precision:{opt_results['best_precision']:.4f}")
    print(f"  Macro Recall:   {opt_results['best_recall']:.4f}")
    
    # 10. Run 10-stage ablation suite
    ablation_df = run_ablation_experiments(
        train_feat_df=balanced_train,
        val_feat_df=val_feat,
        val_s1_ids=val_s1_ids,
        s1_to_val_matches=val_gt,
        abs_threshold=best_thresh,
        margin_threshold=best_margin
    )
    print("\nAblation Experiment Results:")
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(RESULTS_DIR / "ablation_experiments.csv", index=False)
    
    # 11. Error Analysis
    val_preds = apply_decision_rules(val_feat, abs_threshold=best_thresh, margin_threshold=best_margin)
    fn_df, fp_df, err_summary = generate_error_analysis(
        val_cand_feat_df=val_feat,
        s1_df=val_s1,
        query_df=val_q,
        s1_to_true_matches=val_gt,
        s1_to_pred_matches=val_preds,
        output_path=RESULTS_DIR / "validation_error_analysis.csv"
    )
    
    # Save final validation metrics
    metrics_summary = {
        "best_threshold": best_thresh,
        "best_margin": best_margin,
        "macro_f0.5": best_f05,
        "macro_precision": opt_results["best_precision"],
        "macro_recall": opt_results["best_recall"],
        "candidate_recall_union": val_recall_stats.get("candidate_recall_union", 0.0),
        "false_positives": err_summary["total_false_positives"],
        "false_negatives": err_summary["total_false_negatives"],
        "candidate_generation_failures": err_summary["candidate_generation_failures"],
        "matcher_threshold_failures": err_summary["matcher_threshold_failures"]
    }
    pd.DataFrame([metrics_summary]).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR / 'metrics.csv'}")
    print("\n[DONE] Leakage-Free Validation & Evaluation Complete.")


if __name__ == "__main__":
    main()
