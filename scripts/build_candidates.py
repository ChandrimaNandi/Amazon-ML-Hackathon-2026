"""
Script to build candidate pairs and evaluate multi-channel Recall@K.
Run from project root: python3 scripts/build_candidates.py [--sample-queries 10000]
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH, RESULTS_DIR
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.normalization import create_normalized_features
from src.candidate_generation import CandidateGenerator
from src.evaluation import evaluate_candidate_recall_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Build Multi-Channel Candidate Pairs")
    parser.add_argument("--sample-s1", type=int, default=25000, help="S1 reference entities sample size")
    parser.add_argument("--sample-queries", type=int, default=10000, help="Query sample size")
    parser.add_argument("--k-name", type=int, default=25, help="BM25 Name Top K")
    parser.add_argument("--k-addr", type=int, default=20, help="Char TF-IDF Address Top K")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"[BUILD CANDIDATES] Running on Train Data ({args.sample_s1:,} S1, {args.sample_queries:,} Queries)")
    print("=" * 60)
    
    s1_df = pd.read_csv(TRAIN_S1_PATH, sep="\t", nrows=args.sample_s1, keep_default_na=False, dtype=str)
    gt_df, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    s1_sample_ids = set(s1_df["entity_id"])
    active_gt = {s1: s1_to_matches.get(s1, set()) for s1 in s1_sample_ids}
    needed_q_ids = set()
    for q_set in active_gt.values():
        needed_q_ids.update(q_set)
        
    s2_df = pd.read_csv(TRAIN_S2_PATH, sep="\t", nrows=args.sample_queries * 2, keep_default_na=False, dtype=str)
    s3_df = pd.read_csv(TRAIN_S3_PATH, sep="\t", nrows=args.sample_queries * 2, keep_default_na=False, dtype=str)
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Filter queries to relevant ones plus negatives
    query_df = query_df[query_df["entity_id"].isin(needed_q_ids) | (query_df.index < args.sample_queries)].head(args.sample_queries).reset_index(drop=True)
    
    print("\nNormalizing text representations...")
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    generator = CandidateGenerator(
        k_bm25_name=args.k_name,
        k_bm25_comb=args.k_name,
        k_tfidf_name=args.k_name,
        k_tfidf_addr=args.k_addr
    )
    generator.fit(s1_df)
    cand_df, stats = generator.generate_candidates(query_df)
    
    print("\nCandidate Generation Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    rec_stats = evaluate_candidate_recall_diagnostics(cand_df, active_gt)
    print("\nRecall Diagnostics:")
    for k, v in rec_stats.items():
        print(f"  {k}: {v}")
        
    rec_csv = RESULTS_DIR / "candidate_recall.csv"
    pd.DataFrame([rec_stats]).to_csv(rec_csv, index=False)
    print(f"\nSaved recall results to {rec_csv}")
    print("\n[DONE] Candidate Building Complete.")


if __name__ == "__main__":
    main()
