"""
Script to build candidate pairs.
Run from project root: python3 scripts/build_candidates.py [--test]
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, RESULTS_DIR
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.normalization import create_normalized_features
from src.candidate_generation import generate_candidate_union
from src.evaluation import evaluate_candidate_recall


def main():
    parser = argparse.ArgumentParser(description="Build Candidate Pairs for Entity Resolution")
    parser.add_argument("--test", action="store_true", help="Build candidates for test set instead of train set")
    parser.add_argument("--k-name", type=int, default=30, help="BM25 Name Top K")
    parser.add_argument("--k-addr", type=int, default=20, help="Char TF-IDF Address Top K")
    args = parser.parse_args()
    
    if args.test:
        print("=" * 60)
        print("[BUILD CANDIDATES] Running on TEST Dataset")
        print("=" * 60)
        s1_df = load_source_tsv(TEST_S1_PATH)
        s2_df = load_source_tsv(TEST_S2_PATH)
        s3_df = load_source_tsv(TEST_S3_PATH)
        s1_to_matches = None
    else:
        print("=" * 60)
        print("[BUILD CANDIDATES] Running on TRAIN Dataset")
        print("=" * 60)
        s1_df = load_source_tsv(TRAIN_S1_PATH)
        s2_df = load_source_tsv(TRAIN_S2_PATH)
        s3_df = load_source_tsv(TRAIN_S3_PATH)
        _, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
        
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    cand_df, stats = generate_candidate_union(
        s1_df=s1_df,
        query_df=query_df,
        k_name=args.k_name,
        k_address=args.k_addr,
        k_combined=args.k_name,
        k_char=args.k_name
    )
    
    print("\nCandidate Generation Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    if s1_to_matches is not None:
        rec_stats = evaluate_candidate_recall(cand_df, s1_to_matches)
        for k, v in rec_stats.items():
            print(f"  {k}: {v}")
            
        # Log to results/candidate_recall.csv
        rec_csv = RESULTS_DIR / "candidate_recall.csv"
        rec_df = pd.DataFrame([{
            "K_name": args.k_name,
            "K_addr": args.k_addr,
            "candidate_recall": rec_stats["candidate_recall"],
            "total_true_pairs": rec_stats["total_true_pairs"],
            "found_pairs": rec_stats["found_pairs"],
            "avg_candidates_per_query": stats["avg_candidates_per_query"],
            "elapsed_seconds": stats["elapsed_seconds"],
        }])
        rec_df.to_csv(rec_csv, index=False)
        print(f"\nSaved recall results to {rec_csv}")
        
    print("\n[DONE] Candidate Building Complete.")


if __name__ == "__main__":
    main()
