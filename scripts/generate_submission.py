"""
Script to generate submission files and run official validator.
Run from project root: python3 scripts/generate_submission.py
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH,
    TEST_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.normalization import create_normalized_features
from src.candidate_generation import generate_candidate_union
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel
from src.inference import generate_submission_files


def main():
    print("=" * 60)
    print("[1/3] TRAINING MATCHING MODEL FOR SUBMISSION")
    print("=" * 60)
    
    s1_df = load_source_tsv(TRAIN_S1_PATH)
    s2_df = load_source_tsv(TRAIN_S2_PATH)
    s3_df = load_source_tsv(TRAIN_S3_PATH)
    _, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    if len(query_df) > 50000:
        query_df = query_df.sample(n=50000, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(query_df):,} query records for training submission model.")
        
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    cand_df, _ = generate_candidate_union(s1_df, query_df, k_name=20, k_address=15)
    feat_df = extract_candidate_features(cand_df, s1_df, query_df, s1_to_matches)
    
    model = EntityMatcherModel()
    model.fit(feat_df)
    
    print("\n" + "=" * 60)
    print("[2/3] GENERATING TEST SUBMISSION FILES")
    print("=" * 60)
    
    summary = generate_submission_files(
        model=model,
        test_dir=TEST_DIR,
        output_matching_path=SUBMISSION_MATCHING_PATH,
        output_candidate_path=SUBMISSION_CANDIDATE_PATH,
        abs_threshold=0.50,
        margin_threshold=0.05
    )
    
    print("\nSubmission Output Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    print("\n" + "=" * 60)
    print("[3/3] EXECUTING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 60)
    
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(SUBMISSION_MATCHING_PATH),
        "--candidate", str(SUBMISSION_CANDIDATE_PATH),
        "--test-dir", str(TEST_DIR)
    ]
    
    print(f"Command: {' '.join(validator_cmd)}")
    result = subprocess.run(validator_cmd, capture_output=True, text=True)
    
    print("\nValidator Output:")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    print(f"Validator Exit Code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Official submission validation: PASS")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Official submission validation: FAIL")
        print("=" * 60)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
