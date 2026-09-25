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
    TEST_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH, RESULTS_DIR,
    load_threshold_config
)
from src.data_loader import load_source_tsv, load_ground_truth, load_coherent_training_sample
from src.normalization import create_normalized_features
from src.candidate_generation import CandidateGenerator
from src.features import extract_candidate_features
from src.negative_sampling import build_controlled_training_pairs
from src.ranking import EntityMatcherModel
from src.inference import run_chunked_inference


def main():
    print("=" * 60)
    print("[1/3] PREPARING ENTITY MATCHER MODEL FOR SUBMISSION")
    print("=" * 60)
    
    model_path = RESULTS_DIR / "matcher_model.pkl"
    thresh_cfg = load_threshold_config()
    frozen_thresh = thresh_cfg["abs_threshold"]
    frozen_margin = thresh_cfg["margin_threshold"]
    print(f"Loaded frozen optimal parameters: Threshold={frozen_thresh:.2f}, Margin={frozen_margin:.2f}")
            
    if model_path.exists():
        print(f"Loading pre-trained model from {model_path}...")
        model = EntityMatcherModel.load_model(model_path)
    else:
        print("Training model on coherent training sample...")
        s1_df, query_df, active_gt = load_coherent_training_sample(
            s1_path=TRAIN_S1_PATH,
            gt_path=TRAIN_GROUND_TRUTH_PATH,
            s2_path=TRAIN_S2_PATH,
            s3_path=TRAIN_S3_PATH,
            sample_s1_rows=25000,
            max_active_queries=25000,
            num_unmatched_queries=2000,
            random_seed=42
        )
        s1_df = create_normalized_features(s1_df)
        query_df = create_normalized_features(query_df)
        
        generator = CandidateGenerator()
        generator.fit(s1_df)
        cands, _ = generator.generate_candidates(query_df)
        
        feat_df = extract_candidate_features(cands, s1_df, query_df, active_gt)
        balanced_train, _ = build_controlled_training_pairs(feat_df, max_negatives_per_positive=8)
        
        model = EntityMatcherModel()
        model.fit(balanced_train)
        model.save_model(model_path)
        
    print("\n" + "=" * 60)
    print("[2/3] GENERATING TEST SUBMISSION FILES VIA STREAMING CHUNKS")
    print("=" * 60)
    
    summary = run_chunked_inference(
        model=model,
        test_dir=TEST_DIR,
        output_matching_path=SUBMISSION_MATCHING_PATH,
        output_candidate_path=SUBMISSION_CANDIDATE_PATH,
        abs_threshold=frozen_thresh,
        margin_threshold=frozen_margin,
        chunk_size=50000
    )
    
    print("\nSubmission Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    print("\n" + "=" * 60)
    print("[3/3] EXECUTING SUBMISSION VALIDATOR")
    print("=" * 60)
    
    validator_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
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
        print("Official submission validation: PASS (Ready for Submission!)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Official submission validation: FAIL")
        print("=" * 60)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
