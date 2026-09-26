"""
Inference & Submission Validation Runner for Fast Hybrid ER.
Loads trained model and optimal thresholds, runs streaming test inference,
and executes official validation script to verify format and candidate subset rules.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fast_hybrid_er.configs.default_config import (
    TEST_DIR, RESULTS_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    print_hardware_summary
)
from fast_hybrid_er.src.classifier import AsymmetricEntityRanker
from fast_hybrid_er.src.pipeline import run_streaming_inference


def run_test_and_validation(max_test_queries: int = None):
    """Loads trained model, executes inference, and validates submission."""
    print_hardware_summary()

    # Load Model
    model_path = RESULTS_DIR / "hybrid_er_model.pkl"
    if not model_path.exists():
        print(f"Model file {model_path} not found. Please run scripts/run_train.py first.")
        sys.exit(1)

    ranker = AsymmetricEntityRanker()
    ranker.load(model_path)

    # Load Thresholds
    thresh_path = RESULTS_DIR / "threshold_config.json"
    abs_th = 0.50
    margin_th = 0.05
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            abs_th = float(cfg.get("abs_threshold", 0.50))
            margin_th = float(cfg.get("margin_threshold", 0.05))
            print(f"Loaded frozen thresholds: abs={abs_th:.2f}, margin={margin_th:.2f}")

    # Run Streaming Inference
    summary = run_streaming_inference(
        model=ranker,
        test_dir=TEST_DIR,
        output_matching_path=SUBMISSION_MATCHING_PATH,
        output_candidate_path=SUBMISSION_CANDIDATE_PATH,
        abs_threshold=abs_th,
        margin_threshold=margin_th,
        max_queries=max_test_queries
    )

    # Official Submission Verification
    validator_path = PROJECT_ROOT / "utils" / "validate_submission.py"
    if validator_path.exists():
        print("\n" + "=" * 60)
        print("[VALIDATION] Running Official Submission Validator...")
        print("=" * 60)
        cmd = [
            sys.executable,
            str(validator_path),
            "--matching", str(SUBMISSION_MATCHING_PATH),
            "--candidate", str(SUBMISSION_CANDIDATE_PATH),
            "--test-dir", str(TEST_DIR)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("[VALIDATION] SUCCESS: Submission passed all validation checks! (Exit code 0)")
        else:
            print(f"[VALIDATION] WARNING: Validator returned exit code {res.returncode}")

    return summary


if __name__ == "__main__":
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_test_and_validation(max_test_queries=max_q)
