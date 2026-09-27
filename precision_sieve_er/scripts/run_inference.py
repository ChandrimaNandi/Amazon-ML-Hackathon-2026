"""
Streaming Test Inference Script for Precision-Sieve ER.
Runs Tier 1 exact sieve + GBDT ensemble scoring with disk-sharded streaming (<2.5 GB peak RAM, zero disk bloat).
Automatically validates submission TSVs with utils/validate_submission.py.
"""

import sys
import os
import time
import json
import argparse
import subprocess
import logging
from pathlib import Path

# Ensure root workspace is in sys.path
sys_path_root = str(Path(__file__).resolve().parents[2])
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from precision_sieve_er.configs.default_config import (
    discover_dataset_paths, RESULTS_DIR, OUTPUT_DIR,
    SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE, DEFAULT_ABS_THRESHOLD, DEFAULT_MARGIN_THRESHOLD,
    print_gpu_info
)
from precision_sieve_er.src.classifier import AsymmetricPrecisionEnsemble
from precision_sieve_er.src.pipeline import run_precision_sieve_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PrecisionSieve.Inference")


def parse_args():
    parser = argparse.ArgumentParser(description="Precision-Sieve ER Streaming Test Inference")
    parser.add_argument("--model_path", type=str, default=str(RESULTS_DIR / "precision_ensemble.pkl"), help="Path to trained GBDT ensemble weights")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory containing test_source1-3 TSV files")
    parser.add_argument("--output_matching", type=str, default=str(SUBMISSION_MATCHING_PATH), help="Path for output matching_results.tsv")
    parser.add_argument("--output_candidates", type=str, default=str(SUBMISSION_CANDIDATE_PATH), help="Path for output candidate_pairs.tsv")
    parser.add_argument("--abs_threshold", type=float, default=None, help="Decision boundary threshold (default: loads from best_thresholds.json or 0.60)")
    parser.add_argument("--margin_threshold", type=float, default=None, help="Ambiguity rejection margin (default: loads from best_thresholds.json or 0.10)")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Queries per chunk in streaming loop")
    parser.add_argument("--validate", action="store_true", default=True, help="Run utils/validate_submission.py on completion")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("      PRECISION-SIEVE ER: STREAMING TEST INFERENCE PIPELINE")
    print("=" * 70)
    print_gpu_info()

    # Discover test directory
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
    else:
        _, test_dir = discover_dataset_paths()

    matching_path = Path(args.output_matching).resolve()
    candidate_path = Path(args.output_candidates).resolve()
    model_path = Path(args.model_path).resolve()

    logger.info(f"Test Directory:      {test_dir}")
    logger.info(f"Model Path:          {model_path}")
    logger.info(f"Output Matching:     {matching_path}")
    logger.info(f"Output Candidates:   {candidate_path}")

    assert test_dir.exists(), f"Test directory not found: {test_dir}"
    assert (test_dir / "test_source1.tsv").exists(), f"Missing test_source1.tsv in {test_dir}"

    # Load thresholds
    abs_th = args.abs_threshold
    margin_th = args.margin_threshold

    thresholds_json = model_path.parent / "best_thresholds.json"
    if thresholds_json.exists():
        try:
            with open(thresholds_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if abs_th is None:
                    abs_th = float(cfg.get("abs_threshold", DEFAULT_ABS_THRESHOLD))
                if margin_th is None:
                    margin_th = float(cfg.get("margin_threshold", DEFAULT_MARGIN_THRESHOLD))
                logger.info(f"Loaded optimal thresholds from {thresholds_json.name}: Abs={abs_th:.2f}, Margin={margin_th:.2f}")
        except Exception as e:
            logger.warning(f"Could not read thresholds JSON: {e}")

    if abs_th is None:
        abs_th = DEFAULT_ABS_THRESHOLD
    if margin_th is None:
        margin_th = DEFAULT_MARGIN_THRESHOLD

    # Load Ensemble Model
    if not model_path.exists():
        logger.error(f"Trained model not found at {model_path}!")
        logger.error("Please run `python precision_sieve_er/scripts/run_train.py` first to train the ensemble.")
        sys.exit(1)

    logger.info(f"Loading ensemble model from {model_path}...")
    ensemble = AsymmetricPrecisionEnsemble.load(str(model_path))

    # Run Streaming Inference
    result = run_precision_sieve_inference(
        ensemble=ensemble,
        test_dir=test_dir,
        output_matching_path=matching_path,
        output_candidate_path=candidate_path,
        abs_threshold=abs_th,
        margin_threshold=margin_th,
        chunk_size=args.chunk_size
    )

    print("\n" + "=" * 60)
    print("             INFERENCE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Reference S1 Entities: {result['num_s1']:,}")
    print(f"  Total Processed Queries:{result['total_queries']:,}")
    print(f"  Tier 1 Exact Matches:  {result['tier1_exact_matches']:,}")
    print(f"  Total Predictions:     {result['num_matches']:,}")
    print(f"  Total Candidates:      {result['num_candidates']:,}")
    print(f"  Elapsed Time:          {result['runtime_sec']:.1f} seconds")
    print("=" * 60 + "\n")

    # Run Validation
    if args.validate:
        validate_script = Path(sys_path_root) / "utils" / "validate_submission.py"
        if validate_script.exists():
            logger.info("Executing official submission validator (utils/validate_submission.py)...")
            cmd = [
                sys.executable,
                str(validate_script),
                "--matching", str(matching_path),
                "--candidate", str(candidate_path),
                "--test-dir", str(test_dir)
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            print(ret.stdout)
            if ret.stderr:
                print(ret.stderr)
            if ret.returncode == 0:
                logger.info("Submission validation PASSED perfectly! Files are ready for hackathon upload.")
            else:
                logger.warning(f"Validator exited with status {ret.returncode}.")
        else:
            logger.warning(f"Validator script not found at {validate_script}")


if __name__ == "__main__":
    main()
