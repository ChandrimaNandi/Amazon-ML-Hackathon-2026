"""
ColBERT-Ditto Entity Resolution: Streaming Test Inference Script.
Processes multi-million query test sets within a bounded <4.5 GB RAM footprint,
evaluates ColBERT Late-Interaction MaxSim on GPU and verifies submission with validate_submission.py.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from colbert_ditto_er.configs.default_config import (
    TEST_DIR, OUTPUT_DIR, RESULTS_DIR,
    SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE, DEFAULT_BASE_MODEL, DEFAULT_EMBED_DIM, DEFAULT_MAX_SEQ_LEN,
    print_gpu_info
)
from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder
from colbert_ditto_er.src.pipeline import run_streaming_colbert_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ColBERT_Ditto.RunInference")


def main():
    parser = argparse.ArgumentParser(description="ColBERT-Ditto Streaming Test Inference")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--weights-path", type=Path, default=RESULTS_DIR / "colbert_model.pt")
    parser.add_argument("--config-path", type=Path, default=RESULTS_DIR / "threshold_config.json")
    args = parser.parse_args()

    print_gpu_info()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load threshold configuration
    abs_th = 0.45
    margin_th = 0.05
    model_name = DEFAULT_BASE_MODEL

    if args.config_path.exists():
        logger.info(f"Loading threshold configuration from {args.config_path}...")
        with open(args.config_path) as f:
            cfg = json.load(f)
            abs_th = cfg.get("abs_threshold", abs_th)
            margin_th = cfg.get("margin_threshold", margin_th)
            model_name = cfg.get("model_name", model_name)
    else:
        logger.warning(f"Config path {args.config_path} not found. Using default thresholds (abs={abs_th}, margin={margin_th}).")

    # 2. Initialize Model and Load Weights
    logger.info(f"Initializing ColBERT model ({model_name})...")
    model = ColBERTTokenEncoder(
        base_model_name=model_name,
        embed_dim=DEFAULT_EMBED_DIM,
        max_seq_len=DEFAULT_MAX_SEQ_LEN
    )

    if args.weights_path.exists():
        logger.info(f"Loading trained weights from {args.weights_path}...")
        state_dict = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"Model weights {args.weights_path} not found. Running with base pre-trained weights.")

    output_match = args.output_dir / "matching_results.tsv"
    output_cand = args.output_dir / "candidate_pairs.tsv"

    # 3. Run Streaming Inference
    res = run_streaming_colbert_inference(
        model=model,
        test_dir=args.test_dir,
        output_matching_path=output_match,
        output_candidate_path=output_cand,
        abs_threshold=abs_th,
        margin_threshold=margin_th,
        chunk_size=args.chunk_size,
        max_queries=args.max_queries,
        device=device
    )

    # 4. Verify Submission Format with validate_submission.py
    validator_path = PROJECT_ROOT / "utils" / "validate_submission.py"
    if validator_path.exists() and output_match.exists():
        logger.info("=" * 60)
        logger.info(f"[VALIDATION] Validating submission files using {validator_path.name}...")
        cmd = [
            sys.executable, str(validator_path),
            "--matching", str(output_match),
            "--candidate", str(output_cand),
            "--test-dir", str(args.test_dir)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
            
        if proc.returncode == 0:
            logger.info(">>> SUCCESS: Official Submission Validator PASSED (Exit Code 0).")
        else:
            logger.error(f">>> VALIDATION FAILED with exit code {proc.returncode}.")
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
