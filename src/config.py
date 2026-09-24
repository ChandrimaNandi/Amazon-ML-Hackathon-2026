"""
Centralized Configuration for Business Entity Resolution Pipeline.
Auto-detects local directory vs Kaggle dataset environment.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Auto-detect Kaggle Dataset Paths
KAGGLE_DATASET_CANDIDATE = Path("/kaggle/input/entity-data/dataset")

if KAGGLE_DATASET_CANDIDATE.exists():
    DATASET_DIR = KAGGLE_DATASET_CANDIDATE
    OUTPUT_DIR = Path("/kaggle/working/output")
    RESULTS_DIR = Path("/kaggle/working/results")
    LOGS_DIR = Path("/kaggle/working/logs")
elif Path("/kaggle/input").exists():
    # Dynamic search under /kaggle/input if dataset name differs
    found_dataset = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs and "test" in dirs:
            found_dataset = Path(root)
            break
    if found_dataset:
        DATASET_DIR = found_dataset
    else:
        DATASET_DIR = PROJECT_ROOT / "dataset"
    OUTPUT_DIR = Path("/kaggle/working/output")
    RESULTS_DIR = Path("/kaggle/working/results")
    LOGS_DIR = Path("/kaggle/working/logs")
else:
    DATASET_DIR = PROJECT_ROOT / "dataset"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Ensure output directories exist
for p in [OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset Paths
TRAIN_S1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1_PATH = TEST_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DIR / "test_source3.tsv"

SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# General Settings
RANDOM_SEED = 42

# Candidate Generation Settings
BM25_K_VALUES = [5, 10, 20, 50, 100]
DEFAULT_K_NAME = 30
DEFAULT_K_ADDRESS = 20
DEFAULT_K_COMBINED = 30
DEFAULT_K_CHAR_TFIDF = 30

# Character TF-IDF Settings
CHAR_NGRAM_RANGE = (3, 5)
CHAR_TFIDF_MAX_FEATURES = 250000

# Ranking Model Settings
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Evaluation Metric (F0.5 Macro)
BETA = 0.5
