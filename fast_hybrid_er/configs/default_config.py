"""
Centralized Configuration & Hardware Discovery for Fast Hybrid Neural-Phonetic ER.
Optimized for Kaggle Dual NVIDIA Tesla T4 GPUs (or Single GPU / Multi-Core CPU Fallback).
"""

import os
import sys
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FastHybridER.Config")

# Base Directory Resolution
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PACKAGE_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically inspects available CPU, system RAM, and CUDA GPUs."""
    cuda_avail = HAS_TORCH and torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    torch_ver = torch.__version__ if HAS_TORCH else "N/A"

    gpus = []
    if cuda_avail:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })

    vm = psutil.virtual_memory()
    return {
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
    }


def print_hardware_summary() -> None:
    """Prints environment information in required monitoring format."""
    hw = get_hardware_info()
    print("=" * 60)
    print("Fast Hybrid Neural-Phonetic ER: Hardware Discovery")
    print("=" * 60)
    print(f"GPU count: {hw['gpu_count']}")
    if hw["gpu_count"] > 0:
        for i, g in enumerate(hw["gpus"]):
            print(f"GPU {i}: {g['name']} ({g['total_memory_gb']} GB, CC {g['compute_capability']})")
    else:
        print("GPU 0: None detected (Using multi-core CPU OpenMP fallback)")
    print(f"CUDA version: {hw['cuda_version']}")
    print(f"PyTorch CUDA availability: {hw['cuda_available']}")
    print(f"CPU threads: {hw['cpu_count']} | Available RAM: {hw['available_ram_gb']} GB / {hw['total_ram_gb']} GB")
    print("=" * 60)


def discover_dataset_paths(base_hint: Optional[Path] = None) -> Dict[str, Path]:
    """
    Dynamically discovers training and test TSV files without hardcoded paths.
    Checks base_hint, /kaggle/input, and local workspace roots.
    """
    search_roots = []
    if base_hint and Path(base_hint).exists():
        search_roots.append(Path(base_hint))
    if Path("/kaggle/input").exists():
        search_roots.append(Path("/kaggle/input"))
    search_roots.append(PROJECT_ROOT)

    dataset_dir = None
    for root in search_roots:
        for r, dirs, files in os.walk(root):
            if "train" in dirs and "test" in dirs:
                dataset_dir = Path(r)
                break
        if dataset_dir:
            break

    if not dataset_dir:
        dataset_dir = PROJECT_ROOT / "dataset"

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    def find_file(d: Path, pattern: str) -> Path:
        matches = list(d.glob(pattern))
        if matches:
            return matches[0]
        return d / pattern.replace("*", "")

    return {
        "dataset_dir": dataset_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_s1": find_file(train_dir, "*source1*.tsv"),
        "train_s2": find_file(train_dir, "*source2*.tsv"),
        "train_s3": find_file(train_dir, "*source3*.tsv"),
        "train_gt": find_file(train_dir, "*ground_truth*.tsv"),
        "test_s1": find_file(test_dir, "*source1*.tsv"),
        "test_s2": find_file(test_dir, "*source2*.tsv"),
        "test_s3": find_file(test_dir, "*source3*.tsv"),
    }


# Path references
DISCOVERED_PATHS = discover_dataset_paths()
DATASET_DIR = DISCOVERED_PATHS["dataset_dir"]
TRAIN_DIR = DISCOVERED_PATHS["train_dir"]
TEST_DIR = DISCOVERED_PATHS["test_dir"]

TRAIN_S1_PATH = DISCOVERED_PATHS["train_s1"]
TRAIN_S2_PATH = DISCOVERED_PATHS["train_s2"]
TRAIN_S3_PATH = DISCOVERED_PATHS["train_s3"]
TRAIN_GROUND_TRUTH_PATH = DISCOVERED_PATHS["train_gt"]
TEST_S1_PATH = DISCOVERED_PATHS["test_s1"]
TEST_S2_PATH = DISCOVERED_PATHS["test_s2"]
TEST_S3_PATH = DISCOVERED_PATHS["test_s3"]

# Writable Output Dirs
if Path("/kaggle/input").exists():
    OUTPUT_DIR = Path("/kaggle/working/output")
    RESULTS_DIR = Path("/kaggle/working/results")
    LOGS_DIR = Path("/kaggle/working/logs")
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RESULTS_DIR = PACKAGE_ROOT / "results"
    LOGS_DIR = PACKAGE_ROOT / "logs"

for p in [OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Reproducibility Seed
RANDOM_SEED = 42

# Candidate Generation & Blocking Hyperparameters
LSH_NUM_PERMUTATIONS = 64
LSH_NUM_BANDS = 16
LSH_ROWS_PER_BAND = 4
MAX_BUCKET_CANDIDATES = 20
BM25_TOP_K_NAME = 10
BM25_TOP_K_COMB = 5

# Asymmetric LightGBM Parameters (Optimized for Macro F0.5)
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "max_depth": -1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Evaluation Metric Configuration
BETA = 0.5  # Macro F0.5 (Precision-heavy)
FP_PENALTY_WEIGHT = 2.0  # Training sample weight multiplier on false positives

# Chunking & Streaming
def get_optimal_chunk_size() -> int:
    """Dynamically calculates streaming chunk size based on available system RAM."""
    hw = get_hardware_info()
    avail_ram = hw["available_ram_gb"]
    if avail_ram >= 20.0:
        return 50000
    elif avail_ram >= 12.0:
        return 35000
    else:
        return 20000

DEFAULT_CHUNK_SIZE = get_optimal_chunk_size()
