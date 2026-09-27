"""
Centralized Configuration and Dynamic Hardware Discovery for Precision-Sieve ER.
Auto-detects Kaggle environment, dual Tesla T4 GPUs, and discovers the entity-data dataset.
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

logger = logging.getLogger("PrecisionSieve.Config")

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PACKAGE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically queries CPU, RAM, and GPU specs."""
    cuda_avail = HAS_TORCH and torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    
    gpus = []
    if cuda_avail:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2)
            })
            
    vm = psutil.virtual_memory()
    return {
        "cuda_available": cuda_avail,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
    }


def print_gpu_info() -> None:
    """Prints hardware and GPU specifications."""
    info = get_hardware_info()
    gpu_count = info["gpu_count"]
    print(f"GPU count: {gpu_count}")
    if gpu_count > 0:
        for gpu in info["gpus"]:
            print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.2f} GB)")
    else:
        print("GPU: None detected (Using multi-core CPU OpenMP fallback)")
    print(f"CPU count: {info['cpu_count']}")
    print(f"RAM available: {info['available_ram_gb']:.2f} GB / {info['total_ram_gb']:.2f} GB")


def discover_dataset_paths() -> Tuple[Path, Path]:
    """
    Dynamically discovers training and test directories across Kaggle and local workspace.
    Prioritizes user's exact Kaggle dataset mount:
    /kaggle/input/datasets/chandrimanandi/entity-data/dataset
    """
    candidate_train_dirs = [
        Path("/kaggle/input/datasets/chandrimanandi/entity-data/dataset/train"),
        Path("/kaggle/input/datasets/chandrimanandi/entity-data/train"),
        Path("/kaggle/input/entity-data/dataset/train"),
        Path("/kaggle/input/entity-data/train"),
        Path("/kaggle/input/amazon-ml-hackathon-2026/dataset/train"),
        Path("/kaggle/input/amazon-ml-challenge-2026/dataset/train"),
        Path("/kaggle/input/student-resource/dataset/train"),
        Path("/kaggle/input/dataset/train"),
        PROJECT_ROOT / "dataset" / "train",
        Path("./dataset/train"),
        Path("../dataset/train")
    ]
    
    candidate_test_dirs = [
        Path("/kaggle/input/datasets/chandrimanandi/entity-data/dataset/test"),
        Path("/kaggle/input/datasets/chandrimanandi/entity-data/test"),
        Path("/kaggle/input/entity-data/dataset/test"),
        Path("/kaggle/input/entity-data/test"),
        Path("/kaggle/input/amazon-ml-hackathon-2026/dataset/test"),
        Path("/kaggle/input/amazon-ml-challenge-2026/dataset/test"),
        Path("/kaggle/input/student-resource/dataset/test"),
        Path("/kaggle/input/dataset/test"),
        PROJECT_ROOT / "dataset" / "test",
        Path("./dataset/test"),
        Path("../dataset/test")
    ]
    
    train_dir = None
    for p in candidate_train_dirs:
        if p.exists() and (p / "train_ground_truth.tsv").exists():
            train_dir = p.resolve()
            break
            
    # Recursive search under /kaggle/input if needed
    if train_dir is None and Path("/kaggle/input").is_dir():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_ground_truth.tsv" in files:
                train_dir = Path(root).resolve()
                break
                
    test_dir = None
    for p in candidate_test_dirs:
        if p.exists() and (p / "test_source1.tsv").exists():
            test_dir = p.resolve()
            break
            
    if test_dir is None and Path("/kaggle/input").is_dir():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "test_source1.tsv" in files:
                test_dir = Path(root).resolve()
                break
                
    if train_dir is None:
        train_dir = (PROJECT_ROOT / "dataset" / "train").resolve()
    if test_dir is None:
        test_dir = (PROJECT_ROOT / "dataset" / "test").resolve()
        
    logger.info(f"[CONFIG] Discovered Train Directory: {train_dir}")
    logger.info(f"[CONFIG] Discovered Test Directory:  {test_dir}")
    return train_dir, test_dir


# Output and Artifact directories
TRAIN_DIR, TEST_DIR = discover_dataset_paths()

OUTPUT_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", PROJECT_ROOT / "output")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PACKAGE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Core 99+ Hyperparameters
BETA = 0.5                      # Precision weighting beta
FP_PENALTY_WEIGHT = 4.0         # 4x penalty on false positives
DEFAULT_CHUNK_SIZE = 50000      # Queries per streaming chunk (fast and memory-safe)
DEFAULT_ABS_THRESHOLD = 0.60    # High precision decision boundary
DEFAULT_MARGIN_THRESHOLD = 0.10 # Ambiguity rejection margin
