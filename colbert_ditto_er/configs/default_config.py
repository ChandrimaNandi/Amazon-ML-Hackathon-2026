"""
Centralized Configuration and Dynamic Hardware Discovery for ColBERT-Ditto ER.
Auto-detects environment (Local vs Kaggle) and hardware (Dual Tesla T4 GPUs vs Single GPU vs CPU).
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

logger = logging.getLogger("ColBERT_Ditto.Config")

# Base directory: student_resource root
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
COLBERT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically queries CPU, RAM, and CUDA GPU specifications."""
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
                "major_capability": props.major,
                "minor_capability": props.minor,
            })
            
    vm = psutil.virtual_memory()
    info = {
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
    }
    return info


def print_gpu_info() -> None:
    """Prints hardware and GPU information in the required Kaggle monitoring format."""
    info = get_hardware_info()
    gpu_count = info["gpu_count"]
    print(f"GPU count: {gpu_count}")
    if gpu_count > 0:
        for gpu in info["gpus"]:
            print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.2f} GB)")
    else:
        print("GPU: None detected (Using CPU multi-threading OpenMP)")
    print(f"CPU count: {info['cpu_count']}")
    print(f"RAM available: {info['available_ram_gb']:.2f} GB / {info['total_ram_gb']:.2f} GB")


def discover_dataset_paths() -> Tuple[Path, Path]:
    """
    Dynamically discovers training and test directories across Kaggle and local workspace.
    Checks:
    1. /kaggle/input/datasets/chandrimanandi/entity-data/dataset
    2. /kaggle/input/entity-data/dataset
    3. /kaggle/input/amazon-ml-hackathon-2026/dataset/
    4. Recursive scan under /kaggle/input
    5. Local workspace ./dataset
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
            
    # Recursive search under /kaggle/input if not found yet
    if train_dir is None and Path("/kaggle/input").is_dir():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_ground_truth.tsv" in files or any("ground_truth" in f for f in files):
                train_dir = Path(root).resolve()
                break
                
    test_dir = None
    for p in candidate_test_dirs:
        if p.exists() and (p / "test_source1.tsv").exists():
            test_dir = p.resolve()
            break
            
    if test_dir is None and Path("/kaggle/input").is_dir():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "test_source1.tsv" in files or any("test_source" in f for f in files):
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
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = COLBERT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Model Hyperparameters
DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 64          # ColBERT token embedding dimension
DEFAULT_MAX_SEQ_LEN = 72        # Max tokens for serialized Ditto business records
DEFAULT_BATCH_SIZE = 128        # Batch size for token encoding on T4
DEFAULT_CHUNK_SIZE = 30000      # Queries per streaming chunk in test inference
DEFAULT_NUM_CANDIDATES = 12     # Number of candidate reference entities per query
DEFAULT_MARGIN = 0.20           # Margin for pairwise ranking loss
