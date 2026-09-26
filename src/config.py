"""
Centralized, Dynamic Configuration & Hardware Discovery Module.
Supports automatic discovery across local repository and Kaggle environments (including dual NVIDIA T4 GPUs).
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

logger = logging.getLogger(__name__)

# Base Paths
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically queries CPU, RAM, and CUDA GPU specs."""
    cuda_avail = HAS_TORCH and torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    torch_ver = torch.__version__ if HAS_TORCH else "N/A"
    
    gpus = []
    if cuda_avail:
        for i in range(gpu_count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
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
    """
    Prints hardware and GPU information in the required Kaggle monitoring format:
    GPU count:
    GPU 0:
    GPU 1:
    CUDA version:
    PyTorch CUDA availability:
    """
    info = get_hardware_info()
    print(f"GPU count: {info['gpu_count']}")
    if info["gpu_count"] > 0:
        for i, g in enumerate(info["gpus"]):
            print(f"GPU {i}: {g['name']} ({g['total_memory_gb']} GB)")
    else:
        print("GPU 0: None detected (CPU fallback enabled)")
    print(f"CUDA version: {info['cuda_version']}")
    print(f"PyTorch CUDA availability: {info['cuda_available']}")


def release_memory() -> None:
    """Explicitly releases CPU and GPU memory between major pipeline stages."""
    import gc
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


class StageTimer:
    """Tracks elapsed time across major pipeline stages for transparent monitoring."""
    def __init__(self):
        import time
        self.timings: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
        self.total_start: float = time.time()

    def start(self, stage: str) -> None:
        import time
        self._starts[stage] = time.time()

    def stop(self, stage: str) -> float:
        import time
        if stage in self._starts:
            dur = time.time() - self._starts[stage]
            self.timings[stage] = dur
            return dur
        return 0.0

    def print_summary(self) -> None:
        import time
        total_runtime = time.time() - self.total_start
        print("\nStage Timing Summary:")
        print(f"Data loading time:          {self.timings.get('data_loading', 0.0):.2f}s")
        print(f"Normalization time:         {self.timings.get('normalization', 0.0):.2f}s")
        print(f"Candidate-generation time:  {self.timings.get('candidate_generation', 0.0):.2f}s")
        print(f"Feature-extraction time:    {self.timings.get('feature_extraction', 0.0):.2f}s")
        print(f"Training time:              {self.timings.get('training', 0.0):.2f}s")
        print(f"Inference time:             {self.timings.get('inference', 0.0):.2f}s")
        print(f"Total runtime:              {total_runtime:.2f}s")



def discover_dataset_paths(base_hint: Optional[Path] = None) -> Dict[str, Path]:
    """
    Dynamically discovers train and test dataset files without hardcoding folder names.
    Searches base_hint, /kaggle/input, and project directory.
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
        # Fallback to direct name
        clean_name = pattern.replace("*", "")
        return d / clean_name

    paths = {
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
    return paths


# Discover paths dynamically
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

# Writable Output & Results Dirs
if Path("/kaggle/input").exists():
    OUTPUT_DIR = Path("/kaggle/working/output")
    RESULTS_DIR = Path("/kaggle/working/results")
    LOGS_DIR = Path("/kaggle/working/logs")
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"

for p in [OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SUBMISSION_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
SUBMISSION_PREDICTIONS_PATH = OUTPUT_DIR / "predictions.tsv"

# Reproducibility Seed
RANDOM_SEED = 42

# Candidate Generation Hyperparameters
DEFAULT_K_NAME = 25
DEFAULT_K_ADDRESS = 20
DEFAULT_K_COMBINED = 25
DEFAULT_K_CHAR_TFIDF = 25
BM25_K_VALUES = [1, 5, 10, 20, 50]

# Text Retrieval Settings
CHAR_NGRAM_RANGE = (3, 5)
CHAR_TFIDF_MAX_FEATURES = 150000

# Memory-Safe Chunk Sizes
def get_device() -> Any:
    """Returns primary torch device (cuda:0 if available, else cpu)."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu") if HAS_TORCH else "cpu"


def get_available_devices() -> List[Any]:
    """
    Returns list of all available PyTorch devices without assuming 2 GPUs.
    If 2 GPUs present: [torch.device("cuda:0"), torch.device("cuda:1")].
    If 1 GPU present: [torch.device("cuda:0")].
    If 0 GPUs: [torch.device("cpu")].
    """
    if HAS_TORCH and torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(count)]
    return [torch.device("cpu")] if HAS_TORCH else ["cpu"]


def get_optimal_batch_and_chunk_sizes() -> Tuple[int, int]:
    """
    Dynamically adjusts streaming chunk size and retrieval batch size based on:
    - Available CPU RAM
    - Number of available CUDA GPUs
    - User environment overrides
    """
    # Check environment variable overrides first
    env_chunk = os.environ.get("CHUNK_SIZE")
    env_batch = os.environ.get("RETRIEVAL_BATCH")
    if env_chunk and env_batch:
        return int(env_chunk), int(env_batch)

    hw = get_hardware_info()
    avail_ram = hw["available_ram_gb"]
    n_gpus = hw["gpu_count"]

    if n_gpus >= 2 and avail_ram >= 12.0:
        chunk_size = 50000
        batch_size = 2500
    elif n_gpus >= 1 and avail_ram >= 10.0:
        chunk_size = 40000
        batch_size = 2000
    elif avail_ram >= 10.0:
        chunk_size = 30000
        batch_size = 2000
    else:
        chunk_size = 15000
        batch_size = 1000

    if env_chunk:
        chunk_size = int(env_chunk)
    if env_batch:
        batch_size = int(env_batch)

    return chunk_size, batch_size


# Memory-Safe Chunk Sizes (Dynamically Determined)
DEFAULT_CHUNK_SIZE, DEFAULT_RETRIEVAL_BATCH = get_optimal_batch_and_chunk_sizes()

# Precision-Oriented Ranking Model Parameters (LightGBM)
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 45,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Target Evaluation Metric: Macro F0.5
BETA = 0.5

# Configurable Data Sampling Parameters (Zero Hardcoding)
# Allows seamless scaling between quick notebook verification and full multi-million row inference
SAMPLE_S1_ROWS: Optional[int] = int(os.environ.get("SAMPLE_S1_ROWS", 25000)) if os.environ.get("SAMPLE_S1_ROWS", "25000") != "0" else None
SAMPLE_QUERY_ROWS: Optional[int] = int(os.environ.get("SAMPLE_QUERY_ROWS", 25000)) if os.environ.get("SAMPLE_QUERY_ROWS", "25000") != "0" else None
SAMPLE_ACTIVE_QUERIES: Optional[int] = int(os.environ.get("SAMPLE_ACTIVE_QUERIES", 25000)) if os.environ.get("SAMPLE_ACTIVE_QUERIES", "25000") != "0" else None
MAX_TEST_QUERIES: Optional[int] = int(os.environ.get("MAX_TEST_QUERIES", 0)) if os.environ.get("MAX_TEST_QUERIES", "0") not in ("0", "none", "None", "") else None


def save_threshold_config(
    abs_threshold: float,
    margin_threshold: float,
    path: Optional[Path] = None,
    extra_metrics: Optional[Dict[str, Any]] = None
) -> Path:
    """Persists optimal validation thresholds to a JSON artifact for reproducible test inference."""
    import json
    save_path = path or (RESULTS_DIR / "threshold_config.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "abs_threshold": float(abs_threshold),
        "margin_threshold": float(margin_threshold),
        "beta": float(BETA),
        "metrics": extra_metrics or {}
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"[CONFIG] Saved threshold config to {save_path}")
    return save_path


def load_threshold_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads optimal validation thresholds from disk, with fallback defaults if missing."""
    import json
    load_path = path or (RESULTS_DIR / "threshold_config.json")
    if load_path.exists():
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[CONFIG] Loaded frozen threshold config from {load_path}")
            return data
        except Exception as e:
            logger.warning(f"[CONFIG] Failed reading {load_path}: {e}. Using defaults.")
    # Safe defaults
    return {
        "abs_threshold": 0.65,
        "margin_threshold": 0.05,
        "beta": BETA,
        "metrics": {}
    }

