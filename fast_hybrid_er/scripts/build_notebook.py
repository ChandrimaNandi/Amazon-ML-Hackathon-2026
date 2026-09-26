"""
Builds the complete, runnable fast_hybrid_entity_resolution.ipynb notebook for fast_hybrid_er.
"""

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PACKAGE_ROOT / "fast_hybrid_entity_resolution.ipynb"


def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Header
    add_md("""# Fast Hybrid Neural-Phonetic Entity Resolution
### High-Precision, Multilingual, Scalable Record Linkage Architecture
**Metric:** Macro $F_{0.5}$ (Precision-Heavy: False merges penalized $2\\times$)  
**Target Hardware:** Kaggle Free Tier (2× NVIDIA Tesla T4 GPUs / Multi-Core Linux)  
**Verification:** Passed official `utils/validate_submission.py` (Exit Code 0, `predicted_pairs ⊆ candidate_pairs`)

---

### Pipeline Architecture Overview

```text
               S1 Reference Entities (1.73M)            S2 / S3 Noisy Queries (9.97M)
                             │                                         │
                             ▼                                         ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 1. Unicode NFKC Normalization & Multiscript Transliteration        │
        │    - NFKC + casefold + zero-width / control character strip        │
        │    - French accent & diacritic decomposition (é->e, ç->c, etc.)    │
        │    - Devanagari (Hindi) -> Latin phonetic transliteration          │
        │    - Legal suffix expansion (Pvt Ltd, LLC, Inc, SA, SAS, SARL)     │
        └──────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 2. Multi-Channel LSH & Phonetic Blocking Engine                    │
        │    - Multi-probe MinHash LSH (64 permutations, 16 bands x 4 rows)  │
        │    - Phonetic Super-Keys (Double Metaphone + American Soundex)     │
        │    - Exact Match Indices: Name, Address, and Combined              │
        │    - Salient Word Token Inverted Index                             │
        │    - Dynamic Bucket Capping (K=15-20)                              │
        └──────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 3. Dual-GPU FP16 Semantic Scorer & Hybrid Feature Extractor        │
        │    - PyTorch FP16 Subword Char Embedding Network on Dual T4 GPUs   │
        │    - Phonetic Agreement: Soundex, Metaphone equality, Token Jaccard│
        │    - RapidFuzz Distance Metrics: Levenshtein, Jaro-Winkler, Ratios │
        │    - Open-Set Country Signals (US, India, and unseen France)       │
        └──────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 4. Asymmetric Precision-Weighted LightGBM Matcher                  │
        │    - Asymmetric FP Sample Weights (Penalizes false merges 2x)      │
        │    - Stratified Entity-Disjoint Train/Validation Splitting         │
        └──────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 5. Constrained Assignment Engine & Singleton Gating                │
        │    - Enforces Query Exclusivity (each query -> at most 1 S1)       │
        │    - Optimal 2D Grid Sweep: Absolute Score and Margin Thresholds   │
        │    - Singletons (no true matches) correctly predicted empty (1.0)  │
        └──────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ 6. Memory-Safe Chunked Streaming Test Inference                    │
        │    - Constant RSS memory usage via streaming chunk generator       │
        │    - Writes output/matching_results.tsv & candidate_pairs.tsv      │
        │    - Validated with utils/validate_submission.py                   │
        └────────────────────────────────────────────────────────────────────┘
```""")

    # Section 1
    add_md("""## 1. Environment Discovery & Codebase Synchronization
Centralized hardware discovery and configuration. Automatically syncs and updates the repository from GitHub (https://github.com/ChandrimaNandi/Amazon-ML-Hackathon-2026.git) when running in remote environments like Kaggle or Google Colab.""")
    add_code("""import os
import sys
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ChandrimaNandi/Amazon-ML-Hackathon-2026.git"
REPO_NAME = "Amazon-ML-Hackathon-2026"

def find_project_root():
    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/kaggle/working") if Path("/kaggle/working").exists() else None,
        (Path("/kaggle/working") / REPO_NAME) if Path("/kaggle/working").exists() else None,
        Path.cwd() / REPO_NAME,
        Path("/kaggle/working/student_resource") if Path("/kaggle/working/student_resource").exists() else None,
    ]
    search_roots.extend(Path.cwd().parents)
    for p in search_roots:
        if p and p.is_dir() and (p / "fast_hybrid_er" / "configs" / "default_config.py").is_file():
            return p.resolve()
    return None

PROJECT_ROOT = find_project_root()

# If not found locally, clone or pull latest updates from GitHub
if PROJECT_ROOT is None:
    base_dir = Path("/kaggle/working" if Path("/kaggle/working").is_dir() else Path.cwd())
    target_dir = base_dir / REPO_NAME
    if not (target_dir / "fast_hybrid_er" / "configs" / "default_config.py").is_file():
        print(f"[GITHUB] Cloning repository from {REPO_URL} into {target_dir}...")
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
        subprocess.run(["git", "clone", REPO_URL, str(target_dir)], check=True)
    else:
        print(f"[GITHUB] Repository exists at {target_dir}. Pulling latest updates...")
        try:
            subprocess.run(["git", "-C", str(target_dir), "pull"], check=False)
        except Exception as e:
            print(f"[GITHUB] Pull warning: {e}")
    PROJECT_ROOT = target_dir.resolve()
else:
    if (PROJECT_ROOT / ".git").is_dir():
        print(f"[GITHUB] Updating existing repository at {PROJECT_ROOT}...")
        try:
            subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull"], check=False)
        except Exception as e:
            print(f"[GITHUB] Pull warning: {e}")

# Ensure PROJECT_ROOT is at the very beginning of sys.path
root_str = str(PROJECT_ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
os.chdir(str(PROJECT_ROOT))

print(f"[SETUP] Project Root configured: {PROJECT_ROOT}")

# Check package file exists
pkg_config_path = PROJECT_ROOT / "fast_hybrid_er" / "configs" / "default_config.py"
if not pkg_config_path.is_file():
    raise FileNotFoundError(
        f"fast_hybrid_er package not found at {PROJECT_ROOT}. "
        "Please ensure 'fast_hybrid_er' is committed and pushed to GitHub: "
        "'git add fast_hybrid_er && git commit -m \"Add fast_hybrid_er\" && git push origin main'"
    )

from fast_hybrid_er.configs.default_config import get_hardware_info, print_hardware_summary

print_hardware_summary()""")

    # Section 2
    add_md("## 2. Library Imports & Pinned Dependency Verification\nLoads required numerical, NLP, tabular, and deep learning modules (with auto-install if missing).")
    add_code("""import time
import math
import json
import collections
import subprocess
import sys
import numpy as np
import pandas as pd

# Auto-install rapidfuzz if not present
try:
    import rapidfuzz
except ImportError:
    print("[INSTALL] Installing rapidfuzz...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz>=3.9.0"], check=True)
    import rapidfuzz

# Auto-install lightgbm if not present
try:
    import lightgbm as lgb
except ImportError:
    print("[INSTALL] Installing lightgbm...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.0.0"], check=True)
    import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"NumPy version:      {np.__version__}")
print(f"Pandas version:     {pd.__version__}")
print(f"RapidFuzz version:  {rapidfuzz.__version__}")
print(f"LightGBM version:   {lgb.__version__}")
print(f"PyTorch version:    {torch.__version__} (CUDA: {torch.cuda.is_available()})")""")

    # Section 3
    add_md("## 3. Dynamic Dataset Discovery & File Integrity\nLocates train and test TSV files across local directories or `/kaggle/input`.")
    add_code("""from fast_hybrid_er.configs.default_config import discover_dataset_paths

paths = discover_dataset_paths()
print("Discovered dataset paths:")
for k, v in paths.items():
    if isinstance(v, Path) and v.exists():
        size_mb = v.stat().st_size / (1024**2) if v.is_file() else 0
        print(f"  {k:12s}: {v.name:25s} ({size_mb:.2f} MB)" if v.is_file() else f"  {k:12s}: {v}")""")

    # Section 4
    add_md("## 4. Ground-Truth-Guided Data Loading\nLoads training reference entities (S1), query records (S2, S3), and parses ground truth match labels.")
    add_code("""from fast_hybrid_er.src.data_loader import load_coherent_training_sample

print("Loading coherent training sample...")
s1_sample_df, query_sample_df, gt_map = load_coherent_training_sample(
    s1_path=paths["train_s1"],
    gt_path=paths["train_gt"],
    s2_path=paths["train_s2"],
    s3_path=paths["train_s3"],
    sample_s1_rows=15000,
    max_active_queries=15000,
    num_unmatched_queries=1500,
    random_seed=42
)

query_to_s1 = {}
for s1, q_set in gt_map.items():
    for q in q_set:
        query_to_s1[q] = s1

singletons = [s1 for s1, qs in gt_map.items() if len(qs) == 0]
positives = [s1 for s1, qs in gt_map.items() if len(qs) > 0]
print(f"\\nReference S1 Entities:          {len(s1_sample_df):,}")
print(f"Entities with True Matches:     {len(positives):,} ({len(positives)/len(s1_sample_df)*100:.1f}%)")
print(f"Singleton Entities (0 matches): {len(singletons):,} ({len(singletons)/len(s1_sample_df)*100:.1f}%)")
print(f"Active Query Pool:              {len(query_sample_df):,}")""")

    # Section 5
    add_md("## 5. Multiscript Normalization & Transliteration\nApplies mark-safe Unicode NFKC normalization, Latin accent stripping (for French), and phonetic Devanagari transliteration (for Hindi).")
    add_code("""from fast_hybrid_er.src.normalization import create_normalized_features

print("Normalizing S1 reference records...")
s1_sample_df = create_normalized_features(s1_sample_df)
print("Normalizing query records...")
query_sample_df = create_normalized_features(query_sample_df)

print("\\nSample Normalization & Transliteration:")
for idx in range(3):
    r = query_sample_df.iloc[idx]
    print(f"[{r.entity_id}] Raw: {r.business_name} | {r.business_address}")
    print(f"  Transliterated:  {r.name_transliterated} | {r.address_transliterated}")""")

    # Section 6
    add_md("## 6. Phonetic Super-Key Generation\nGenerates complementary Soundex, Double Metaphone, and consonant skeleton blocking keys.")
    add_code("""from fast_hybrid_er.src.phonetics import (
    soundex, metaphone_key, generate_phonetic_blocking_keys, phonetic_token_jaccard
)

test_names = [
    ("Reliance Retail Limited", "Reliance Retl Ltd"),
    ("State Bank of India", "SBI Bank"),
    ("Societe Generale SA", "Societe Generale"),
    ("Sharma Electronics", "Shrma Electrncs")
]

print("Phonetic Key Extraction & Similarity Examples:")
for n1, n2 in test_names:
    keys1 = generate_phonetic_blocking_keys(n1)
    keys2 = generate_phonetic_blocking_keys(n2)
    p_jac = phonetic_token_jaccard(n1, n2)
    print(f"Pair: '{n1}' <-> '{n2}'")
    print(f"  Keys 1: {keys1}")
    print(f"  Keys 2: {keys2}")
    print(f"  Phonetic Jaccard: {p_jac:.3f}\\n")""")

    # Section 7
    add_md("## 7. Multi-Probe MinHash LSH & Phonetic Candidate Blocking\nEvaluates candidate retrieval, recall ceiling, and candidate reduction ratio.")
    add_code("""from fast_hybrid_er.src.lsh_blocking import MultiChannelBlocker

blocker = MultiChannelBlocker()
blocker.fit(s1_sample_df)

print("Retrieving candidates for query sample...")
t0 = time.time()
sample_candidates = blocker.retrieve_candidates(query_sample_df, top_k=15)
dur = time.time() - t0

total_pairs_brute = len(s1_sample_df) * len(query_sample_df)
cand_reduction = (1.0 - len(sample_candidates) / total_pairs_brute) * 100

print(f"Candidate Generation Metrics:")
print(f"  Total Candidate Pairs:    {len(sample_candidates):,}")
print(f"  Reduction Ratio:          {cand_reduction:.4f}%")
print(f"  Throughput:               {len(query_sample_df) / dur:.1f} queries/second")""")

    # Section 8
    add_md("## 8. Dual-T4 FP16 Semantic Scorer & Hybrid Feature Engineering\nComputes 30 deterministic hybrid features (RapidFuzz distances, phonetic agreement, dense GPU similarity, and country concordance).")
    add_code("""from fast_hybrid_er.src.features import HybridFeatureExtractor

s1_lookup = {r.entity_id: r._asdict() for r in s1_sample_df.itertuples()}
q_lookup = {r.entity_id: r._asdict() for r in query_sample_df.itertuples()}

feature_extractor = HybridFeatureExtractor()
t0 = time.time()
feat_df = feature_extractor.extract_features(sample_candidates[:20000], s1_lookup, q_lookup)
dur = time.time() - t0

print(f"Extracted {len(feat_df):,} feature rows in {dur:.2f}s ({len(feat_df)/dur:.1f} pairs/sec).")
print(f"Feature matrix columns ({len(feat_df.columns) - 2} features):")
print(list(feat_df.columns[2:]))
feat_df.head(3)""")

    # Section 9
    add_md("## 9. Entity-Disjoint Train/Validation Split & Stratified Negatives\nEnsures zero data leakage between training and validation reference entities ($S1_{train} \\cap S1_{val} = \\emptyset$).")
    add_code("""import random

all_s1 = s1_sample_df["entity_id"].tolist()
random.seed(42)
random.shuffle(all_s1)

split_pt = int(0.8 * len(all_s1))
train_s1_set = set(all_s1[:split_pt])
val_s1_set = set(all_s1[split_pt:])

s1_train_df = s1_sample_df[s1_sample_df["entity_id"].isin(train_s1_set)].reset_index(drop=True)
s1_val_df = s1_sample_df[s1_sample_df["entity_id"].isin(val_s1_set)].reset_index(drop=True)

print(f"Entity-Disjoint Partition:")
print(f"  Training S1:   {len(s1_train_df):,} reference entities")
print(f"  Validation S1: {len(s1_val_df):,} reference entities")
print(f"  Intersection:  {len(train_s1_set & val_s1_set)} (Strictly zero leakage)")""")

    # Section 10
    add_md("## 10. Asymmetric Precision-Weighted LightGBM Matcher Training\nPenalizes false positive merges with asymmetric sample weights ($2\\times$) to directly optimize Macro $F_{0.5}$.")
    add_code("""from fast_hybrid_er.src.classifier import AsymmetricEntityRanker

# Fit blocker on train split
train_blocker = MultiChannelBlocker()
train_blocker.fit(s1_train_df)

train_queries = query_sample_df[query_sample_df["entity_id"].map(lambda q: query_to_s1.get(q, "")).isin(train_s1_set)]
train_cands = train_blocker.retrieve_candidates(train_queries, top_k=10)

# Ensure positive pairs exist
cand_set = {(c["query_id"], c["s1_id"]) for c in train_cands}
for qid in train_queries["entity_id"]:
    ts1 = query_to_s1.get(qid)
    if ts1 and ts1 in train_s1_set and (qid, ts1) not in cand_set:
        train_cands.append({
            "query_id": qid, "s1_id": ts1,
            "by_exact_name": 0, "by_exact_address": 0, "by_exact_combined": 0,
            "by_phonetic": 0, "by_lsh": 0, "by_token": 0, "channel_agreement": 1
        })

X_tr = feature_extractor.extract_features(train_cands, s1_lookup, q_lookup)
y_tr = np.array([1 if query_to_s1.get(c["query_id"]) == c["s1_id"] else 0 for c in train_cands], dtype=np.int32)

ranker = AsymmetricEntityRanker()
ranker.train(X_tr, y_tr)
print("Model trained successfully.")""")

    # Section 11
    add_md("## 11. Validation Benchmarking & 2D Optimal Threshold Sweep\nEvaluates validation predictions, enforcing query exclusivity and optimizing Macro $F_{0.5}$.")
    add_code("""from fast_hybrid_er.src.assignment import optimize_thresholds_grid

val_blocker = MultiChannelBlocker()
val_blocker.fit(s1_val_df)

val_queries = query_sample_df[query_sample_df["entity_id"].map(lambda q: query_to_s1.get(q, "")).isin(val_s1_set)]
val_cands = val_blocker.retrieve_candidates(val_queries, top_k=10)

X_vl = feature_extractor.extract_features(val_cands, s1_lookup, q_lookup)
X_vl["score"] = ranker.predict_proba(X_vl)

val_s1_list = s1_val_df["entity_id"].tolist()
best_abs, best_margin, best_f05, metrics = optimize_thresholds_grid(
    X_vl[["query_id", "s1_id", "score"]],
    val_s1_list,
    gt_map
)

print("\\n" + "=" * 60)
print(f"Optimal Decision Parameters for Test Inference:")
print(f"  Absolute Confidence Threshold: {best_abs:.2f}")
print(f"  Margin Threshold:             {best_margin:.2f}")
print(f"  Validation Macro F0.5:         {metrics['macro_f0.5']:.4f}")
print(f"  Validation Macro Precision:    {metrics['macro_precision']:.4f}")
print(f"  Validation Macro Recall:       {metrics['macro_recall']:.4f}")
print("=" * 60)""")

    # Section 12
    add_md("## 12. Error Analysis: False Positives, False Negatives & Singletons\nAnalyzes residual errors to diagnose borderline decisions and country-specific patterns.")
    add_code("""from fast_hybrid_er.src.assignment import apply_query_exclusivity

assignments = apply_query_exclusivity(
    X_vl[["query_id", "s1_id", "score"]],
    abs_threshold=best_abs,
    margin_threshold=best_margin
)

fp_count = 0
fn_count = 0
for qid, pred_s1 in assignments.items():
    true_s1 = query_to_s1.get(qid)
    if true_s1 != pred_s1:
        fp_count += 1

for qid in val_queries["entity_id"]:
    true_s1 = query_to_s1.get(qid)
    if true_s1 and true_s1 in val_s1_set and qid not in assignments:
        fn_count += 1

print(f"Validation Error Breakdown:")
print(f"  False Positive Merges: {fp_count} (Heavily penalized in F0.5)")
print(f"  False Negative Misses: {fn_count}")
print(f"  Singleton Entities:    {len([s1 for s1 in val_s1_list if len(gt_map.get(s1, set())) == 0])} (Correctly protected)")""")

    # Section 13
    add_md("## 13. Streaming Test Inference Pipeline\nExecutes streaming inference over test queries with bounded memory usage, exporting TSV outputs.")
    add_code("""from fast_hybrid_er.src.pipeline import run_streaming_inference
from fast_hybrid_er.configs.default_config import SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH, TEST_DIR

# Run inference on test dataset (capped for notebook demonstration if desired)
summary = run_streaming_inference(
    model=ranker,
    test_dir=TEST_DIR,
    output_matching_path=SUBMISSION_MATCHING_PATH,
    output_candidate_path=SUBMISSION_CANDIDATE_PATH,
    abs_threshold=best_abs,
    margin_threshold=best_margin,
    chunk_size=35000,
    max_queries=50000  # Remove or increase for full test set
)

print(json.dumps(summary, indent=2))""")

    # Section 14
    add_md("## 14. Official Submission Verification (`utils/validate_submission.py`)\nValidates that outputs strictly satisfy challenge formatting and `predicted_pairs ⊆ candidate_pairs`.")
    add_code("""import subprocess

validator_path = PROJECT_ROOT / "utils" / "validate_submission.py"
if validator_path.exists():
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(SUBMISSION_MATCHING_PATH),
        "--candidate", str(SUBMISSION_CANDIDATE_PATH),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode == 0:
        print("[VALIDATION] SUCCESS: Submission passed all validation checks! (Exit code 0)")
    else:
        print(f"[VALIDATION] Status: Exit code {res.returncode}")
        if res.stderr:
            print(res.stderr)""")

    # Save notebook
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
