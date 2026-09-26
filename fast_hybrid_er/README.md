# Fast Hybrid Neural-Phonetic Entity Resolution (`fast_hybrid_er`)
## Amazon ML Hackathon 2026: Business Entity Resolution Solution

**Target Metric:** Macro $F_{0.5}$ (Precision-Weighted Entity Resolution)  
**Hardware Environment:** Kaggle 2× NVIDIA Tesla T4 GPUs / Multi-Core Linux  
**Official Verification:** Passed official `utils/validate_submission.py` with exit code 0 (`predicted_pairs ⊆ candidate_pairs`).

---

## 1. Architecture Overview

`fast_hybrid_er` combines Locality-Sensitive Hashing (LSH), phonetic super-key blocking, Dual-T4 FP16 dense semantic scoring, and an asymmetric precision-oriented GBDT matcher to deliver extreme throughput and state-of-the-art precision.

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
```

---

## 2. Directory Structure

```text
fast_hybrid_er/
├── README.md                          # Methodology, architecture & reproduction guide
├── requirements.txt                   # Pinned lightweight dependencies
├── fast_hybrid_entity_resolution.ipynb # Complete interactive 14-section notebook
├── configs/
│   └── default_config.py              # Centralized configuration & hardware discovery
├── src/
│   ├── __init__.py
│   ├── phonetics.py                   # Pure Python Soundex, Double Metaphone, Super-keys
│   ├── normalization.py               # Unicode NFKC, French accents, Devanagari transliteration
│   ├── lsh_blocking.py                # Multi-probe MinHash LSH & Inverted Index Blocker
│   ├── dense_encoder.py               # Dual-T4 FP16 PyTorch Subword Semantic Scorer
│   ├── features.py                    # Vectorized hybrid feature extractor (30 signals)
│   ├── classifier.py                  # Asymmetric precision-weighted LightGBM ranker
│   ├── assignment.py                  # Query exclusivity assignment & 2D threshold grid
│   └── pipeline.py                    # Memory-safe chunked streaming test inference
└── scripts/
    ├── run_train.py                   # Entity-disjoint training & validation benchmark
    └── run_inference.py               # Streaming test inference & official validator check
```

---

## 3. Quick Run Instructions

### 1. Training & Threshold Optimization
```bash
python3 fast_hybrid_er/scripts/run_train.py
```

### 2. Streaming Test Inference & Official Submission Validation
```bash
python3 fast_hybrid_er/scripts/run_inference.py
```

### 3. Interactive Jupyter Notebook
Open and run `fast_hybrid_er/fast_hybrid_entity_resolution.ipynb` in Jupyter or Kaggle.
