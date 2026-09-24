# Business Entity Resolution / Record Linkage Pipeline

A complete, high-precision, multilingual, scalable **Business Entity Resolution / Record Linkage** solution designed for the Business Entity Resolution Challenge 2026 dataset.

## Problem Statement

The goal is to match noisy business records from **Source 2** (S2) and **Source 3** (S3) against a deduplicated reference dataset **Source 1** (S1).
- **Source 1 (S1)**: Reference deduplicated business entities.
- **Source 2 (S2)**: Noisy business records.
- **Source 3 (S3)**: Noisy business records.

Each S1 entity can correspond to **zero**, **one**, or **multiple** matching records across S2 and S3.
Target Metric: **Macro F0.5**, which heavily prioritizes precision and penalizes false positive matches (especially on singleton/zero-match entities).

---

## Technical Constraints & Compliance
- **No LLMs, Commercial APIs, or External Geocoding**.
- **MIT / Apache 2.0 Licensed** models and libraries.
- Open-set `country` feature (handles France, India, US, and unseen countries).
- Full candidate set guarantee: Every predicted match is present in `candidate_pairs.tsv`.
- Multilingual and multiscript Unicode-safe processing (Devanagari, Latin, Arabic, Cyrillic, CJK, etc.).

---

## Pipeline Architecture

```
dataset/ (train & test TSVs)
   │
   ▼
src/data_loader.py (Memory-efficient TSV reading & ground truth parsing)
   │
   ▼
src/normalization.py (Unicode NFKC, case fold, punctuation cleaning)
   │
   ▼
src/candidate_generation.py (Exact Match + BM25 + Char-TFIDF Cosine Retrieval Union)
   │
   ▼
src/features.py (Pairwise edit distance, Jaro-Winkler, token Jaccard, retrieval agreement, script/country pairs)
   │
   ▼
src/ranking.py (LightGBM classifier + hard-negative mining)
   │
   ▼
src/thresholding.py & src/singleton.py (Margin & confidence logic optimized for Macro F0.5)
   │
   ▼
src/inference.py (Generates output/matching_results.tsv & output/candidate_pairs.tsv)
   │
   ▼
utils/validate_submission.py (Official submission validator check)
```

---

## Directory Structure

```text
student_resource/
├── notebooks/
│   └── entity_resolution_experiments.ipynb   # Main interactive notebook (22 sections)
├── src/
│   ├── __init__.py
│   ├── config.py                              # Centralized configuration & parameters
│   ├── data_loader.py                         # TSV data reading & parsing
│   ├── profiling.py                           # Dataset statistics & script profiling
│   ├── normalization.py                       # Unicode NFKC & text representations
│   ├── retrieval.py                           # BM25 & Char TF-IDF retrievers
│   ├── candidate_generation.py                # Multi-view candidate union & Recall@K
│   ├── similarity.py                          # String distance metrics (RapidFuzz)
│   ├── features.py                            # Pairwise candidate feature extractor
│   ├── ranking.py                             # LightGBM matcher & hard-negative miner
│   ├── thresholding.py                        # Decision rule & margin logic
│   ├── singleton.py                           # Singleton entity rejection analysis
│   ├── evaluation.py                          # Macro F0.5, Precision, Recall metrics
│   └── inference.py                           # Submission generator
├── scripts/
│   ├── profile_dataset.py                     # Profile shapes, missing data, countries, scripts
│   ├── build_candidates.py                    # Build candidate pairs & evaluate Recall@K
│   ├── train_model.py                         # Train GBDT model with hard-negative mining
│   ├── evaluate.py                            # Run local validation & compute Macro F0.5
│   └── generate_submission.py                 # Generate submission files & run validator
├── output/
│   ├── matching_results.tsv                   # Final matched entity predictions
│   └── candidate_pairs.tsv                    # Blocking candidate pairs
├── results/
│   ├── experiments.csv                        # Logged experimental metrics
│   ├── candidate_recall.csv                   # Recall@K metrics
│   └── metrics.csv                            # Final validation scores
├── requirements.txt                           # Project dependencies
└── README.md                                  # Documentation
```

---

## Quick Start & Reproduction Commands

### 1. Dataset Profiling
```bash
python3 scripts/profile_dataset.py
```

### 2. Candidate Generation & Recall@K Evaluation
```bash
python3 scripts/build_candidates.py --k-name 30 --k-addr 20
```

### 3. Model Training & Hard Negative Mining
```bash
python3 scripts/train_model.py --sample-size 50000
```

### 4. Validation & Macro F0.5 Evaluation
```bash
python3 scripts/evaluate.py
```

### 5. Generate Submission & Execute Official Validator
```bash
python3 scripts/generate_submission.py
```

### 6. Official Submission Validator
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### 7. Interactive Notebook
Launch Jupyter Notebook to inspect step-by-step visualizations and diagnostic cells:
```bash
jupyter notebook notebooks/entity_resolution_experiments.ipynb
```
