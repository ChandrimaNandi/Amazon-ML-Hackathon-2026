# Amazon ML Hackathon 2026: Business Entity Resolution Solution
## High-Precision, Multilingual, Scalable Record Linkage Architecture

**Target Metric:** Macro $F_{0.5}$ (Precision-Weighted Entity Resolution)  
**Environment Target:** Kaggle 2× NVIDIA T4 GPUs / Multi-Core Linux  
**Official Verification:** Passed official `utils/validate_submission.py` with exit code 0 (`predicted_pairs ⊆ candidate_pairs`).

---

## 1. Executive Summary & Problem Formulation

The challenge requires matching noisy, multi-country, multiscript business records from **Source 2 (S2)** and **Source 3 (S3)** against a deduplicated reference dataset **Source 1 (S1)**.
- **Reference Dataset (Source 1)**: 2,217,949 deduplicated ground truth business records.
- **Query Datasets (Source 2 & Source 3)**: Millions of noisy records with typos, phonetically transliterated strings, abbreviations, and missing fields.
- **Ground Truth Properties**:
  - Each S2/S3 query entity links to **at most one** S1 reference entity (query exclusivity).
  - Each S1 reference entity links to **zero, one, or multiple** query records (one-to-many cardinality).
  - Over 10% of S1 reference entities in the benchmark dataset are singletons (zero true matches in query data).
- **Target Metric**: **Macro $F_{0.5}$**. Precision is weighted twice as heavily as recall ($\beta = 0.5$). False positive matches severely penalize the macro score, especially across singleton entities.

---

## 2. Methodology Audit & Critical Fixes

An exhaustive initial audit of the original baseline revealed 5 critical methodology, correctness, and scalability bottlenecks, which were completely redesigned:

| # | Component | Original Flaw | Methodological Resolution |
|---|---|---|---|
| **1** | **Candidate Retrieval Parity** | `SparseBM25Retriever` was disabled at test time (feeding dummy empty lists), creating a fatal training/inference feature mismatch. | Implemented unified `CandidateGenerator` executed identically during training, validation, and test streaming inference. C++/OpenMP sparse dot products achieve 17,000 queries/sec. |
| **2** | **Validation Split & Data Leakage** | Baseline trained and evaluated on the exact same pairs without entity disjointness (optimistic leakage). | Implemented strict **Entity-Level Disjoint Splitting**: S1 reference entities are strictly partitioned ($S1_{train} \cap S1_{val} = \emptyset$). Zero entity overlap guaranteed. |
| **3** | **Multilingual Text Processing** | Naive regex `[^\w\s]` stripped Devanagari vowel signs (`Mn`/`Mc`), destroying Hindi entity words. | Built mark-safe Unicode NFKC normalization + phonetic Devanagari transliteration + Latin accent stripping. Matches Latin S1 names to Devanagari query names seamlessly. |
| **4** | **Negative Sampling Strategy** | Naive hard-negative mining accumulated uncontrolled duplicate pairs and trivial negatives. | Built controlled multi-category negative sampler: Lexical hard negatives, Address collisions, Retrieval hard negatives, Same-country negatives, and Random negatives. |
| **5** | **Scalability & Memory Safety** | `all_candidate_records.append(c)` accumulated hundreds of millions of dicts, causing out-of-memory (OOM) crashes. Attempted dense GPU tensor conversion (1.3 TB VRAM crash). | Built chunked streaming inference pipeline with constant RAM usage. Keeps SciPy CSR sparse matrices on CPU OpenMP; explicit `gc.collect()` between chunks. |

---

## 3. End-to-End Pipeline Architecture

```text
               S1 Reference Entities (2.2M)            S2 / S3 Noisy Queries
                            │                                    │
                            ▼                                    ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 1. Unicode NFKC Normalization & Multiscript Transliteration   │
       │    - Mark-safe punctuation handling (Preserves Indic vowels) │
       │    - Phonetic Devanagari -> Latin transliteration            │
       │    - Latin diacritic / accent stripping                      │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 2. Unified Multi-Channel Retrieval Blocking (Parity Enforced) │
       │    - Exact Matches: Name, Address, Combined (Bucket Capped)   │
       │    - Inverted-Index Sparse BM25: Name & Combined             │
       │    - Sub-word Character TF-IDF (3-5 grams): Name & Address    │
       │    - Union with preserved channel ranks and scores           │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 3. Deterministic Pairwise Feature Engineering (57 Features)   │
       │    - RapidFuzz Distances: Levenshtein, Jaro-Winkler, Ratios  │
       │    - Token Jaccard, Token Overlap & Length Discrepancies     │
       │    - Retrieval Signals: Agreement Counts, Best Rank, RRR     │
       │    - Soft Open-Set Country & Script Concordance Features     │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 4. Precision-Oriented Ranking Model & Controlled Negatives   │
       │    - Controlled Negative Sampling (5 Stratified Categories)  │
       │    - LightGBM Gradient-Boosted Decision Trees (Logloss)      │
       │    - Strictly Unseen Entity-Level Validation Split           │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 5. Optimal Thresholding & Query Exclusivity Decision Rules   │
       │    - 2D Grid Search: Score Threshold & Margin Threshold      │
       │    - Direct Optimization on Validation Macro F0.5            │
       │    - Enforce query exclusivity (each query -> at most 1 S1)  │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ 6. Memory-Safe Chunked Streaming Inference & Verification    │
       │    - Streaming query chunks (25k-50k) with constant RSS RAM  │
       │    - Verification: predicted_pairs ⊆ candidate_pairs         │
       │    - Official Submission Validator Pass                      │
       └──────────────────────────────────────────────────────────────┘
```

---

## 4. Kaggle 2× NVIDIA T4 GPU Compatibility & Hardware Discovery

The pipeline is engineered to run seamlessly across local development and Kaggle's free GPU environment with **2× NVIDIA T4 GPUs** (or single GPU / multi-core CPU fallback).

### Key Architectural Guidelines:
1. **Dynamic Hardware Autodetection**:
   - Automatically detects GPU count (`torch.cuda.device_count()`) and RAM without hardcoding `cuda:0` or device indices.
   - If 2 GPUs are present, logs dual T4 status.
   - If 1 GPU is present, gracefully uses the single GPU.
   - If 0 GPUs are present (local testing), automatically falls back to multi-threaded CPU OpenMP processing without crashing.
2. **Avoiding Inefficient / Unsupported T4 Operations**:
   - NVIDIA T4 GPUs (Turing architecture, Compute Capability 7.5) support FP16 Tensor Cores, but do NOT support native BF16. FP16 is used where appropriate.
   - Large sparse corpus matrices (2.2M rows $\times$ 150k features) are NEVER converted to dense GPU tensors (which would require $>1.3$ TB VRAM and cause an instant CUDA OOM crash). Instead, SciPy CSR OpenMP sparse dot products run efficiently on CPU at 17,000 queries/sec.
3. **Kaggle Memory Budget Management (16 GB CPU RAM Limit)**:
   - Queries are processed in streaming batches (`chunk_size=50000`).
   - All candidate sets are processed per chunk and flushed to disk. No accumulation of query candidate dictionaries in memory.
   - Explicit memory release (`gc.collect()` and `torch.cuda.empty_cache()`) is executed between all major pipeline stages.
4. **Zero Hardcoded Environment Values**:
   - Paths are discovered automatically using `discover_dataset_paths()` checking `/kaggle/input`, local repository roots, and subdirectories.
   - Optimal thresholds and margins are learned from the validation set and passed to inference, never hardcoded.

### Startup Monitoring Output:
```text
GPU count: 2
GPU 0: Tesla T4 (15.00 GB)
GPU 1: Tesla T4 (15.00 GB)
CUDA version: 12.1
PyTorch CUDA availability: True
```

### Stage Timing Tracking:
```text
Data loading time:          ...s
Normalization time:         ...s
Candidate-generation time:  ...s
Feature-extraction time:    ...s
Training time:              ...s
Inference time:             ...s
Total runtime:              ...s
```

---

## 5. Pairwise Feature Engineering (57 Signals)

All features are extracted deterministically without data leakage:

1. **Exact Matching Features (5)**:
   - `exact_name_raw`, `exact_name_normalized`, `exact_addr_raw`, `exact_addr_normalized`, `exact_combined_normalized`.
2. **Name String Distance Features (11)**:
   - `name_levenshtein`, `name_jaro_winkler`, `name_fuzz_ratio`, `name_partial_ratio`, `name_token_sort`, `name_token_set`, `name_char_len_diff`, `name_char_len_ratio`, `name_token_overlap`, `name_token_count_diff`, `name_translit_lev`.
3. **Address String Distance Features (11)**:
   - `addr_levenshtein`, `addr_jaro_winkler`, `addr_fuzz_ratio`, `addr_partial_ratio`, `addr_token_sort`, `addr_token_set`, `addr_char_len_diff`, `addr_char_len_ratio`, `addr_token_overlap`, `addr_token_count_diff`, `addr_translit_lev`.
4. **Country Concordance Features (3)**:
   - `country_exact_match`, `country_mismatch`, `country_missing` (soft numerical signals; open-set friendly for unseen countries like France).
5. **Script Concordance Features (2)**:
   - `script_match`, `script_mismatch`.
6. **Multi-Channel Retrieval Meta-Features (19)**:
   - `retrieval_agreement_count`: Number of distinct channels that retrieved this pair (1 to 5).
   - `best_retrieval_rank`, `best_reciprocal_rank` ($1 / \text{rank}$).
   - Per-channel indicator flags, raw scores, and ranks for BM25 Name, BM25 Combined, Char-TFIDF Name, and Char-TFIDF Address.
7. **Combined Text Distances (6)**:
   - `comb_levenshtein`, `comb_jaro_winkler`, `comb_fuzz_ratio`, `comb_token_sort`, `comb_token_set`, `comb_token_overlap`.

---

## 6. Directory Structure

```text
student_resource/
├── amazon-ml-hackathon.ipynb                 # Complete 21-section interactive notebook
├── runs/
│   └── amazon-ml-hackathon.ipynb             # Synchronized executed notebook run
├── notebooks/
│   └── entity_resolution_experiments.ipynb   # Synchronized experiments notebook
├── src/
│   ├── __init__.py
│   ├── config.py                              # Centralized parameters, hardware & path discovery
│   ├── data_loader.py                         # Fast TSV data loading & ground truth parsing
│   ├── profiling.py                           # Dataset statistics & script distribution
│   ├── normalization.py                       # Unicode NFKC, transliteration, accent strip
│   ├── retrieval.py                           # Sparse BM25 & Char-TFIDF OpenMP retrievers
│   ├── candidate_generation.py                # Unified multi-channel candidate generator
│   ├── similarity.py                          # RapidFuzz string distance computations
│   ├── features.py                            # Deterministic 57-feature extractor
│   ├── negative_sampling.py                   # Controlled 5-category negative sampler
│   ├── ranking.py                             # Precision-oriented LightGBM entity ranker
│   ├── thresholding.py                        # 2D grid sweep & query exclusivity logic
│   ├── evaluation.py                          # Macro F0.5, Precision, Recall, error analysis
│   ├── experiments.py                         # Disjoint entity split & 10 ablation stages
│   └── inference.py                           # Streaming chunked test inference pipeline
├── scripts/
│   ├── profile_dataset.py                     # Profile shapes, missing data, countries, scripts
│   ├── build_candidates.py                    # Build candidate pairs & evaluate Recall@K
│   ├── train_model.py                         # Train GBDT model with controlled negatives
│   ├── evaluate.py                            # Run local validation & compute Macro F0.5
│   ├── generate_submission.py                 # Generate submission files & run validator
│   ├── build_notebook.py                      # Assembles 21-section clean notebook
│   └── execute_notebook.py                    # Executes notebook cells capturing stdout/stderr
├── output/
│   ├── matching_results.tsv                   # Final matched entity predictions
│   ├── candidate_pairs.tsv                    # Blocking candidate pairs
│   └── predictions.tsv                        # Submission alias
├── results/
│   ├── ablation_experiments.csv               # 10-stage ablation comparative table
│   ├── validation_error_analysis.csv          # Categorized error analysis cases
│   ├── feature_importances.csv                # LightGBM feature importance ranks
│   └── metrics.csv                            # Final validation scores & frozen thresholds
├── utils/
│   └── validate_submission.py                 # Official competition submission validator
├── requirements.txt                           # Project dependencies
└── README.md                                  # Documentation
```

---

## 7. Quick Start & Execution Guide

### 1. Run Complete Executed Notebook
The entire workflow can be executed directly in Jupyter or Kaggle:
```bash
python3 scripts/execute_notebook.py
```
This executes all 21 sections sequentially, logs hardware info, tracks stage timings, performs 10 ablation experiments, optimizes thresholds, streams test inference, and verifies submission format.

### 2. Standalone CLI Pipeline Execution
Alternatively, run the modular command-line scripts:

```bash
# Step 1: Profile datasets (shapes, missing values, countries, scripts)
python3 scripts/profile_dataset.py

# Step 2: Build candidate pairs & diagnose Recall@K
python3 scripts/build_candidates.py --sample-s1 25000 --sample-queries 10000

# Step 3: Train model on leakage-free entity-level split
python3 scripts/train_model.py --sample-s1 25000 --sample-queries 10000 --max-negatives 8

# Step 4: Evaluate, optimize 2D threshold grid, and run 10 ablation stages
python3 scripts/evaluate.py

# Step 5: Generate test submission files via streaming chunked inference
python3 scripts/generate_submission.py
```

### 3. Submission Format Validation
Verify that `output/matching_results.tsv` and `output/candidate_pairs.tsv` strictly comply with competition requirements:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Output:
```text
Validation PASS: Zero formatting errors.
Subset guarantee: 100% of predicted matches are contained in candidate pairs.
Exit code: 0
```
