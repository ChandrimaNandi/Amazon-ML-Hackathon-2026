"""
Scalable, Streaming Inference Pipeline Module for Business Entity Resolution.

Processes multi-million query test datasets (S2 and S3) in memory-safe chunks:
- Uses CandidateGenerator with all retrieval channels (Exact, BM25, CharTFIDF)
- Strict pipeline parity between training, validation, and test inference
- Extracts deterministic features and applies frozen validation thresholds
- Eliminates RAM candidate accumulation via disk-sharded streaming
- Guarantees: predicted_pairs ⊆ candidate_pairs
- Writes official tab-separated files:
  * output/matching_results.tsv
  * output/candidate_pairs.tsv
"""

import time
import os
import gc
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Any, Optional, List
import logging

from src.config import (
    TEST_DIR, TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE, load_threshold_config, release_memory
)
from src.data_loader import load_source_tsv
from src.normalization import create_normalized_features
from src.candidate_generation import CandidateGenerator
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel
from src.thresholding import apply_decision_rules

logger = logging.getLogger(__name__)


def run_chunked_inference(
    model: EntityMatcherModel,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    chunk_size: Optional[int] = None,
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes streaming chunked inference on test dataset with bounded memory footprint.
    Uses disk-sharded streaming to avoid accumulating hundreds of millions of candidates in RAM.
    
    Args:
        model: Fitted EntityMatcherModel.
        test_dir: Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv.
        output_matching_path: Output path for final predictions TSV.
        output_candidate_path: Output path for blocking candidates TSV.
        abs_threshold: Frozen optimal absolute score threshold from validation (loaded if None).
        margin_threshold: Frozen optimal margin threshold from validation (loaded if None).
        chunk_size: Number of queries per chunk to stream without exceeding RAM.
        max_queries: Optional limit for testing/debugging.
        
    Returns:
        Summary dictionary of inference metrics.
    """
    # 0. Resolve Thresholds & Chunk Size Dynamically
    if abs_threshold is None or margin_threshold is None:
        thresh_cfg = load_threshold_config()
        if abs_threshold is None:
            abs_threshold = float(thresh_cfg["abs_threshold"])
        if margin_threshold is None:
            margin_threshold = float(thresh_cfg["margin_threshold"])

    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    logger.info("=" * 60)
    logger.info("[INFERENCE] STARTING STREAMING DISK-SHARDED INFERENCE PIPELINE")
    logger.info(f"  Test Directory:     {test_dir}")
    logger.info(f"  Absolute Threshold: {abs_threshold:.2f}")
    logger.info(f"  Margin Threshold:   {margin_threshold:.2f}")
    logger.info(f"  Chunk Size:         {chunk_size:,} queries")
    logger.info(f"  Max Queries:        {max_queries if max_queries else 'ALL'}")
    logger.info("=" * 60)
    
    start_t = time.time()
    process = psutil.Process(os.getpid())
    
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    
    # 1. Load and normalize reference S1 entities
    logger.info(f"[INFERENCE] Loading reference S1 entities from {s1_path.name}...")
    s1_df = load_source_tsv(s1_path)
    all_s1_ids = s1_df["entity_id"].tolist()
    num_s1 = len(all_s1_ids)
    
    logger.info("[INFERENCE] Applying Unicode NFKC & transliteration normalization to S1...")
    s1_df = create_normalized_features(s1_df)
    
    # 2. Fit unified CandidateGenerator once on reference S1
    generator = CandidateGenerator(
        k_exact_cap=30,
        k_bm25_name=15,
        k_bm25_comb=15,
        k_tfidf_name=15,
        k_tfidf_addr=10
    )
    generator.fit(s1_df)
    
    # 3. Setup Disk-Sharded Candidate & Prediction Streaming
    # Shard reference entities to guarantee O(1) RAM regardless of query count
    num_shards = 16 if num_s1 >= 160000 else max(1, (num_s1 + 9999) // 10000)
    shard_size = (num_s1 + num_shards - 1) // num_shards
    s1_to_shard = {s1_id: min(idx // shard_size, num_shards - 1) for idx, s1_id in enumerate(all_s1_ids)}
    
    shard_dir = output_matching_path.parent / "temp_inference_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    cand_fps = {s: open(shard_dir / f"shard_{s}_cands.txt", "w", encoding="utf-8") for s in range(num_shards)}
    pred_fps = {s: open(shard_dir / f"shard_{s}_preds.txt", "w", encoding="utf-8") for s in range(num_shards)}
    
    total_queries_processed = 0
    total_candidates_generated = 0
    
    # 4. Stream queries from Source 2 and Source 3 in chunks
    query_sources = [("Source 2", s2_path), ("Source 3", s3_path)]
    
    try:
        for src_name, src_path in query_sources:
            if not src_path.exists():
                logger.warning(f"[INFERENCE] Query file {src_path} not found, skipping.")
                continue
                
            logger.info(f"\n[INFERENCE] Streaming query chunks from {src_name} ({src_path.name})...")
            
            chunk_iter = pd.read_csv(
                src_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
                chunksize=chunk_size
            )
            
            for chunk_idx, raw_chunk in enumerate(chunk_iter, start=1):
                if max_queries and total_queries_processed >= max_queries:
                    logger.info(f"[INFERENCE] Reached max_queries limit ({max_queries:,}). Stopping query stream.")
                    break
                    
                # Clean fields
                raw_chunk["entity_id"] = raw_chunk["entity_id"].astype(str).str.strip()
                raw_chunk["business_name"] = raw_chunk["business_name"].astype(str).str.strip()
                raw_chunk["business_address"] = raw_chunk["business_address"].astype(str).str.strip()
                raw_chunk["country"] = raw_chunk["country"].astype(str).str.strip()
                
                # Normalize query chunk
                q_chunk = create_normalized_features(raw_chunk)
                n_queries_in_chunk = len(q_chunk)
                
                # A. Generate candidates using exact same multi-channel pipeline
                cand_df, _ = generator.generate_candidates(q_chunk)
                total_candidates_generated += len(cand_df)
                
                if not cand_df.empty:
                    # B. Extract features
                    feat_df = extract_candidate_features(cand_df, s1_df, q_chunk)
                    
                    # C. Score candidates with model
                    feat_df["pred_score"] = model.predict_proba(feat_df)
                    
                    # D. Apply frozen decision rules
                    chunk_preds = apply_decision_rules(
                        cand_df_with_probs=feat_df,
                        abs_threshold=abs_threshold,
                        margin_threshold=margin_threshold,
                        enforce_query_exclusivity=True
                    )
                    
                    # E. Stream candidates directly to disk shards with bulk buffering
                    shard_cand_buffers: Dict[int, List[str]] = {s: [] for s in range(num_shards)}
                    for s1_id, q_id in zip(cand_df["s1_id"], cand_df["query_id"]):
                        sh = s1_to_shard.get(s1_id)
                        if sh is not None:
                            shard_cand_buffers[sh].append(f"{s1_id}\t{q_id}\n")
                            
                    for sh, lines in shard_cand_buffers.items():
                        if lines:
                            cand_fps[sh].write("".join(lines))
                            
                    # F. Stream predictions directly to disk shards with bulk buffering
                    shard_pred_buffers: Dict[int, List[str]] = {s: [] for s in range(num_shards)}
                    for s1_id, q_ids in chunk_preds.items():
                        sh = s1_to_shard.get(s1_id)
                        if sh is not None:
                            for q_id in q_ids:
                                shard_pred_buffers[sh].append(f"{s1_id}\t{q_id}\n")
                                
                    for sh, lines in shard_pred_buffers.items():
                        if lines:
                            pred_fps[sh].write("".join(lines))
                                
                    del feat_df, chunk_preds, shard_cand_buffers, shard_pred_buffers
                    
                total_queries_processed += n_queries_in_chunk
                rss_mb = process.memory_info().rss / (1024 ** 2)
                logger.info(
                    f"[INFERENCE] {src_name} Chunk {chunk_idx}: Processed {n_queries_in_chunk:,} queries "
                    f"(Total: {total_queries_processed:,}) | Memory RSS: {rss_mb:.1f} MB"
                )
                
                del q_chunk, cand_df, raw_chunk
                release_memory()
                
                if max_queries and total_queries_processed >= max_queries:
                    break
                    
    finally:
        # Close all shard file handles safely
        for fp in cand_fps.values():
            fp.close()
        for fp in pred_fps.values():
            fp.close()

    # 5. Assemble Official Output TSVs Shard-by-Shard (Guarantees Low Peak RAM & Strict Subset)
    logger.info("\n[INFERENCE] Assembling submission files from disk shards (strictly preserving subset integrity)...")
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pred_alias = output_matching_path.parent / "predictions.tsv"
    
    total_predictions_made = 0
    total_candidates_recorded = 0
    n_zero = 0
    n_single = 0
    n_multi = 0
    
    with open(output_candidate_path, "w", encoding="utf-8") as f_cand, \
         open(output_matching_path, "w", encoding="utf-8") as f_match, \
         open(pred_alias, "w", encoding="utf-8") as f_alias:
        
        # Write exact required headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_alias.write("source1_entity_id\tmatched_entity_ids\n")
        
        for sh in range(num_shards):
            start_idx = sh * shard_size
            end_idx = min(start_idx + shard_size, num_s1)
            shard_s1_list = all_s1_ids[start_idx:end_idx]
            
            # Temporary in-memory structures for current shard only
            shard_cands: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}
            shard_preds: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}
            
            # Read shard candidate file
            cand_shard_file = shard_dir / f"shard_{sh}_cands.txt"
            if cand_shard_file.exists():
                with open(cand_shard_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            s1, q = parts
                            if s1 in shard_cands:
                                shard_cands[s1].add(q)
                try:
                    cand_shard_file.unlink()
                except OSError:
                    pass
                    
            # Read shard prediction file
            pred_shard_file = shard_dir / f"shard_{sh}_preds.txt"
            if pred_shard_file.exists():
                with open(pred_shard_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            s1, q = parts
                            if s1 in shard_preds:
                                shard_preds[s1].add(q)
                try:
                    pred_shard_file.unlink()
                except OSError:
                    pass
                    
            # Write out rows for this shard, enforcing strict subset
            for s1_id in shard_s1_list:
                c_set = shard_cands[s1_id]
                # Guaranteed: predicted_pairs ⊆ candidate_pairs
                p_set = shard_preds[s1_id].intersection(c_set)
                
                total_candidates_recorded += len(c_set)
                total_predictions_made += len(p_set)
                
                if len(p_set) == 0:
                    n_zero += 1
                elif len(p_set) == 1:
                    n_single += 1
                else:
                    n_multi += 1
                    
                cand_str = ",".join(sorted(c_set))
                pred_str = ",".join(sorted(p_set))
                
                f_cand.write(f"{s1_id}\t{cand_str}\n")
                f_match.write(f"{s1_id}\t{pred_str}\n")
                f_alias.write(f"{s1_id}\t{pred_str}\n")
                
            del shard_cands, shard_preds, shard_s1_list
            release_memory()
            
    # Clean up temporary shard directory
    try:
        shard_dir.rmdir()
    except OSError:
        pass
        
    elapsed = time.time() - start_t
    peak_rss_mb = process.memory_info().rss / (1024 ** 2)
    
    summary = {
        "num_test_s1_entities": num_s1,
        "total_queries_processed": total_queries_processed,
        "total_candidates_generated": total_candidates_generated,
        "total_candidates_recorded": total_candidates_recorded,
        "avg_candidates_per_query": round(total_candidates_generated / max(total_queries_processed, 1), 2),
        "total_predicted_matches": total_predictions_made,
        "s1_zero_match_count": n_zero,
        "s1_single_match_count": n_single,
        "s1_multi_match_count": n_multi,
        "peak_memory_mb": round(peak_rss_mb, 1),
        "elapsed_seconds": round(elapsed, 2),
        "chunk_size": chunk_size
    }
    
    logger.info("=" * 60)
    logger.info(f"[INFERENCE COMPLETED in {elapsed:.2f}s]")
    logger.info(f"  Test S1 Entities:     {num_s1:,}")
    logger.info(f"  Queries Processed:    {total_queries_processed:,}")
    logger.info(f"  Predicted Matches:    {total_predictions_made:,}")
    logger.info(f"  Zero Matches:         {n_zero:,}")
    logger.info(f"  Single Matches:       {n_single:,}")
    logger.info(f"  Multi Matches:        {n_multi:,}")
    logger.info(f"  Peak Memory RSS:      {peak_rss_mb:.1f} MB")
    logger.info(f"  Subset Compliance:    100% verified (predicted_pairs ⊆ candidate_pairs)")
    logger.info("=" * 60)
    
    return summary
