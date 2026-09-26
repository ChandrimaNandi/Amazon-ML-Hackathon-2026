"""
Memory-Safe Chunked Streaming Inference Pipeline for Fast Hybrid ER.
Streams multi-million query test datasets with constant RSS memory footprint,
producing official TSV outputs and guaranteeing predicted_pairs ⊆ candidate_pairs.
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

from fast_hybrid_er.configs.default_config import (
    TEST_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE
)
from fast_hybrid_er.src.normalization import create_normalized_features
from fast_hybrid_er.src.lsh_blocking import MultiChannelBlocker
from fast_hybrid_er.src.features import HybridFeatureExtractor
from fast_hybrid_er.src.classifier import AsymmetricEntityRanker
from fast_hybrid_er.src.assignment import apply_query_exclusivity

logger = logging.getLogger("FastHybridER.Pipeline")


def load_tsv_chunks(file_path: Path, chunk_size: int):
    """Generator yielding DataFrame chunks from large TSV file."""
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size, dtype=str):
        yield chunk


def run_streaming_inference(
    model: AsymmetricEntityRanker,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05,
    chunk_size: Optional[int] = None,
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs memory-safe streaming inference on test datasets (S2 and S3) against reference S1.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    start_t = time.time()
    logger.info("=" * 60)
    logger.info("[STREAMING INFERENCE] Starting Fast Hybrid Neural-Phonetic Pipeline")
    logger.info(f"  Test Directory:     {test_dir}")
    logger.info(f"  Absolute Threshold: {abs_threshold:.2f}")
    logger.info(f"  Margin Threshold:   {margin_threshold:.2f}")
    logger.info(f"  Chunk Size:         {chunk_size:,}")
    logger.info("=" * 60)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    # 1. Load and normalize reference S1 entities
    logger.info(f"[INFERENCE] Loading reference S1 entities from {s1_path.name}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    all_s1_ids = s1_df["entity_id"].tolist()
    num_s1 = len(all_s1_ids)

    logger.info("[INFERENCE] Normalizing S1 reference text...")
    s1_df = create_normalized_features(s1_df)

    # Fast dictionary lookup for feature extraction
    s1_lookup = {
        row.entity_id: {
            "business_name": getattr(row, "business_name", ""),
            "name_normalized": getattr(row, "name_normalized", ""),
            "name_transliterated": getattr(row, "name_transliterated", ""),
            "address_normalized": getattr(row, "address_normalized", ""),
            "address_transliterated": getattr(row, "address_transliterated", ""),
            "combined_normalized": getattr(row, "combined_normalized", ""),
            "combined_transliterated": getattr(row, "combined_transliterated", ""),
            "country": getattr(row, "country", "")
        }
        for row in s1_df.itertuples()
    }

    # 2. Fit MultiChannelBlocker once on S1
    blocker = MultiChannelBlocker()
    blocker.fit(s1_df)

    # Feature extractor with Dual-T4 GPU scorer
    feature_extractor = HybridFeatureExtractor()

    # Accumulator maps: s1_id -> set of query_ids
    # Initialized for all S1 entities to guarantee every S1 has an output row
    all_candidates_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    all_matches_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}

    total_queries_processed = 0
    total_candidates_generated = 0
    total_matches_predicted = 0

    # 3. Stream queries from Source 2 and Source 3
    for q_file in [s2_path, s3_path]:
        if not q_file.exists():
            logger.warning(f"File {q_file} not found, skipping.")
            continue

        logger.info(f"[INFERENCE] Streaming queries from {q_file.name}...")
        for chunk_df in load_tsv_chunks(q_file, chunk_size=chunk_size):
            if max_queries and total_queries_processed >= max_queries:
                break

            if max_queries and (total_queries_processed + len(chunk_df) > max_queries):
                chunk_df = chunk_df.iloc[:max_queries - total_queries_processed]

            # Normalize chunk
            chunk_df = create_normalized_features(chunk_df)
            q_lookup = {
                row.entity_id: {
                    "business_name": getattr(row, "business_name", ""),
                    "name_normalized": getattr(row, "name_normalized", ""),
                    "name_transliterated": getattr(row, "name_transliterated", ""),
                    "address_normalized": getattr(row, "address_normalized", ""),
                    "address_transliterated": getattr(row, "address_transliterated", ""),
                    "combined_normalized": getattr(row, "combined_normalized", ""),
                    "combined_transliterated": getattr(row, "combined_transliterated", ""),
                    "country": getattr(row, "country", "")
                }
                for row in chunk_df.itertuples()
            }

            # Retrieve candidates (LSH + Phonetic + Exact)
            cands = blocker.retrieve_candidates(chunk_df, top_k=15)
            total_candidates_generated += len(cands)

            # Record candidates for S1
            for c in cands:
                s1_id = c["s1_id"]
                if s1_id in all_candidates_map:
                    all_candidates_map[s1_id].add(c["query_id"])

            if len(cands) > 0:
                # Extract features
                feat_df = feature_extractor.extract_features(cands, s1_lookup, q_lookup)
                # Score with model
                scores = model.predict_proba(feat_df)
                feat_df["score"] = scores

                # Apply query exclusivity
                assignments = apply_query_exclusivity(
                    feat_df[["query_id", "s1_id", "score"]],
                    abs_threshold=abs_threshold,
                    margin_threshold=margin_threshold
                )

                # Assign matches
                for qid, s1_id in assignments.items():
                    if s1_id in all_matches_map:
                        all_matches_map[s1_id].add(qid)
                        # Ensure candidate subset invariant: predicted match must be in candidates
                        all_candidates_map[s1_id].add(qid)
                        total_matches_predicted += 1

            total_queries_processed += len(chunk_df)
            gc.collect()

        logger.info(f"[INFERENCE] Finished processing {q_file.name}. Total queries so far: {total_queries_processed:,}")

    # 4. Write Official TSV outputs
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[INFERENCE] Exporting {output_matching_path.name}...")
    with open(output_matching_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = sorted(list(all_matches_map[s1_id]))
            match_str = ",".join(matches)
            f_match.write(f"{s1_id}\t{match_str}\n")

    logger.info(f"[INFERENCE] Exporting {output_candidate_path.name}...")
    with open(output_candidate_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = sorted(list(all_candidates_map[s1_id]))
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    elapsed = time.time() - start_t
    rate = total_queries_processed / elapsed if elapsed > 0 else 0.0

    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "total_queries_processed": total_queries_processed,
        "total_candidates_generated": total_candidates_generated,
        "total_matches_predicted": total_matches_predicted,
        "queries_per_second": round(rate, 2),
        "matching_tsv": str(output_matching_path),
        "candidate_tsv": str(output_candidate_path),
    }

    logger.info("=" * 60)
    logger.info("[STREAMING INFERENCE] Inference Completed Successfully!")
    logger.info(f"  Elapsed Time:     {elapsed:.2f}s ({rate:.1f} queries/sec)")
    logger.info(f"  Total Matches:    {total_matches_predicted:,}")
    logger.info(f"  Matching Output:  {output_matching_path}")
    logger.info(f"  Candidate Output: {output_candidate_path}")
    logger.info("=" * 60)

    return summary
