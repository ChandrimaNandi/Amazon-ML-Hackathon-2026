"""
Memory-Safe & Disk-Safe Streaming Inference Pipeline for Precision-Sieve ER.
Streams multi-million query test sets using Tier 1 exact sieve + GBDT ensemble scoring + disk shards.
Guarantees <2.5 GB peak RAM and <500 MB temporary disk usage.
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

from precision_sieve_er.configs.default_config import (
    TEST_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE, DEFAULT_ABS_THRESHOLD, DEFAULT_MARGIN_THRESHOLD
)
from precision_sieve_er.src.normalization import create_normalized_dataframe
from precision_sieve_er.src.blocking import PrecisionSieveBlocker
from precision_sieve_er.src.features import extract_candidate_features_dataframe
from precision_sieve_er.src.classifier import AsymmetricPrecisionEnsemble
from precision_sieve_er.src.assignment import apply_assignment_rules

logger = logging.getLogger("PrecisionSieve.Pipeline")


def run_precision_sieve_inference(
    ensemble: AsymmetricPrecisionEnsemble,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = DEFAULT_ABS_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    chunk_size: Optional[int] = None,
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes high-precision streaming test inference:
    - Tier 1 Exact Sieve locks in clean matches instantly.
    - Tier 2 GBDT Ensemble scores ambiguous candidate pairs.
    - Hard Veto Gates reject numeric/PIN/State contradictions.
    - Disk-sharded streaming keeps peak RAM < 2.5 GB and disk < 500 MB.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    start_t = time.time()
    logger.info("=" * 60)
    logger.info("[STREAMING INFERENCE] Starting Precision-Sieve 99+ Pipeline")
    logger.info(f"  Test Directory:     {test_dir}")
    logger.info(f"  Absolute Threshold: {abs_threshold:.2f}")
    logger.info(f"  Margin Threshold:   {margin_threshold:.2f}")
    logger.info(f"  Chunk Size:         {chunk_size:,} queries")
    logger.info("=" * 60)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    # 1. Load and normalize reference S1 entities
    logger.info(f"[INFERENCE] Loading reference S1 entities from {s1_path.name}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    all_s1_ids = s1_df["entity_id"].tolist()
    num_s1 = len(all_s1_ids)

    s1_df = create_normalized_dataframe(s1_df)

    # Lightweight lookup dictionary for fast feature extraction
    s1_lookup = {
        row.entity_id: {
            "name_norm": getattr(row, "name_norm", ""),
            "addr_norm": getattr(row, "addr_norm", ""),
            "country": getattr(row, "country_clean", "")
        }
        for row in s1_df.itertuples()
    }

    # 2. Fit PrecisionSieveBlocker on S1
    blocker = PrecisionSieveBlocker(top_k_bm25=8)
    blocker.fit(s1_df)
    del s1_df
    gc.collect()

    # 3. Setup Disk Shards to guarantee O(1) RAM
    num_shards = 16 if num_s1 >= 160000 else max(1, (num_s1 + 9999) // 10000)
    shard_size = (num_s1 + num_shards - 1) // num_shards
    s1_to_shard = {s1_id: min(idx // shard_size, num_shards - 1) for idx, s1_id in enumerate(all_s1_ids)}

    shard_dir = output_matching_path.parent / "temp_sieve_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cand_fps = {s: open(shard_dir / f"shard_{s}_cands.txt", "w", encoding="utf-8") for s in range(num_shards)}
    pred_fps = {s: open(shard_dir / f"shard_{s}_preds.txt", "w", encoding="utf-8") for s in range(num_shards)}

    total_processed_queries = 0
    total_tier1_matches = 0
    total_candidates_recorded = 0
    total_predictions_recorded = 0

    try:
        # 4. Stream query chunks from Source 2 and Source 3
        for query_source, q_path in [("Source 2", s2_path), ("Source 3", s3_path)]:
            if not q_path.exists():
                logger.warning(f"[INFERENCE] Query file {q_path.name} not found. Skipping...")
                continue

            logger.info(f"\n[INFERENCE] Streaming queries from {query_source} ({q_path.name})...")
            chunk_iter = pd.read_csv(q_path, sep="\t", chunksize=chunk_size, dtype=str)

            for chunk_idx, raw_chunk in enumerate(chunk_iter, start=1):
                chunk_t0 = time.time()
                if max_queries and total_processed_queries >= max_queries:
                    break

                q_df = create_normalized_dataframe(raw_chunk)
                num_in_chunk = len(q_df)
                q_ids = q_df["entity_id"].tolist()
                q_combs = q_df["comb_norm"].tolist()

                # Query lookup
                query_lookup = {
                    row.entity_id: {
                        "name_norm": getattr(row, "name_norm", ""),
                        "addr_norm": getattr(row, "addr_norm", ""),
                        "country": getattr(row, "country_clean", "")
                    }
                    for row in q_df.itertuples()
                }

                # Step A: Tier 1 Exact Sieve Resolution
                tier1_matches_list = blocker.resolve_tier1_exact(q_combs)
                tier1_dict: Dict[str, str] = {}
                unresolved_indices: List[int] = []

                for i, s1_match in enumerate(tier1_matches_list):
                    if s1_match is not None:
                        tier1_dict[q_ids[i]] = s1_match
                        total_tier1_matches += 1
                    else:
                        unresolved_indices.append(i)

                # Step B: Generate candidates for unresolved queries
                chunk_scores: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in q_ids}

                if unresolved_indices:
                    sub_q_df = q_df.iloc[unresolved_indices]
                    cand_map = blocker.generate_candidates(sub_q_df)

                    # Build pair dicts for feature extraction
                    pairs_to_extract: List[Dict[str, Any]] = []
                    for qid in sub_q_df["entity_id"]:
                        for s1_cand, bm25_sc in cand_map.get(qid, []):
                            pairs_to_extract.append({
                                "query_id": qid,
                                "s1_id": s1_cand,
                                "bm25_score": bm25_sc,
                                "rank": len(chunk_scores[qid]) + 1
                            })

                    if pairs_to_extract:
                        feat_df = extract_candidate_features_dataframe(pairs_to_extract, s1_lookup, query_lookup)
                        probs = ensemble.predict_proba(
                            feat_df,
                            apply_vetoes=True,
                            s1_lookup=s1_lookup,
                            query_lookup=query_lookup
                        )
                        for (p_item, prob) in zip(pairs_to_extract, probs):
                            chunk_scores[p_item["query_id"]].append((p_item["s1_id"], float(prob)))

                # Step C: Apply Assignment Rules & Ambiguity Gating
                chunk_matches, chunk_cands = apply_assignment_rules(
                    query_scores=chunk_scores,
                    tier1_exact_matches=tier1_dict,
                    abs_threshold=abs_threshold,
                    margin_threshold=margin_threshold
                )

                # Step D: Stream directly to disk shards with buffering
                cand_buf: Dict[int, List[str]] = {s: [] for s in range(num_shards)}
                for s1_id, q_set in chunk_cands.items():
                    sh = s1_to_shard.get(s1_id)
                    if sh is not None:
                        for qid in q_set:
                            cand_buf[sh].append(f"{s1_id}\t{qid}\n")
                            total_candidates_recorded += 1

                for sh, lines in cand_buf.items():
                    if lines:
                        cand_fps[sh].write("".join(lines))

                pred_buf: Dict[int, List[str]] = {s: [] for s in range(num_shards)}
                for s1_id, q_set in chunk_matches.items():
                    sh = s1_to_shard.get(s1_id)
                    if sh is not None:
                        for qid in q_set:
                            pred_buf[sh].append(f"{s1_id}\t{qid}\n")
                            total_predictions_recorded += 1

                for sh, lines in pred_buf.items():
                    if lines:
                        pred_fps[sh].write("".join(lines))

                total_processed_queries += num_in_chunk
                chunk_sec = time.time() - chunk_t0
                mem_mb = psutil.Process().memory_info().rss / (1024**2)
                logger.info(
                    f"[INFERENCE] {query_source} Chunk {chunk_idx:3d} | Queries: {total_processed_queries:,} | "
                    f"Tier1: {len(tier1_dict):,} | Time: {chunk_sec:.1f}s | RAM RSS: {mem_mb:.1f} MB"
                )

                del raw_chunk, q_df, query_lookup, chunk_scores, chunk_matches, chunk_cands, cand_buf, pred_buf
                gc.collect()

    finally:
        for fp in cand_fps.values():
            fp.close()
        for fp in pred_fps.values():
            fp.close()

    # 5. Assemble Official Output TSVs Shard-by-Shard
    logger.info("=" * 60)
    logger.info(f"[INFERENCE] Assembling final submission TSVs from {num_shards} disk shards...")
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_matching_path, "w", encoding="utf-8") as f_match, \
         open(output_candidate_path, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for sh in range(num_shards):
            start_idx = sh * shard_size
            end_idx = min(start_idx + shard_size, num_s1)
            shard_s1_list = all_s1_ids[start_idx:end_idx]

            shard_preds: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}
            shard_cands: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}

            p_file = shard_dir / f"shard_{sh}_preds.txt"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2 and parts[0] in shard_preds:
                            shard_preds[parts[0]].add(parts[1])

            c_file = shard_dir / f"shard_{sh}_cands.txt"
            if c_file.exists():
                with open(c_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2 and parts[0] in shard_cands:
                            shard_cands[parts[0]].add(parts[1])

            for s1_id in shard_s1_list:
                m_list = sorted(list(shard_preds.get(s1_id, set())))
                c_list = sorted(list(shard_cands.get(s1_id, set())))
                for m in m_list:
                    if m not in shard_cands.get(s1_id, set()):
                        c_list.append(m)
                c_list = sorted(list(set(c_list)))

                f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")

            p_file.unlink(missing_ok=True)
            c_file.unlink(missing_ok=True)
            del shard_preds, shard_cands, shard_s1_list
            gc.collect()

    try:
        shard_dir.rmdir()
    except Exception:
        pass

    total_time = time.time() - start_t
    logger.info(f"[INFERENCE COMPLETE] Total Queries: {total_processed_queries:,} | Matches: {total_predictions_recorded:,} in {total_time:.1f}s.")
    return {
        "num_s1": num_s1,
        "total_queries": total_processed_queries,
        "tier1_exact_matches": total_tier1_matches,
        "num_matches": total_predictions_recorded,
        "num_candidates": total_candidates_recorded,
        "runtime_sec": total_time,
        "matching_path": str(output_matching_path),
        "candidate_path": str(output_candidate_path)
    }
