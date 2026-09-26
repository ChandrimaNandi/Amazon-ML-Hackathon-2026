"""
Memory-Safe Chunked Streaming Inference Pipeline for ColBERT-Ditto ER.
Processes multi-million query test sets within a bounded <4.5 GB RAM footprint,
evaluating ColBERT Late-Interaction MaxSim on GPU and generating official submission TSVs.
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from colbert_ditto_er.configs.default_config import (
    TEST_DIR, SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH,
    DEFAULT_CHUNK_SIZE, DEFAULT_MAX_SEQ_LEN
)
from colbert_ditto_er.src.normalization import create_normalized_dataframe
from colbert_ditto_er.src.serialization import serialize_dataframe, serialize_record
from colbert_ditto_er.src.blocking import SparseBlockingEngine
from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder, compute_maxsim_score
from colbert_ditto_er.src.assignment import apply_assignment_rules

logger = logging.getLogger("ColBERT_Ditto.Pipeline")


def run_streaming_colbert_inference(
    model: ColBERTTokenEncoder,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = 0.45,
    margin_threshold: float = 0.05,
    chunk_size: Optional[int] = None,
    max_queries: Optional[int] = None,
    device: Optional["torch.device"] = None
) -> Dict[str, Any]:
    """
    Executes memory-safe streaming inference on test datasets (Source 2 and Source 3)
    against reference Source 1 entities using ColBERT late-interaction scoring.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if device is None:
        device = torch.device("cuda:0" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")

    start_t = time.time()
    logger.info("=" * 60)
    logger.info("[STREAMING INFERENCE] Starting ColBERT-Ditto Late-Interaction Pipeline")
    logger.info(f"  Test Directory:     {test_dir}")
    logger.info(f"  Absolute Threshold: {abs_threshold:.2f}")
    logger.info(f"  Margin Threshold:   {margin_threshold:.2f}")
    logger.info(f"  Chunk Size:         {chunk_size:,}")
    logger.info(f"  Device:             {device}")
    logger.info("=" * 60)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    # 1. Load reference S1 entities
    logger.info(f"[INFERENCE] Loading reference S1 entities from {s1_path.name}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    all_s1_ids = s1_df["entity_id"].tolist()
    num_s1 = len(all_s1_ids)
    logger.info(f"[INFERENCE] Loaded {num_s1:,} reference S1 records.")

    # 2. Normalize and serialize S1
    s1_df = create_normalized_dataframe(s1_df)
    s1_df["serialized"] = serialize_dataframe(s1_df)
    
    # Fast in-memory lookup for candidate serialization: {s1_id: serialized_text}
    s1_text_lookup: Dict[str, str] = dict(zip(s1_df["entity_id"], s1_df["serialized"]))

    # 3. Fit Sparse Blocker on S1
    blocker = SparseBlockingEngine(top_k_candidates=12)
    blocker.fit(s1_df)

    # Move model to device in eval mode
    model.to(device)
    model.eval()

    # Pre-load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.base_model_name)

    # Accumulator maps initialized for ALL reference entities (guarantees complete row coverage)
    all_candidates_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    all_matches_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}

    # 4. Stream S2 and S3 query datasets
    total_processed_queries = 0

    for query_source, q_path in [("Source2", s2_path), ("Source3", s3_path)]:
        if not q_path.exists():
            logger.warning(f"[INFERENCE] File {q_path.name} not found. Skipping...")
            continue

        logger.info(f"[INFERENCE] Streaming queries from {query_source} ({q_path.name})...")
        chunk_iter = pd.read_csv(q_path, sep="\t", chunksize=chunk_size, dtype=str)

        for chunk_idx, chunk_df in enumerate(chunk_iter, start=1):
            chunk_t0 = time.time()
            if max_queries and total_processed_queries >= max_queries:
                logger.info(f"[INFERENCE] Reached query budget ({max_queries:,}). Stopping.")
                break

            num_in_chunk = len(chunk_df)
            chunk_df = create_normalized_dataframe(chunk_df)
            chunk_df["serialized"] = serialize_dataframe(chunk_df)
            q_ids = chunk_df["entity_id"].tolist()
            q_texts = chunk_df["serialized"].tolist()

            # Retrieve candidate S1 IDs for each query in chunk
            cand_map = blocker.generate_candidates_for_chunk(chunk_df)

            # Build list of unique pairs to score with ColBERT
            eval_queries: List[str] = []
            eval_refs: List[str] = []
            eval_pairs: List[Tuple[str, str]] = []

            for qid, q_text in zip(q_ids, q_texts):
                cands = cand_map.get(qid, [])
                for s1_cand in cands:
                    ref_text = s1_text_lookup.get(s1_cand, "")
                    if ref_text:
                        eval_queries.append(q_text)
                        eval_refs.append(ref_text)
                        eval_pairs.append((qid, s1_cand))

            # Batch encode and score pairs on GPU via Late-Interaction MaxSim
            scores: List[float] = []
            batch_eval_size = 256

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    for b_start in range(0, len(eval_pairs), batch_eval_size):
                        b_end = min(b_start + batch_eval_size, len(eval_pairs))
                        b_q = eval_queries[b_start:b_end]
                        b_r = eval_refs[b_start:b_end]

                        tok_q = tokenizer(b_q, padding=True, truncation=True, max_length=DEFAULT_MAX_SEQ_LEN, return_tensors="pt")
                        tok_r = tokenizer(b_r, padding=True, truncation=True, max_length=DEFAULT_MAX_SEQ_LEN, return_tensors="pt")

                        q_ids_t = tok_q["input_ids"].to(device)
                        q_mask_t = tok_q["attention_mask"].to(device)
                        d_ids_t = tok_r["input_ids"].to(device)
                        d_mask_t = tok_r["attention_mask"].to(device)

                        # Encode tokens
                        q_emb = model.encode_tokens(q_ids_t, q_mask_t)
                        d_emb = model.encode_tokens(d_ids_t, d_mask_t)

                        # Compute MaxSim score
                        b_maxsim = compute_maxsim_score(q_emb, q_mask_t, d_emb, d_mask_t)
                        # Calibrated probability
                        b_probs = torch.sigmoid((b_maxsim - model.bias) / torch.clamp(model.tau, min=0.01))
                        scores.extend(b_probs.cpu().numpy().tolist())

            # Group scored candidates per query
            chunk_query_scores: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in q_ids}
            for (qid, s1_cand), sc in zip(eval_pairs, scores):
                chunk_query_scores[qid].append((s1_cand, sc))

            # Apply assignment rules (Query Exclusivity & Thresholds)
            chunk_matches, chunk_cands = apply_assignment_rules(
                chunk_query_scores,
                all_s1_ids,
                abs_threshold=abs_threshold,
                margin_threshold=margin_threshold
            )

            # Merge chunk results into global accumulators
            for s1, q_set in chunk_cands.items():
                all_candidates_map[s1].update(q_set)
            for s1, m_set in chunk_matches.items():
                all_matches_map[s1].update(m_set)

            total_processed_queries += num_in_chunk
            chunk_sec = time.time() - chunk_t0
            mem_mb = psutil.Process().memory_info().rss / (1024**2)
            logger.info(
                f"[INFERENCE] {query_source} Chunk {chunk_idx:3d} | Queries: {total_processed_queries:,} | "
                f"Pairs: {len(eval_pairs):,} | Time: {chunk_sec:.1f}s | RAM: {mem_mb:.1f} MB"
            )

            # Explicit garbage collection between chunks to keep constant memory footprint
            del chunk_df, cand_map, eval_queries, eval_refs, eval_pairs, scores, chunk_query_scores
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5. Flush results to official TSV format
    logger.info("=" * 60)
    logger.info(f"[INFERENCE] Writing official TSV submission files to {output_matching_path.parent}...")
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)

    # Write matching_results.tsv
    with open(output_matching_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ids:
            matched_list = sorted(list(all_matches_map.get(s1, set())))
            f_match.write(f"{s1}\t{','.join(matched_list)}\n")

    # Write candidate_pairs.tsv
    with open(output_candidate_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1_ids:
            cand_list = sorted(list(all_candidates_map.get(s1, set())))
            f_cand.write(f"{s1}\t{','.join(cand_list)}\n")

    total_time = time.time() - start_t
    num_matches = sum(len(v) for v in all_matches_map.values())
    num_candidates = sum(len(v) for v in all_candidates_map.values())

    logger.info(f"[INFERENCE] Complete in {total_time:.1f}s.")
    logger.info(f"  Total S1 Entities:     {num_s1:,}")
    logger.info(f"  Total Candidates:      {num_candidates:,}")
    logger.info(f"  Total Matches:         {num_matches:,}")
    logger.info(f"  Output Matching TSV:   {output_matching_path}")
    logger.info(f"  Output Candidate TSV:  {output_candidate_path}")
    logger.info("=" * 60)

    return {
        "num_s1": num_s1,
        "total_queries": total_processed_queries,
        "num_matches": num_matches,
        "num_candidates": num_candidates,
        "runtime_sec": total_time,
        "matching_path": str(output_matching_path),
        "candidate_path": str(output_candidate_path)
    }
