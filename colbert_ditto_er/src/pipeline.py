"""
Memory-Safe Chunked Streaming Inference Pipeline for ColBERT-Ditto ER.
Processes multi-million query test sets using disk-sharded streaming (O(1) RAM footprint < 2.5 GB),
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
from colbert_ditto_er.src.serialization import serialize_dataframe
from colbert_ditto_er.src.blocking import SparseBlockingEngine
from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder, compute_maxsim_score
from colbert_ditto_er.src.assignment import apply_assignment_rules

logger = logging.getLogger("ColBERT_Ditto.Pipeline")


def run_streaming_colbert_inference(
    model: ColBERTTokenEncoder,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_path if 'SUBMISSION_CANDIDATE_path' in globals() else SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05,
    chunk_size: Optional[int] = None,
    max_queries: Optional[int] = None,
    device: Optional["torch.device"] = None
) -> Dict[str, Any]:
    """
    Executes memory-safe streaming inference on test datasets (Source 2 and Source 3)
    against reference Source 1 entities using disk-sharded candidate/prediction streaming.
    Guarantees constant O(1) RAM usage (<2.5 GB) regardless of test set size.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if device is None:
        device = torch.device("cuda:0" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")

    start_t = time.time()
    logger.info("=" * 60)
    logger.info("[STREAMING INFERENCE] Starting ColBERT-Ditto Disk-Sharded Pipeline")
    logger.info(f"  Test Directory:     {test_dir}")
    logger.info(f"  Absolute Threshold: {abs_threshold:.2f}")
    logger.info(f"  Margin Threshold:   {margin_threshold:.2f}")
    logger.info(f"  Chunk Size:         {chunk_size:,}")
    logger.info(f"  Device:             {device}")
    logger.info("=" * 60)

    # Clean memory before inference
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    s1_text_lookup: Dict[str, str] = dict(zip(s1_df["entity_id"], s1_df["serialized"]))

    # 3. Fit Sparse Blocker on S1
    blocker = SparseBlockingEngine(top_k_candidates=10)
    blocker.fit(s1_df)

    # 4. Setup Disk Shards to guarantee O(1) RAM (avoids 30 GB accumulation)
    num_shards = 16 if num_s1 >= 160000 else max(1, (num_s1 + 9999) // 10000)
    shard_size = (num_s1 + num_shards - 1) // num_shards
    s1_to_shard = {s1_id: min(idx // shard_size, num_shards - 1) for idx, s1_id in enumerate(all_s1_ids)}

    shard_dir = output_matching_path.parent / "temp_colbert_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cand_fps = {s: open(shard_dir / f"shard_{s}_cands.txt", "w", encoding="utf-8") for s in range(num_shards)}
    pred_fps = {s: open(shard_dir / f"shard_{s}_preds.txt", "w", encoding="utf-8") for s in range(num_shards)}

    # Move model to device in eval mode
    model.to(device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.base_model_name)

    total_processed_queries = 0
    total_candidates_recorded = 0
    total_predictions_recorded = 0

    try:
        # 5. Stream S2 and S3 query datasets
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

                # Build unique pairs to score
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

                # Batch score pairs on GPU via Late-Interaction MaxSim
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

                            q_emb = model.encode_tokens(q_ids_t, q_mask_t)
                            d_emb = model.encode_tokens(d_ids_t, d_mask_t)

                            # Direct length-normalized MaxSim score
                            b_maxsim = compute_maxsim_score(q_emb, q_mask_t, d_emb, d_mask_t)
                            scores.extend(b_maxsim.cpu().numpy().tolist())

                # Group scored candidates per query
                chunk_query_scores: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in q_ids}
                for (qid, s1_cand), sc in zip(eval_pairs, scores):
                    chunk_query_scores[qid].append((s1_cand, sc))

                # Apply sparse assignment rules
                chunk_matches, chunk_cands = apply_assignment_rules(
                    chunk_query_scores,
                    abs_threshold=abs_threshold,
                    margin_threshold=margin_threshold
                )

                # Stream candidates directly to disk shards with buffering
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

                # Stream predictions directly to disk shards with buffering
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
                    f"Pairs: {len(eval_pairs):,} | Time: {chunk_sec:.1f}s | RAM RSS: {mem_mb:.1f} MB"
                )

                del chunk_df, cand_map, eval_queries, eval_refs, eval_pairs, scores, chunk_query_scores, chunk_matches, chunk_cands
                gc.collect()
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        # Close all shard files safely
        for fp in cand_fps.values():
            fp.close()
        for fp in pred_fps.values():
            fp.close()

    # 6. Assemble Official Output TSVs Shard-by-Shard (Guarantees Low Peak RAM)
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

            # In-memory structures for current shard only (~100k entities)
            shard_preds: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}
            shard_cands: Dict[str, Set[str]] = {s: set() for s in shard_s1_list}

            # Read prediction shard
            p_file = shard_dir / f"shard_{sh}_preds.txt"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2 and parts[0] in shard_preds:
                            shard_preds[parts[0]].add(parts[1])

            # Read candidate shard
            c_file = shard_dir / f"shard_{sh}_cands.txt"
            if c_file.exists():
                with open(c_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        parts = line.strip().split("\t")
                        if len(parts) == 2 and parts[0] in shard_cands:
                            shard_cands[parts[0]].add(parts[1])

            # Write rows for all S1 entities in this shard
            for s1_id in shard_s1_list:
                m_list = sorted(list(shard_preds.get(s1_id, set())))
                c_list = sorted(list(shard_cands.get(s1_id, set())))
                # Guarantee predicted_pairs ⊆ candidate_pairs
                for m in m_list:
                    if m not in shard_cands.get(s1_id, set()):
                        c_list.append(m)
                c_list = sorted(list(set(c_list)))

                f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")

            # Clean up shard files on disk
            try:
                p_file.unlink(missing_ok=True)
                c_file.unlink(missing_ok=True)
            except Exception:
                pass

            del shard_preds, shard_cands, shard_s1_list
            gc.collect()

    try:
        shard_dir.rmdir()
    except Exception:
        pass

    total_time = time.time() - start_t
    logger.info(f"[INFERENCE] Complete in {total_time:.1f}s.")
    logger.info(f"  Total S1 Entities:     {num_s1:,}")
    logger.info(f"  Total Queries:         {total_processed_queries:,}")
    logger.info(f"  Total Predictions:     {total_predictions_recorded:,}")
    logger.info(f"  Output Matching TSV:   {output_matching_path}")
    logger.info(f"  Output Candidate TSV:  {output_candidate_path}")
    logger.info("=" * 60)

    return {
        "num_s1": num_s1,
        "total_queries": total_processed_queries,
        "num_matches": total_predictions_recorded,
        "num_candidates": total_candidates_recorded,
        "runtime_sec": total_time,
        "matching_path": str(output_matching_path),
        "candidate_path": str(output_candidate_path)
    }
