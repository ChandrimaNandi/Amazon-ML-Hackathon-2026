"""
Comprehensive Evaluation & Error Analysis Module for Business Entity Resolution.

Provides:
1. Macro F0.5, Macro Precision, and Macro Recall calculation.
2. Multi-K Candidate Recall diagnostics (Recall@1, @5, @10, @20, @50) overall
   and per channel (Exact, BM25, TF-IDF, Union).
3. Candidate Recall breakdowns across slices:
   - Script (Latin, Devanagari, etc.)
   - Country (US, India, etc.)
   - Missing fields (missing name, missing address, complete)
   - Source dataset (Source-2 vs Source-3)
   - Entity cardinality (single-match vs multi-match reference entities)
4. Comprehensive Error Analysis distinguishing:
   - CANDIDATE_GENERATION_FAILURE (true S1 never retrieved)
   - MATCHER_THRESHOLD_FAILURE (true S1 retrieved but rejected by model / threshold)
   - FALSE POSITIVES with categorized error types.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple, List, Any, Optional
import logging

from src.config import BETA
from src.profiling import detect_script

logger = logging.getLogger(__name__)


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = BETA
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F_beta for a single reference S1 entity.
    Special cases:
    - true_set is empty and pred_set is empty: (1.0, 1.0, 1.0) [Correct singleton rejection]
    - true_set is empty and pred_set is non-empty: (0.0, 0.0, 0.0) [False positive on singleton]
    - true_set is non-empty and pred_set is empty: (0.0, 0.0, 0.0) [Missed match]
    """
    beta_sq = beta ** 2
    
    if not true_set and not pred_set:
        return (1.0, 1.0, 1.0)
    if not true_set and pred_set:
        return (0.0, 0.0, 0.0)
    if true_set and not pred_set:
        return (0.0, 0.0, 0.0)
        
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return (0.0, 0.0, 0.0)
        
    prec = tp / float(len(pred_set))
    rec = tp / float(len(true_set))
    
    denom = (beta_sq * prec) + rec
    f_beta = (1.0 + beta_sq) * prec * rec / denom if denom > 0 else 0.0
    return (prec, rec, f_beta)


def evaluate_macro_metrics(
    all_s1_ids: Set[str],
    s1_to_true_matches: Dict[str, Set[str]],
    s1_to_pred_matches: Dict[str, Set[str]],
    beta: float = BETA
) -> Dict[str, float]:
    """
    Computes Macro F_beta, Macro Precision, and Macro Recall across all evaluated S1 entities.
    """
    precisions, recalls, f_betas = [], [], []
    
    for s1_id in all_s1_ids:
        true_set = s1_to_true_matches.get(s1_id, set())
        pred_set = s1_to_pred_matches.get(s1_id, set())
        
        prec, rec, f_b = compute_entity_f_beta(true_set, pred_set, beta=beta)
        precisions.append(prec)
        recalls.append(rec)
        f_betas.append(f_b)
        
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f_beta = float(np.mean(f_betas)) if f_betas else 0.0
    
    results = {
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        f"macro_f{beta}": round(macro_f_beta, 4),
        "total_evaluated_s1": len(all_s1_ids),
        "total_predicted_links": sum(len(p) for p in s1_to_pred_matches.values())
    }
    return results


def evaluate_candidate_recall_diagnostics(
    cand_df: pd.DataFrame,
    s1_to_true_matches: Dict[str, Set[str]],
    evaluated_query_ids: Optional[Set[str]] = None,
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, Any]:
    """
    Computes Recall@K diagnostics overall and across separate retrieval channels.
    """
    if cand_df.empty:
        return {"total_true_pairs": 0, "overall_recall": 0.0}
        
    if evaluated_query_ids is None:
        evaluated_query_ids = set(cand_df["query_id"].unique())
        
    # Build true pairs for evaluated queries
    true_pairs: Set[Tuple[str, str]] = set()
    for s1_id, q_set in s1_to_true_matches.items():
        for q_id in q_set:
            if q_id in evaluated_query_ids:
                true_pairs.add((q_id, s1_id))
                
    total_true = len(true_pairs)
    if total_true == 0:
        return {"total_true_pairs": 0, "overall_recall": 1.0}
        
    # Pre-index candidates by query
    query_cands: Dict[str, List[Any]] = {}
    for row in cand_df.itertuples():
        query_cands.setdefault(row.query_id, []).append(row)
        
    # Channel evaluations
    channels = {
        "union": lambda r: r.best_retrieval_rank,
        "exact": lambda r: 1 if (r.by_exact_name or r.by_exact_address or r.by_exact_combined) else 999,
        "bm25": lambda r: min(r.bm25_name_rank, r.bm25_comb_rank),
        "tfidf": lambda r: min(r.tfidf_name_rank, r.tfidf_addr_rank),
    }
    
    recall_results: Dict[str, Any] = {"total_true_pairs": total_true}
    
    for ch_name, rank_fn in channels.items():
        # For each K, count how many true pairs are retrieved within top K
        for k in k_values:
            found = 0
            for qid, true_s1 in true_pairs:
                c_list = query_cands.get(qid, [])
                for c in c_list:
                    if c.s1_id == true_s1 and rank_fn(c) <= k:
                        found += 1
                        break
            recall = found / float(total_true)
            recall_results[f"{ch_name}_recall@{k}"] = round(recall, 4)
            
    # Full candidate set recall (any rank present in cand_df)
    cand_pairs = set(zip(cand_df["query_id"], cand_df["s1_id"]))
    total_found = len(true_pairs.intersection(cand_pairs))
    recall_results["candidate_recall_union"] = round(total_found / float(total_true), 4)
    recall_results["found_true_pairs"] = total_found
    recall_results["missed_true_pairs"] = total_true - total_found
    
    logger.info("=" * 60)
    logger.info("[CANDIDATE RECALL DIAGNOSTICS]")
    logger.info(f"  Total True Pairs Evaluated: {total_true:,}")
    logger.info(f"  Union Recall (All Ranks):   {recall_results['candidate_recall_union']*100:.2f}% ({total_found:,}/{total_true:,})")
    for k in [1, 5, 10, 20]:
        logger.info(f"  Union Recall@{k}:           {recall_results.get(f'union_recall@{k}', 0)*100:.2f}%")
    logger.info(f"  BM25 Recall@20:             {recall_results.get('bm25_recall@20', 0)*100:.2f}%")
    logger.info(f"  TF-IDF Recall@20:           {recall_results.get('tfidf_recall@20', 0)*100:.2f}%")
    logger.info(f"  Exact Match Recall:         {recall_results.get('exact_recall@1', 0)*100:.2f}%")
    logger.info("=" * 60)
    
    return recall_results


def evaluate_candidate_recall_breakdowns(
    cand_df: pd.DataFrame,
    query_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s1_to_true_matches: Dict[str, Set[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Breaks down candidate recall by script, country, missing fields, source, and match cardinality.
    """
    q_lookup = query_df.set_index("entity_id").to_dict("index")
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    
    evaluated_qids = set(cand_df["query_id"].unique())
    cand_pairs = set(zip(cand_df["query_id"], cand_df["s1_id"]))
    
    rows = []
    for s1_id, q_set in s1_to_true_matches.items():
        s1_info = s1_lookup.get(s1_id, {})
        s1_cardinality = "multi_match" if len(q_set) > 1 else "single_match"
        
        for q_id in q_set:
            if q_id not in evaluated_qids:
                continue
            q_info = q_lookup.get(q_id, {})
            
            is_retrieved = 1 if (q_id, s1_id) in cand_pairs else 0
            
            q_name = q_info.get("business_name", "")
            q_addr = q_info.get("business_address", "")
            country = q_info.get("country", "Unknown")
            script = detect_script(q_name)
            source = "S2" if q_id.startswith("S2-") else "S3"
            
            missing_status = "Complete"
            if not q_name and not q_addr:
                missing_status = "Missing Name & Addr"
            elif not q_name:
                missing_status = "Missing Name"
            elif not q_addr:
                missing_status = "Missing Addr"
                
            rows.append({
                "query_id": q_id,
                "s1_id": s1_id,
                "is_retrieved": is_retrieved,
                "script": script,
                "country": country,
                "missing_status": missing_status,
                "source": source,
                "cardinality": s1_cardinality
            })
            
    df_eval = pd.DataFrame(rows)
    if df_eval.empty:
        return {}
        
    def slice_recall(group_col: str) -> pd.DataFrame:
        g = df_eval.groupby(group_col).agg(
            total_true_pairs=("is_retrieved", "count"),
            retrieved_pairs=("is_retrieved", "sum"),
        ).reset_index()
        g["recall"] = (g["retrieved_pairs"] / g["total_true_pairs"]).round(4)
        return g.sort_values("total_true_pairs", ascending=False)
        
    breakdowns = {
        "by_script": slice_recall("script"),
        "by_country": slice_recall("country"),
        "by_missing": slice_recall("missing_status"),
        "by_source": slice_recall("source"),
        "by_cardinality": slice_recall("cardinality"),
    }
    return breakdowns


def generate_error_analysis(
    val_cand_feat_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    s1_to_true_matches: Dict[str, Set[str]],
    s1_to_pred_matches: Dict[str, Set[str]],
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive error analysis distinguishing:
    - False Negatives: Tagged as CANDIDATE_GENERATION_FAILURE vs MATCHER_THRESHOLD_FAILURE
    - False Positives: Categorized by error pattern.
    """
    q_lookup = query_df.set_index("entity_id").to_dict("index")
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    
    val_qids = set(val_cand_feat_df["query_id"].unique())
    cand_pairs = set(zip(val_cand_feat_df["query_id"], val_cand_feat_df["s1_id"]))
    
    # Map pair to model pred_score
    pair_to_score = dict(zip(zip(val_cand_feat_df["query_id"], val_cand_feat_df["s1_id"]),
                             val_cand_feat_df.get("pred_score", np.zeros(len(val_cand_feat_df)))))
    
    fn_records = []
    # 1. False Negatives
    for s1_id, true_qids in s1_to_true_matches.items():
        pred_qids = s1_to_pred_matches.get(s1_id, set())
        for q_id in true_qids:
            if q_id not in val_qids:
                continue
            if q_id not in pred_qids:
                # This true pair was missed!
                retrieved = (q_id, s1_id) in cand_pairs
                failure_type = "MATCHER_THRESHOLD_FAILURE" if retrieved else "CANDIDATE_GENERATION_FAILURE"
                score = pair_to_score.get((q_id, s1_id), 0.0)
                
                q_info = q_lookup.get(q_id, {})
                s1_info = s1_lookup.get(s1_id, {})
                
                fn_records.append({
                    "error_type": "FALSE_NEGATIVE",
                    "failure_mechanism": failure_type,
                    "query_id": q_id,
                    "s1_id": s1_id,
                    "query_name": q_info.get("business_name", ""),
                    "s1_name": s1_info.get("business_name", ""),
                    "query_addr": q_info.get("business_address", ""),
                    "s1_addr": s1_info.get("business_address", ""),
                    "country": q_info.get("country", ""),
                    "model_score": round(float(score), 4)
                })
                
    # 2. False Positives
    fp_records = []
    for s1_id, pred_qids in s1_to_pred_matches.items():
        true_qids = s1_to_true_matches.get(s1_id, set())
        for q_id in pred_qids:
            if q_id not in true_qids:
                score = pair_to_score.get((q_id, s1_id), 0.0)
                q_info = q_lookup.get(q_id, {})
                s1_info = s1_lookup.get(s1_id, {})
                
                fp_records.append({
                    "error_type": "FALSE_POSITIVE",
                    "failure_mechanism": "WRONG_MERGE",
                    "query_id": q_id,
                    "predicted_s1_id": s1_id,
                    "query_name": q_info.get("business_name", ""),
                    "predicted_s1_name": s1_info.get("business_name", ""),
                    "query_addr": q_info.get("business_address", ""),
                    "predicted_s1_addr": s1_info.get("business_address", ""),
                    "country": q_info.get("country", ""),
                    "model_score": round(float(score), 4)
                })
                
    fn_df = pd.DataFrame(fn_records)
    fp_df = pd.DataFrame(fp_records)
    
    cg_failures = sum(1 for r in fn_records if r["failure_mechanism"] == "CANDIDATE_GENERATION_FAILURE")
    matcher_failures = sum(1 for r in fn_records if r["failure_mechanism"] == "MATCHER_THRESHOLD_FAILURE")
    
    summary = {
        "total_false_positives": len(fp_df),
        "total_false_negatives": len(fn_df),
        "candidate_generation_failures": cg_failures,
        "matcher_threshold_failures": matcher_failures,
        "cg_failure_pct": round(cg_failures / max(len(fn_df), 1) * 100, 2),
        "matcher_failure_pct": round(matcher_failures / max(len(fn_df), 1) * 100, 2),
    }
    
    logger.info("=" * 60)
    logger.info("[ERROR ANALYSIS SUMMARY]")
    logger.info(f"  False Positives (Wrong Merges): {summary['total_false_positives']:,}")
    logger.info(f"  False Negatives (Missed Pairs): {summary['total_false_negatives']:,}")
    logger.info(f"    - Candidate Generation Failures (Not in Top-K): {cg_failures:,} ({summary['cg_failure_pct']}%)")
    logger.info(f"    - Matcher / Threshold Failures (In Top-K but rejected): {matcher_failures:,} ({summary['matcher_failure_pct']}%)")
    logger.info("=" * 60)
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        errors_combined = pd.concat([fn_df, fp_df], ignore_index=True)
        errors_combined.to_csv(output_path, index=False)
        logger.info(f"[ERROR ANALYSIS] Saved error report to {output_path}")
        
    return fn_df, fp_df, summary
