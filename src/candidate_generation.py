"""
Unified Candidate Generation Module for Business Entity Resolution.

Provides a single reusable CandidateGenerator class for:
- Training
- Validation
- Large-scale Test Inference

Channels:
1. Exact Name (exact normalized string match)
2. Exact Address (exact normalized string match)
3. Exact Combined (exact normalized name + address match)
4. BM25 Name (word-level inverted index BM25)
5. BM25 Combined (word-level inverted index BM25 on combined name + address)
6. Char-TFIDF Name (character 3-5 gram cosine similarity)
7. Char-TFIDF Address (character 3-5 gram cosine similarity)

Every candidate pair preserves comprehensive channel flags, scores, and ranks.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

from src.retrieval import CharTFIDFRetriever, SparseBM25Retriever, ExactMatchIndex
from src.normalization import create_normalized_features

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Unified Candidate Generator ensuring identical candidate generation logic
    across training, validation, and final test inference.
    """
    def __init__(
        self,
        k_exact_cap: int = 30,
        k_bm25_name: int = 15,
        k_bm25_comb: int = 15,
        k_tfidf_name: int = 15,
        k_tfidf_addr: int = 10,
        max_features: int = 100000
    ):
        self.k_exact_cap = k_exact_cap
        self.k_bm25_name = k_bm25_name
        self.k_bm25_comb = k_bm25_comb
        self.k_tfidf_name = k_tfidf_name
        self.k_tfidf_addr = k_tfidf_addr
        
        # Retrievers
        self.exact_name = ExactMatchIndex(max_bucket_size=k_exact_cap)
        self.exact_addr = ExactMatchIndex(max_bucket_size=k_exact_cap)
        self.exact_comb = ExactMatchIndex(max_bucket_size=k_exact_cap)
        
        self.bm25_name = SparseBM25Retriever(max_features=max_features)
        self.bm25_comb = SparseBM25Retriever(max_features=max_features)
        
        self.tfidf_name = CharTFIDFRetriever(ngram_range=(3, 5), max_features=max_features)
        self.tfidf_addr = CharTFIDFRetriever(ngram_range=(3, 5), max_features=max_features)
        
        self.is_fitted = False
        self.s1_ids: List[str] = []
        self.s1_countries: Dict[str, str] = {}

    def fit(self, s1_df: pd.DataFrame):
        """
        Fits all retrieval indices on reference S1 corpus.
        Requires normalized fields (computes them if missing).
        """
        logger.info("=" * 60)
        logger.info(f"[CANDIDATE GENERATION] Fitting multi-channel retrieval on {len(s1_df):,} reference entities...")
        logger.info("=" * 60)
        start_t = time.time()
        
        if "name_normalized" not in s1_df.columns:
            s1_df = create_normalized_features(s1_df)
            
        self.s1_ids = s1_df["entity_id"].tolist()
        self.s1_countries = {
            row.entity_id: str(getattr(row, "country", "")).strip().casefold()
            for row in s1_df.itertuples()
        }
        names = s1_df["name_normalized"].tolist()
        addrs = s1_df["address_normalized"].tolist()
        combs = s1_df["combined_normalized"].tolist()
        
        # 1. Exact match indices
        self.exact_name.fit(names, self.s1_ids)
        self.exact_addr.fit(addrs, self.s1_ids)
        self.exact_comb.fit(combs, self.s1_ids)
        
        # 2. BM25 retrievers
        self.bm25_name.fit(names, self.s1_ids)
        self.bm25_comb.fit(combs, self.s1_ids)
        
        # 3. Char TF-IDF retrievers
        self.tfidf_name.fit(names, self.s1_ids)
        self.tfidf_addr.fit(addrs, self.s1_ids)
        
        self.is_fitted = True
        elapsed = time.time() - start_t
        logger.info(f"[CANDIDATE GENERATION] All retrieval channels fitted in {elapsed:.2f}s.")

    def generate_candidates(
        self,
        query_df: pd.DataFrame,
        batch_size: int = 5000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates candidate pairs for query_df against fitted S1 corpus with country hard-blocking.
        
        Returns:
            candidate_df: DataFrame with query_id, s1_id, and all channel flags/ranks/scores.
            stats: Summary metrics dictionary.
        """
        if not self.is_fitted:
            raise ValueError("CandidateGenerator must be fitted before generating candidates.")
            
        start_t = time.time()
        num_queries = len(query_df)
        logger.info(f"[CANDIDATE GENERATION] Generating candidates for {num_queries:,} queries...")
        
        if "name_normalized" not in query_df.columns:
            query_df = create_normalized_features(query_df)
            
        q_ids = query_df["entity_id"].tolist()
        q_names = query_df["name_normalized"].tolist()
        q_addrs = query_df["address_normalized"].tolist()
        q_combs = query_df["combined_normalized"].tolist()
        q_countries = [
            str(getattr(r, "country", "")).strip().casefold()
            for r in query_df.itertuples()
        ]
        
        # Run retrieval channels in parallel batches
        bm25_name_res = self.bm25_name.retrieve_top_k(q_names, top_k=self.k_bm25_name, batch_size=batch_size)
        bm25_comb_res = self.bm25_comb.retrieve_top_k(q_combs, top_k=self.k_bm25_comb, batch_size=batch_size)
        tfidf_name_res = self.tfidf_name.retrieve_top_k(q_names, top_k=self.k_tfidf_name, batch_size=batch_size)
        tfidf_addr_res = self.tfidf_addr.retrieve_top_k(q_addrs, top_k=self.k_tfidf_addr, batch_size=batch_size)
        
        candidate_records = []
        
        for idx in range(num_queries):
            qid = q_ids[idx]
            qname = q_names[idx]
            qaddr = q_addrs[idx]
            qcomb = q_combs[idx]
            q_cntry = q_countries[idx] if idx < len(q_countries) else ""
            
            # Map of s1_id -> candidate dictionary
            candidates: Dict[str, Dict[str, Any]] = {}
            
            def get_cand(s1_id: str) -> Optional[Dict[str, Any]]:
                # Strict country hard-blocking (US never matches India)
                if q_cntry:
                    s1_cntry = self.s1_countries.get(s1_id, "")
                    if s1_cntry and s1_cntry != q_cntry:
                        return None
                        
                if s1_id not in candidates:
                    candidates[s1_id] = {
                        "query_id": qid,
                        "s1_id": s1_id,
                        "by_exact_name": 0,
                        "by_exact_address": 0,
                        "by_exact_combined": 0,
                        "by_bm25_name": 0,
                        "bm25_name_score": 0.0,
                        "bm25_name_rank": 999,
                        "by_bm25_combined": 0,
                        "bm25_comb_score": 0.0,
                        "bm25_comb_rank": 999,
                        "by_tfidf_name": 0,
                        "tfidf_name_score": 0.0,
                        "tfidf_name_rank": 999,
                        "by_tfidf_address": 0,
                        "tfidf_addr_score": 0.0,
                        "tfidf_addr_rank": 999,
                    }
                return candidates[s1_id]
            
            # 1. Exact Name
            for s1_id in self.exact_name.lookup(qname):
                c = get_cand(s1_id)
                if c is not None:
                    c["by_exact_name"] = 1
                
            # 2. Exact Address
            for s1_id in self.exact_addr.lookup(qaddr):
                c = get_cand(s1_id)
                if c is not None:
                    c["by_exact_address"] = 1
                
            # 3. Exact Combined
            for s1_id in self.exact_comb.lookup(qcomb):
                c = get_cand(s1_id)
                if c is not None:
                    c["by_exact_combined"] = 1
                
            # 4. BM25 Name
            for s1_id, sc, rk in bm25_name_res[idx]:
                c = get_cand(s1_id)
                if c is not None:
                    c["by_bm25_name"] = 1
                    c["bm25_name_score"] = max(c["bm25_name_score"], sc)
                    c["bm25_name_rank"] = min(c["bm25_name_rank"], rk)
                
            # 5. BM25 Combined
            for s1_id, sc, rk in bm25_comb_res[idx]:
                c = get_cand(s1_id)
                if c is not None:
                    c["by_bm25_combined"] = 1
                    c["bm25_comb_score"] = max(c["bm25_comb_score"], sc)
                    c["bm25_comb_rank"] = min(c["bm25_comb_rank"], rk)
                
            # 6. Char TF-IDF Name
            for s1_id, sc, rk in tfidf_name_res[idx]:
                c = get_cand(s1_id)
                if c is not None:
                    c["by_tfidf_name"] = 1
                    c["tfidf_name_score"] = max(c["tfidf_name_score"], sc)
                    c["tfidf_name_rank"] = min(c["tfidf_name_rank"], rk)
                
            # 7. Char TF-IDF Address
            for s1_id, sc, rk in tfidf_addr_res[idx]:
                c = get_cand(s1_id)
                if c is not None:
                    c["by_tfidf_address"] = 1
                    c["tfidf_addr_score"] = max(c["tfidf_addr_score"], sc)
                    c["tfidf_addr_rank"] = min(c["tfidf_addr_rank"], rk)
                
            # Calculate composite agreement and reciprocal rank
            for s1_id, c in candidates.items():
                agree_cnt = (
                    c["by_exact_name"] + c["by_exact_address"] + c["by_exact_combined"] +
                    c["by_bm25_name"] + c["by_bm25_combined"] +
                    c["by_tfidf_name"] + c["by_tfidf_address"]
                )
                best_rk = min(
                    c["bm25_name_rank"], c["bm25_comb_rank"],
                    c["tfidf_name_rank"], c["tfidf_addr_rank"]
                )
                c["retrieval_agreement_count"] = agree_cnt
                c["best_retrieval_rank"] = best_rk
                c["best_reciprocal_rank"] = 1.0 / best_rk if best_rk < 999 else 0.0
                candidate_records.append(c)
                
        candidate_df = pd.DataFrame(candidate_records)
        elapsed = time.time() - start_t
        
        avg_cands = len(candidate_df) / max(num_queries, 1)
        stats = {
            "num_queries": num_queries,
            "total_candidate_pairs": len(candidate_df),
            "avg_candidates_per_query": round(avg_cands, 2),
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(
            f"[CANDIDATE GENERATION] Generated {len(candidate_df):,} candidate pairs "
            f"(avg {avg_cands:.1f}/query) in {elapsed:.2f}s."
        )
        return candidate_df, stats


def generate_candidate_union(
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    k_name: int = 25,
    k_address: int = 20,
    k_combined: int = 25,
    k_char: int = 25
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience functional wrapper for CandidateGenerator.
    """
    generator = CandidateGenerator(
        k_bm25_name=k_name,
        k_bm25_comb=k_combined,
        k_tfidf_name=k_char,
        k_tfidf_addr=k_address
    )
    generator.fit(s1_df)
    return generator.generate_candidates(query_df)
