"""
Pillar 1 & 2: Tier 1 Exact Sieve and High-Recall Sparse Blocker.
Combines deterministic O(1) hash resolution for clean pairs with Sparse BM25 retrieval for noisy pairs.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

from precision_sieve_er.src.normalization import create_normalized_dataframe

logger = logging.getLogger("PrecisionSieve.Blocking")


class SparseBM25Retriever:
    """Scalable sparse BM25 retriever using CountVectorizer and SciPy CSR matrix multiplication."""
    def __init__(self, k1: float = 1.5, b: float = 0.75, max_features: int = 100000, max_df: float = 0.25):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.max_df = max_df
        self.vectorizer = CountVectorizer(
            max_features=max_features, max_df=max_df, dtype=np.float32, token_pattern=r"(?u)\b\w+\b"
        )
        self.corpus_bm25_T: Optional[csr_matrix] = None
        self.s1_ids: List[str] = []

    def fit(self, texts: List[str], s1_ids: List[str]):
        if len(texts) < 20:
            self.vectorizer.set_params(max_df=1.0)
        else:
            self.vectorizer.set_params(max_df=self.max_df)
        counts = self.vectorizer.fit_transform(texts)
        doc_lens = np.asarray(counts.sum(axis=1)).flatten()
        avgdl = float(np.mean(doc_lens)) if len(doc_lens) > 0 else 1.0
        N = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.maximum(np.log((N - df + 0.5) / (df + 0.5) + 1.0), 0.0)
        denom = counts.data + self.k1 * (1.0 - self.b + self.b * (doc_lens[counts.nonzero()[0]] / max(avgdl, 1e-6)))
        counts.data = (counts.data * (self.k1 + 1.0) / np.maximum(denom, 1e-6)) * idf[counts.indices]
        self.corpus_bm25_T = counts.T.tocsr()
        self.s1_ids = list(s1_ids)

    def retrieve_batch(self, queries: List[str], top_k: int = 8) -> List[List[Tuple[str, float]]]:
        if not queries or self.corpus_bm25_T is None:
            return [[] for _ in queries]
        q_counts = self.vectorizer.transform(queries)
        scores_matrix = q_counts.dot(self.corpus_bm25_T)
        results = []
        for i in range(scores_matrix.shape[0]):
            row = scores_matrix.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue
            idx_data = row.indices
            data = row.data
            if len(data) > top_k:
                top_part = np.argpartition(-data, top_k)[:top_k]
                order = top_part[np.argsort(-data[top_part])]
            else:
                order = np.argsort(-data)
            results.append([(self.s1_ids[idx_data[j]], float(data[j])) for j in order])
        return results


class PrecisionSieveBlocker:
    """
    Two-Stage Sieve:
    1. Tier 1: Deterministic exact string hash map (O(1) resolution).
    2. Tier 2: Sparse BM25 candidate retrieval for unresolved queries.
    """
    def __init__(self, top_k_bm25: int = 8, max_bucket_size: int = 20):
        self.top_k_bm25 = top_k_bm25
        self.max_bucket_size = max_bucket_size
        self.exact_comb_sieve: Dict[str, str] = {}  # Single unique exact match
        self.exact_name_index: Dict[str, List[str]] = {}
        self.exact_addr_index: Dict[str, List[str]] = {}
        self.bm25 = SparseBM25Retriever(max_features=80000)
        self.s1_ids: List[str] = []

    def fit(self, s1_df: pd.DataFrame):
        if "comb_norm" not in s1_df.columns:
            s1_df = create_normalized_dataframe(s1_df)

        self.s1_ids = s1_df["entity_id"].tolist()
        name_keys = s1_df["name_norm"].tolist()
        addr_keys = s1_df["addr_norm"].tolist()
        comb_keys = s1_df["comb_norm"].tolist()

        self.exact_comb_sieve.clear()
        self.exact_name_index.clear()
        self.exact_addr_index.clear()

        # Build exact sieve
        comb_counts = pd.Series(comb_keys).value_counts()
        for c, eid in zip(comb_keys, self.s1_ids):
            if c and comb_counts.get(c, 0) == 1:
                self.exact_comb_sieve[c] = eid

        for n, eid in zip(name_keys, self.s1_ids):
            if n:
                b = self.exact_name_index.setdefault(n, [])
                if len(b) < self.max_bucket_size:
                    b.append(eid)

        for a, eid in zip(addr_keys, self.s1_ids):
            if a:
                b = self.exact_addr_index.setdefault(a, [])
                if len(b) < self.max_bucket_size:
                    b.append(eid)

        self.bm25.fit(comb_keys, self.s1_ids)
        logger.info(f"[BLOCKING] Sieve fitted: {len(self.exact_comb_sieve):,} unique exact keys indexed.")

    def resolve_tier1_exact(self, query_combs: List[str]) -> List[Optional[str]]:
        """Returns direct exact S1 match ID if present in unique Tier 1 sieve, else None."""
        return [self.exact_comb_sieve.get(c, None) for c in query_combs]

    def generate_candidates(self, query_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generates candidate (s1_id, bm25_score) pairs for queries.
        Returns: {query_id: [(s1_id, bm25_score), ...]}
        """
        if "comb_norm" not in query_df.columns:
            query_df = create_normalized_dataframe(query_df)

        q_ids = query_df["entity_id"].tolist()
        q_names = query_df["name_norm"].tolist()
        q_addrs = query_df["addr_norm"].tolist()
        q_combs = query_df["comb_norm"].tolist()

        bm25_cands = self.bm25.retrieve_batch(q_combs, top_k=self.top_k_bm25)
        candidates_map: Dict[str, List[Tuple[str, float]]] = {}

        for idx, qid in enumerate(q_ids):
            seen: Set[str] = set()
            cand_list: List[Tuple[str, float]] = []

            # 1. Exact Name & Addr Candidates
            for m in self.exact_name_index.get(q_names[idx], []):
                if m not in seen:
                    seen.add(m)
                    cand_list.append((m, 10.0))

            for m in self.exact_addr_index.get(q_addrs[idx], []):
                if len(cand_list) >= self.top_k_bm25:
                    break
                if m not in seen:
                    seen.add(m)
                    cand_list.append((m, 8.0))

            # 2. BM25 Candidates
            for s1_cand, sc in bm25_cands[idx]:
                if len(cand_list) >= self.top_k_bm25:
                    break
                if s1_cand not in seen:
                    seen.add(s1_cand)
                    cand_list.append((s1_cand, sc))

            candidates_map[qid] = cand_list

        return candidates_map
