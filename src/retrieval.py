"""
Retrieval Systems Module for Business Entity Resolution.

Provides unified, highly-scalable, CPU/GPU-aware retrievers:
1. CharTFIDFRetriever: Sub-word character n-gram cosine retriever (C++/OpenMP sparse dot products).
2. SparseBM25Retriever: Word-level inverted index BM25 retriever (scalable to millions of records).
3. ExactMatchRetriever: Fast dictionary hash-map inverted index for exact field matching.

All retrieval mechanisms are strictly identical between training, validation, and inference.
"""

import time
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Dict, List, Tuple, Set, Any, Optional
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class CharTFIDFRetriever:
    """
    Sub-word character n-gram TF-IDF cosine similarity retriever.
    Uses C++/OpenMP multi-threaded sparse dot products via SciPy.
    """
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 150000,
        min_df: int = 1,
        use_gpu: bool = False
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            dtype=np.float32,
            sublinear_tf=True
        )
        self.corpus_matrix_T: csr_matrix = None
        self.s1_ids: List[str] = []
        
        # Verify CUDA capability honestly
        cuda_ok = HAS_TORCH and torch.cuda.is_available() and use_gpu
        self.use_gpu = cuda_ok
        if self.use_gpu:
            logger.info("[RETRIEVAL] GPU acceleration enabled on: " + torch.cuda.get_device_name(0))
        else:
            logger.info("[RETRIEVAL] CharTFIDF using multi-threaded CPU sparse dot products (SciPy OpenMP).")

    def fit(self, corpus_texts: List[str], s1_ids: List[str]):
        """Fits TF-IDF vectorizer and builds transposed sparse corpus matrix."""
        start_t = time.time()
        logger.info(f"[RETRIEVAL] Fitting Char-TFIDF ({self.ngram_range}) on {len(corpus_texts):,} reference texts...")
        corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.corpus_matrix_T = corpus_matrix.T.tocsr()
        self.s1_ids = list(s1_ids)
        elapsed = time.time() - start_t
        logger.info(
            f"[RETRIEVAL] Char-TFIDF fitted in {elapsed:.2f}s. Vocab size: {len(self.vectorizer.vocabulary_):,}, "
            f"Matrix: {corpus_matrix.shape[0]:,}x{corpus_matrix.shape[1]:,} ({corpus_matrix.nnz:,} non-zeros)"
        )

    def retrieve_top_k(
        self,
        query_texts: List[str],
        top_k: int = 20,
        batch_size: int = 2000
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieves top_k reference entities for each query text.
        Returns list of [(s1_id, cosine_score, rank), ...] per query.
        """
        num_queries = len(query_texts)
        if top_k <= 0 or num_queries == 0:
            return [[] for _ in range(num_queries)]

        if self.corpus_matrix_T is None:
            raise ValueError("Retriever has not been fitted.")
            
        results: List[List[Tuple[str, float, int]]] = []
        start_t = time.time()
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_texts[start_idx:end_idx]
            
            # Sparse batch transform
            q_matrix = self.vectorizer.transform(q_batch)
            # Dot product against transposed corpus: (batch_size x corpus_size)
            scores_batch = q_matrix.dot(self.corpus_matrix_T)
            indptr = scores_batch.indptr
            s_data = scores_batch.data
            s_indices = scores_batch.indices
            
            for row_idx in range(scores_batch.shape[0]):
                start = indptr[row_idx]
                end = indptr[row_idx + 1]
                if start == end:
                    results.append([])
                    continue
                
                data = s_data[start:end]
                indices = s_indices[start:end]
                
                if len(data) <= top_k:
                    order = np.argsort(-data)
                else:
                    part = np.argpartition(data, -top_k)[-top_k:]
                    order = part[np.argsort(-data[part])]
                    
                row_cands = []
                for rank, j in enumerate(order, start=1):
                    sc = float(data[j])
                    if sc <= 0.001:
                        break
                    row_cands.append((self.s1_ids[indices[j]], sc, rank))
                results.append(row_cands)
                
        elapsed = time.time() - start_t
        logger.info(f"[RETRIEVAL] Char-TFIDF retrieved top-{top_k} for {num_queries:,} queries in {elapsed:.2f}s ({num_queries/max(elapsed, 0.001):.0f} q/s)")
        return results


BUSINESS_STOP_WORDS = [
    # Corporate entity suffixes & abbreviations
    "limited", "private", "ltd", "pvt", "llc", "inc", "corp", "corporation",
    "company", "co", "enterprises", "enterprise", "group", "holdings", "holding",
    "partners", "associates", "industries", "industry", "ventures", "venture",
    "solutions", "services", "service", "international", "intl", "llp", "plc",
    "gmbh", "sa", "srl", "bv", "ag",
    # Common address designators & noise
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "suite", "ste", "floor", "fl", "building", "bldg", "near", "opp", "opposite",
    "post", "box", "po", "highway", "hwy", "block", "blk", "sector", "sec",
    "plot", "shop", "flat", "room", "no", "unit",
    # Common English prepositions & conjunctions
    "the", "and", "of", "in", "for", "at", "by", "to", "on", "from", "with"
]


class SparseBM25Retriever:
    """
    Fast BM25 inverted-index retriever using sparse matrix dot products.
    Scales to millions of documents and queries with C++/OpenMP acceleration.
    Calculates BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))).
    """
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int = 100000,
        min_df: int = 1,
        max_df: float = 0.25,
        stop_words: Optional[List[str]] = None
    ):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = list(BUSINESS_STOP_WORDS) if stop_words is None else stop_words
        self.vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            max_features=max_features,
            min_df=min_df,
            max_df=max_df if max_df < 1.0 else 1.0,
            stop_words=self.stop_words,
            dtype=np.float32
        )
        self.bm25_matrix_T: csr_matrix = None
        self.s1_ids: List[str] = []

    def fit(self, corpus_texts: List[str], s1_ids: List[str]):
        """Fits count vectorizer, computes BM25 weights, and transposes matrix."""
        start_t = time.time()
        logger.info(f"[RETRIEVAL] Fitting Sparse BM25 (k1={self.k1}, b={self.b}) on {len(corpus_texts):,} reference texts...")
        
        # Word count matrix: (N, V)
        try:
            X = self.vectorizer.fit_transform(corpus_texts)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                logger.warning("[RETRIEVAL] max_df/stop_words resulted in empty vocabulary; falling back to basic vectorizer")
                self.vectorizer = CountVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    max_features=self.max_features,
                    min_df=self.min_df,
                    max_df=1.0,
                    stop_words=None,
                    dtype=np.float32
                )
                X = self.vectorizer.fit_transform(corpus_texts)
            else:
                raise e
                
        N, V = X.shape
        
        # Document frequencies per term
        df = np.bincount(X.indices, minlength=V)
        # BM25 probabilistic IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        
        # Document lengths
        doc_lengths = np.array(X.sum(axis=1)).flatten().astype(np.float32)
        avg_doc_len = float(np.mean(doc_lengths)) if len(doc_lengths) > 0 else 1.0
        
        # Compute BM25 transformed term weights for non-zero elements
        rows, cols = X.nonzero()
        data = X.data.astype(np.float32)
        
        len_norm = (1.0 - self.b + self.b * (doc_lengths[rows] / avg_doc_len)).astype(np.float32)
        tf_component = (data * (self.k1 + 1.0)) / (data + self.k1 * len_norm)
        bm25_data = tf_component * idf[cols]
        
        bm25_matrix = csr_matrix((bm25_data, (rows, cols)), shape=(N, V), dtype=np.float32)
        self.bm25_matrix_T = bm25_matrix.T.tocsr()
        self.s1_ids = list(s1_ids)
        
        elapsed = time.time() - start_t
        logger.info(
            f"[RETRIEVAL] Sparse BM25 fitted in {elapsed:.2f}s. Vocab size: {len(self.vectorizer.vocabulary_):,}, "
            f"Matrix: {N:,}x{V:,} ({bm25_matrix.nnz:,} non-zeros)"
        )

    def retrieve_top_k(
        self,
        query_texts: List[str],
        top_k: int = 20,
        batch_size: int = 2000
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieves top_k reference entities for each query text using BM25 scoring.
        Returns list of [(s1_id, bm25_score, rank), ...] per query.
        """
        num_queries = len(query_texts)
        if top_k <= 0 or num_queries == 0:
            return [[] for _ in range(num_queries)]

        if self.bm25_matrix_T is None:
            raise ValueError("Retriever has not been fitted.")
            
        results: List[List[Tuple[str, float, int]]] = []
        start_t = time.time()
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_texts[start_idx:end_idx]
            
            # Binary term occurrence in query
            q_matrix = self.vectorizer.transform(q_batch)
            q_matrix.data = np.ones_like(q_matrix.data, dtype=np.float32)
            
            # Sparse dot product
            scores_batch = q_matrix.dot(self.bm25_matrix_T)
            indptr = scores_batch.indptr
            s_data = scores_batch.data
            s_indices = scores_batch.indices
            
            for row_idx in range(scores_batch.shape[0]):
                start = indptr[row_idx]
                end = indptr[row_idx + 1]
                if start == end:
                    results.append([])
                    continue
                
                data = s_data[start:end]
                indices = s_indices[start:end]
                
                if len(data) <= top_k:
                    order = np.argsort(-data)
                elif len(data) > 2000:
                    # When candidate list is large, threshold to candidates with score > 0.2
                    # to keep argpartition fast and avoid multi-second pure-Python bottlenecks
                    mask = data > 0.2
                    if np.count_nonzero(mask) >= top_k:
                        sub_data = data[mask]
                        sub_indices = indices[mask]
                        part = np.argpartition(sub_data, -top_k)[-top_k:]
                        order = part[np.argsort(-sub_data[part])]
                        row_cands = []
                        for rank, j in enumerate(order, start=1):
                            sc = float(sub_data[j])
                            if sc <= 0.001:
                                break
                            row_cands.append((self.s1_ids[sub_indices[j]], sc, rank))
                        results.append(row_cands)
                        continue
                    else:
                        part = np.argpartition(data, -top_k)[-top_k:]
                        order = part[np.argsort(-data[part])]
                else:
                    part = np.argpartition(data, -top_k)[-top_k:]
                    order = part[np.argsort(-data[part])]
                    
                row_cands = []
                for rank, j in enumerate(order, start=1):
                    sc = float(data[j])
                    if sc <= 0.001:
                        break
                    row_cands.append((self.s1_ids[indices[j]], sc, rank))
                results.append(row_cands)
                
        elapsed = time.time() - start_t
        logger.info(f"[RETRIEVAL] Sparse BM25 retrieved top-{top_k} for {num_queries:,} queries in {elapsed:.2f}s ({num_queries/max(elapsed, 0.001):.0f} q/s)")
        return results


class ExactMatchIndex:
    """
    Inverted hash-map index for exact string matching on reference entities.
    Supports lookup with frequency capping to avoid massive Cartesian products on generic terms.
    """
    def __init__(self, max_bucket_size: int = 100):
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, Set[str]] = {}

    def fit(self, texts: List[str], s1_ids: List[str]):
        """Indexes non-empty strings to set of S1 IDs."""
        self.index.clear()
        for text, s1_id in zip(texts, s1_ids):
            if text and len(text) >= 2:
                self.index.setdefault(text, set()).add(s1_id)
        logger.info(f"[RETRIEVAL] ExactMatchIndex built with {len(self.index):,} unique keys.")

    def lookup(self, text: str) -> Set[str]:
        """Returns matching S1 IDs if key exists and bucket is within safe size limit."""
        if not text:
            return set()
        bucket = self.index.get(text)
        if not bucket or len(bucket) > self.max_bucket_size:
            return set()
        return bucket
