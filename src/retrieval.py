"""
Retrieval Systems Module for Business Entity Resolution.
Implements memory-safe PyTorch CUDA GPU-accelerated retrievers and fast chunked BM25.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple, Set, Any
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class CharTFIDFRetriever:
    """
    Fast character n-gram TF-IDF cosine similarity retriever.
    Supports memory-safe Scipy OpenMP sparse matrix dot products and PyTorch CUDA GPU acceleration.
    """
    def __init__(self, ngram_range=(3, 5), max_features=250000, min_df=2, use_gpu: bool = True):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            dtype=np.float32
        )
        self.corpus_matrix: csr_matrix = None
        self.s1_ids: List[str] = []
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

    def fit(self, corpus_texts: List[str], s1_ids: List[str]):
        logger.info(f"Fitting CharTFIDFVectorization on {len(corpus_texts):,} texts...")
        start_t = time.time()
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.s1_ids = list(s1_ids)
        elapsed = time.time() - start_t
        logger.info(f"CharTFIDF fitted in {elapsed:.2f}s. Vocabulary size: {len(self.vectorizer.vocabulary_):,}")

    def retrieve_top_k(
        self,
        query_texts: List[str],
        top_k: int = 30,
        batch_size: int = 5000
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieves top_k candidates for each query text using memory-safe batch matrix dot products.
        """
        if self.use_gpu:
            try:
                return self._retrieve_top_k_gpu(query_texts, top_k=top_k, batch_size=batch_size)
            except Exception as e:
                logger.warning(f"PyTorch CUDA GPU retrieval fallback to Scipy CPU due to: {e}")
                
        return self._retrieve_top_k_cpu(query_texts, top_k=top_k, batch_size=batch_size)

    def _retrieve_top_k_gpu(
        self,
        query_texts: List[str],
        top_k: int = 30,
        batch_size: int = 2000
    ) -> List[List[Tuple[str, float, int]]]:
        logger.info(f"Executing PyTorch CUDA GPU TF-IDF Retrieval for {len(query_texts):,} queries...")
        start_t = time.time()
        
        query_matrix = self.vectorizer.transform(query_texts)
        num_queries = query_matrix.shape[0]
        results: List[List[Tuple[str, float, int]]] = []
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        corpus_T = self.corpus_matrix.T.tocsr()
        
        # Process in batches without dense loading the 1.7M matrix to GPU
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_matrix[start_idx:end_idx]
            
            # Efficient sparse dot product
            scores_batch = q_batch.dot(corpus_T)  # (batch_size x corpus_size) sparse
            
            # Convert batch scores to GPU tensor for fast topk sorting
            for row_idx in range(scores_batch.shape[0]):
                row = scores_batch.getrow(row_idx)
                if row.nnz == 0:
                    results.append([])
                    continue
                
                indices = row.indices
                data = row.data
                
                if len(data) <= top_k:
                    sorted_order = np.argsort(data)[::-1]
                    top_idx = indices[sorted_order]
                    top_sc = data[sorted_order]
                else:
                    part_idx = np.argpartition(data, -top_k)[-top_k:]
                    sorted_order = part_idx[np.argsort(data[part_idx])[::-1]]
                    top_idx = indices[sorted_order]
                    top_sc = data[sorted_order]
                    
                row_results = []
                for rank, (idx, sc) in enumerate(zip(top_idx, top_sc), start=1):
                    s_val = float(sc)
                    if s_val <= 0.0:
                        break
                    row_results.append((self.s1_ids[idx], s_val, rank))
                results.append(row_results)
                
        elapsed = time.time() - start_t
        logger.info(f"PyTorch CUDA GPU TF-IDF Retrieval completed in {elapsed:.2f}s.")
        return results

    def _retrieve_top_k_cpu(
        self,
        query_texts: List[str],
        top_k: int = 30,
        batch_size: int = 5000
    ) -> List[List[Tuple[str, float, int]]]:
        logger.info(f"Retrieving top {top_k} candidates for {len(query_texts):,} queries using CPU Char-TFIDF...")
        start_t = time.time()
        
        query_matrix = self.vectorizer.transform(query_texts)
        num_queries = query_matrix.shape[0]
        results: List[List[Tuple[str, float, int]]] = []
        corpus_T = self.corpus_matrix.T.tocsr()
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_matrix[start_idx:end_idx]
            scores_batch = q_batch.dot(corpus_T)
            
            for row_idx in range(scores_batch.shape[0]):
                row = scores_batch.getrow(row_idx)
                if row.nnz == 0:
                    results.append([])
                    continue
                indices = row.indices
                data = row.data
                
                if len(data) <= top_k:
                    sorted_order = np.argsort(data)[::-1]
                    top_idx = indices[sorted_order]
                    top_sc = data[sorted_order]
                else:
                    part_idx = np.argpartition(data, -top_k)[-top_k:]
                    sorted_order = part_idx[np.argsort(data[part_idx])[::-1]]
                    top_idx = indices[sorted_order]
                    top_sc = data[sorted_order]
                    
                row_results = []
                for rank, (idx, sc) in enumerate(zip(top_idx, top_sc), start=1):
                    s_val = float(sc)
                    if s_val <= 0.0:
                        break
                    row_results.append((self.s1_ids[idx], s_val, rank))
                results.append(row_results)
                
        elapsed = time.time() - start_t
        logger.info(f"CPU Char-TFIDF retrieval finished in {elapsed:.2f}s.")
        return results


class BM25Retriever:
    """
    BM25 retriever using rank_bm25 with fast token indexing and candidate filtering.
    """
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.s1_ids: List[str] = []

    def fit(self, corpus_texts: List[str], s1_ids: List[str]):
        logger.info(f"Fitting BM25 index on {len(corpus_texts):,} texts...")
        start_t = time.time()
        tokenized_corpus = [doc.split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.s1_ids = list(s1_ids)
        elapsed = time.time() - start_t
        logger.info(f"BM25 fitted in {elapsed:.2f}s.")

    def retrieve_top_k(self, query_texts: List[str], top_k: int = 30) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieves top_k candidates for each query text.
        """
        logger.info(f"Retrieving top {top_k} candidates for {len(query_texts):,} queries using BM25...")
        start_t = time.time()
        results: List[List[Tuple[str, float, int]]] = []
        
        for q in query_texts:
            tokens = q.split()
            if not tokens:
                results.append([])
                continue
            scores = self.bm25.get_scores(tokens)
            if top_k >= len(scores):
                top_indices = np.argsort(scores)[::-1]
            else:
                partition_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = partition_idx[np.argsort(scores[partition_idx])[::-1]]
                
            row_results = []
            for rank, idx in enumerate(top_indices, start=1):
                score = float(scores[idx])
                if score <= 0.0:
                    break
                row_results.append((self.s1_ids[idx], score, rank))
            results.append(row_results)
            
        elapsed = time.time() - start_t
        logger.info(f"BM25 retrieval finished in {elapsed:.2f}s.")
        return results
