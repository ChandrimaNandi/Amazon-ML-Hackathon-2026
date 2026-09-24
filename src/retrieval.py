"""
Retrieval Systems Module for Business Entity Resolution.
Implements BM25, Character TF-IDF Cosine Similarity, and Exact Match retrievers.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple, Set, Any
import logging

logger = logging.getLogger(__name__)


class CharTFIDFRetriever:
    """
    Fast character n-gram TF-IDF cosine similarity retriever.
    Supports top-K candidate retrieval across large corpus using sparse matrix dot products.
    """
    def __init__(self, ngram_range=(3, 5), max_features=250000, min_df=2):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            dtype=np.float32
        )
        self.corpus_matrix: csr_matrix = None
        self.s1_ids: List[str] = []

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
        batch_size: int = 2000
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieves top_k candidates for each query text.
        Returns list of [(s1_id, score, rank), ...] for each query.
        """
        logger.info(f"Retrieving top {top_k} candidates for {len(query_texts):,} queries using Char-TFIDF...")
        start_t = time.time()
        
        query_matrix = self.vectorizer.transform(query_texts)
        num_queries = query_matrix.shape[0]
        results: List[List[Tuple[str, float, int]]] = []
        
        # Transpose corpus matrix for dot product
        corpus_T = self.corpus_matrix.T
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_matrix[start_idx:end_idx]
            
            # Compute cosine similarity dot product: (batch_size x corpus_size)
            scores_batch = q_batch.dot(corpus_T).toarray()
            
            for row_idx in range(scores_batch.shape[0]):
                scores = scores_batch[row_idx]
                if top_k >= len(scores):
                    top_indices = np.argsort(scores)[::-1]
                else:
                    # argpartition for top_k speedup
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
        logger.info(f"Char-TFIDF retrieval finished in {elapsed:.2f}s.")
        return results


class BM25Retriever:
    """
    BM25 retriever using rank_bm25.
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
