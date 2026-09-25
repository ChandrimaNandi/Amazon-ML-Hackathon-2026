"""
GPU Acceleration & PyTorch Multi-GPU Module for Business Entity Resolution.
Optimized for NVIDIA T4 architecture (dual T4 GPUs on Kaggle) with FP16 Tensor Cores.
Provides automatic device discovery (0, 1, or 2 GPUs) and explicit CPU fallback.
"""

import os
import gc
import logging
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.config import get_hardware_info, get_available_devices, release_memory

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """Checks PyTorch CUDA GPU availability and returns device specs."""
    if not HAS_TORCH:
        return {"cuda_available": False, "device_count": 0, "devices": []}
    
    is_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if is_available else 0
    devices = []
    if is_available:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })
    return {
        "cuda_available": is_available,
        "device_count": device_count,
        "devices": devices
    }


class MultiGPUTensorScorer:
    """
    Distributes dense tensor scoring operations across available CUDA devices (e.g. dual T4 GPUs).
    Uses FP16 for NVIDIA T4 Tensor Core acceleration with explicit CPU fallback.
    """
    def __init__(self, devices: Optional[List[Any]] = None):
        self.devices = devices or get_available_devices()
        self.use_cuda = HAS_TORCH and torch.cuda.is_available() and len(self.devices) > 0 and self.devices[0].type == "cuda"
        if self.use_cuda:
            dev_names = [torch.cuda.get_device_name(d) for d in self.devices]
            logger.info(f"[GPU SCORER] Initialized on {len(self.devices)} GPU(s): {', '.join(dev_names)} with FP16")
        else:
            logger.info("[GPU SCORER] CUDA not available or CPU requested. Operating in CPU mode.")

    def score_dense_pairs(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes dot product or weighted scoring between query and candidate vectors.
        Splits batches evenly across available GPUs if CUDA is available.
        """
        n_pairs = len(query_vectors)
        if n_pairs == 0:
            return np.array([], dtype=np.float32)

        if self.use_cuda:
            try:
                num_gpus = len(self.devices)
                chunk_size = (n_pairs + num_gpus - 1) // num_gpus
                scored_chunks = []

                for dev_idx, dev in enumerate(self.devices):
                    start_i = dev_idx * chunk_size
                    end_i = min(start_i + chunk_size, n_pairs)
                    if start_i >= n_pairs:
                        break

                    q_sub = query_vectors[start_i:end_i]
                    c_sub = candidate_vectors[start_i:end_i]

                    # Use FP16 for T4 Tensor Core efficiency
                    with torch.no_grad():
                        q_t = torch.as_tensor(q_sub, dtype=torch.float16, device=dev)
                        c_t = torch.as_tensor(c_sub, dtype=torch.float16, device=dev)
                        if weights is not None:
                            w_t = torch.as_tensor(weights, dtype=torch.float16, device=dev)
                            scores = torch.sum(q_t * c_t * w_t, dim=1)
                        else:
                            scores = torch.sum(q_t * c_t, dim=1)
                        scored_chunks.append(scores.cpu().numpy().astype(np.float32))

                return np.concatenate(scored_chunks, axis=0)

            except (RuntimeError, Exception) as e:
                logger.warning(f"[GPU FALLBACK: CUDA operation failed ({e}). Releasing VRAM and falling back to CPU]")
                release_memory()

        # Robust CPU Fallback
        if weights is not None:
            return np.sum(query_vectors * candidate_vectors * weights, axis=1, dtype=np.float32)
        return np.sum(query_vectors * candidate_vectors, axis=1, dtype=np.float32)


class GPUCosineTopK:
    """
    High-throughput top-k cosine similarity search for dense vectors.
    Distributes queries across 1 or 2 GPUs with FP16 Tensor Cores and automatic CPU fallback.
    """
    def __init__(self, corpus_embeddings: np.ndarray, s1_ids: List[str], devices: Optional[List[Any]] = None):
        self.s1_ids = list(s1_ids)
        self.devices = devices or get_available_devices()
        self.use_cuda = HAS_TORCH and torch.cuda.is_available() and len(self.devices) > 0 and self.devices[0].type == "cuda"
        self.corpus_cpu = corpus_embeddings.astype(np.float32)
        self.corpus_gpu_shards = {}

        if self.use_cuda:
            try:
                # Distribute corpus to devices in FP16
                for dev in self.devices:
                    self.corpus_gpu_shards[dev] = torch.as_tensor(
                        self.corpus_cpu.T,
                        dtype=torch.float16,
                        device=dev
                    )
                logger.info(f"[GPU COSINE] Corpus loaded onto {len(self.devices)} GPU(s) in FP16.")
            except (RuntimeError, Exception) as e:
                logger.warning(f"[GPU FALLBACK: Could not load corpus onto GPU ({e}). Falling back to CPU.]")
                self.use_cuda = False
                self.corpus_gpu_shards.clear()
                release_memory()

    def query_top_k(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 25,
        batch_size: int = 4000
    ) -> List[List[Tuple[str, float, int]]]:
        """Queries top-k similar items for each query embedding."""
        n_queries = len(query_embeddings)
        results: List[List[Tuple[str, float, int]]] = []

        if self.use_cuda and self.corpus_gpu_shards:
            try:
                num_gpus = len(self.devices)
                chunk_size = (n_queries + num_gpus - 1) // num_gpus

                for dev_idx, dev in enumerate(self.devices):
                    start_i = dev_idx * chunk_size
                    end_i = min(start_i + chunk_size, n_queries)
                    if start_i >= n_queries:
                        break

                    q_sub = query_embeddings[start_i:end_i]
                    corpus_t = self.corpus_gpu_shards[dev]

                    for b_start in range(0, len(q_sub), batch_size):
                        b_end = min(b_start + batch_size, len(q_sub))
                        q_batch = q_sub[b_start:b_end]

                        with torch.no_grad():
                            q_t = torch.as_tensor(q_batch, dtype=torch.float16, device=dev)
                            # Matrix multiplication in FP16 on T4 Tensor Cores
                            sims = torch.matmul(q_t, corpus_t)
                            k_actual = min(top_k, sims.shape[1])
                            top_scores, top_indices = torch.topk(sims, k=k_actual, dim=1)

                            scores_np = top_scores.cpu().numpy().astype(np.float32)
                            indices_np = top_indices.cpu().numpy()

                        for row_i in range(len(scores_np)):
                            row_res = []
                            for rank, (sc, idx) in enumerate(zip(scores_np[row_i], indices_np[row_i]), start=1):
                                if sc <= 0.0:
                                    break
                                row_res.append((self.s1_ids[idx], float(sc), rank))
                            results.append(row_res)

                return results
            except (RuntimeError, Exception) as e:
                logger.warning(f"[GPU FALLBACK: GPU Cosine search failed ({e}). Falling back to CPU.]")
                release_memory()
                results.clear()

        # Safe CPU OpenMP / NumPy Fallback
        for b_start in range(0, n_queries, batch_size):
            b_end = min(b_start + batch_size, n_queries)
            q_batch = query_embeddings[b_start:b_end].astype(np.float32)
            sims = np.dot(q_batch, self.corpus_cpu.T)

            for row_i in range(len(sims)):
                row_scores = sims[row_i]
                if len(row_scores) <= top_k:
                    order = np.argsort(-row_scores)
                else:
                    part = np.argpartition(row_scores, -top_k)[-top_k:]
                    order = part[np.argsort(-row_scores[part])]

                row_res = []
                for rank, idx in enumerate(order, start=1):
                    sc = float(row_scores[idx])
                    if sc <= 0.0:
                        break
                    row_res.append((self.s1_ids[idx], sc, rank))
                results.append(row_res)

        return results


class PyTorchGPUCharTFIDFRetriever:
    """
    Char-TFIDF Cosine Retriever with automatic hardware scaling.
    For sparse matrices, avoids catastrophic dense VRAM allocation (e.g. 2.2M x 150k > 1 TB VRAM)
    by executing SciPy OpenMP sparse dot products on CPU while preserving GPU acceleration for dense tensors.
    """
    def __init__(self, corpus_matrix: csr_matrix, s1_ids: List[str], devices: Optional[List[int]] = None):
        self.s1_ids = list(s1_ids)
        self.num_corpus = corpus_matrix.shape[0]
        self.vocab_size = corpus_matrix.shape[1]
        self.corpus_matrix_T = corpus_matrix.T.tocsr()
        
        # Estimate dense VRAM requirement: rows * cols * 2 bytes (FP16)
        dense_size_bytes = self.num_corpus * self.vocab_size * 2
        dense_size_gb = dense_size_bytes / (1024 ** 3)
        
        # Check if safe for GPU VRAM (< 2.0 GB)
        can_fit_gpu = HAS_TORCH and torch.cuda.is_available() and dense_size_gb <= 2.0
        
        if can_fit_gpu:
            try:
                dev = torch.device("cuda:0")
                coo = corpus_matrix.tocoo()
                indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
                values = torch.tensor(coo.data, dtype=torch.float16)
                shape = torch.Size(coo.shape)
                sparse_t = torch.sparse_coo_tensor(indices, values, shape, device=dev).coalesce()
                self.corpus_dense_T_gpu = sparse_t.to_dense().T
                self.use_gpu = True
                logger.info(f"[GPU RETRIEVER] Small corpus ({dense_size_gb:.2f} GB) loaded to GPU cuda:0 in FP16.")
            except Exception as e:
                logger.info(f"[GPU FALLBACK: Sparse densification failed ({e}). Using SciPy OpenMP sparse dot products.]")
                self.use_gpu = False
                self.corpus_dense_T_gpu = None
                release_memory()
        else:
            logger.info(
                f"[GPU FALLBACK: Corpus size {self.num_corpus:,}x{self.vocab_size:,} would require "
                f"{dense_size_gb:.1f} GB dense VRAM. Utilizing SciPy OpenMP multi-threaded sparse dot products "
                f"for optimal throughput without CUDA OOM.]"
            )
            self.use_gpu = False
            self.corpus_dense_T_gpu = None

    def retrieve_top_k_multi_gpu(
        self,
        query_matrix: csr_matrix,
        top_k: int = 30,
        batch_size: int = 5000
    ) -> List[List[Tuple[str, float, int]]]:
        """Executes retrieval with dynamic hardware routing and zero VRAM crash risk."""
        num_queries = query_matrix.shape[0]
        results: List[List[Tuple[str, float, int]]] = []

        if self.use_gpu and self.corpus_dense_T_gpu is not None:
            try:
                for start_i in range(0, num_queries, batch_size):
                    end_i = min(start_i + batch_size, num_queries)
                    q_batch = query_matrix[start_i:end_i]
                    coo = q_batch.tocoo()
                    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
                    values = torch.tensor(coo.data, dtype=torch.float16)
                    q_dense = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), device=self.corpus_dense_T_gpu.device).to_dense()

                    with torch.no_grad():
                        sims = torch.matmul(q_dense, self.corpus_dense_T_gpu)
                        top_sc, top_idx = torch.topk(sims, k=min(top_k, sims.shape[1]), dim=1)
                        sc_np = top_sc.cpu().numpy().astype(np.float32)
                        idx_np = top_idx.cpu().numpy()

                    for row_i in range(len(sc_np)):
                        row_res = []
                        for rank, (sc, idx) in enumerate(zip(sc_np[row_i], idx_np[row_i]), start=1):
                            if sc <= 0.001:
                                break
                            row_res.append((self.s1_ids[idx], float(sc), rank))
                        results.append(row_res)

                return results
            except Exception as e:
                logger.warning(f"[GPU FALLBACK: GPU retrieval failed ({e}). Falling back to SciPy sparse dot product.]")
                release_memory()
                results.clear()

        # Multi-threaded SciPy OpenMP sparse dot product (fastest & safest for large vocabulary)
        for start_i in range(0, num_queries, batch_size):
            end_i = min(start_i + batch_size, num_queries)
            q_chunk = query_matrix[start_i:end_i]
            scores_chunk = q_chunk.dot(self.corpus_matrix_T)

            for row_idx in range(scores_chunk.shape[0]):
                row = scores_chunk.getrow(row_idx)
                if row.nnz == 0:
                    results.append([])
                    continue
                data = row.data
                indices = row.indices
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

        return results
