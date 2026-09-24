"""
GPU Acceleration & PyTorch Multi-GPU Module for Business Entity Resolution.
Leverages CUDA GPUs (e.g., dual NVIDIA T4 GPUs) for high-speed tensor matrix operations.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """Checks PyTorch CUDA GPU availability and returns device specs."""
    is_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if is_available else 0
    devices = []
    if is_available:
        for i in range(device_count):
            devices.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            })
    return {
        "cuda_available": is_available,
        "device_count": device_count,
        "devices": devices
    }


class PyTorchGPUCharTFIDFRetriever:
    """
    Ultra-fast PyTorch GPU Char-TFIDF Cosine Retriever.
    Runs matrix multiplications on CUDA devices (supports multi-GPU split).
    """
    def __init__(self, corpus_matrix: csr_matrix, s1_ids: List[str], devices: List[int] = [0, 1]):
        self.s1_ids = s1_ids
        self.num_corpus = corpus_matrix.shape[0]
        self.devices = devices if torch.cuda.device_count() >= len(devices) else list(range(torch.cuda.device_count()))
        
        logger.info(f"Loading corpus sparse matrix ({corpus_matrix.shape}) onto PyTorch GPU device cuda:0...")
        # Convert scipy csr_matrix to PyTorch sparse tensor
        coo = corpus_matrix.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        
        self.corpus_sparse_gpu = torch.sparse_coo_tensor(indices, values, shape).to("cuda:0").coalesce()
        self.corpus_dense_T_gpu = self.corpus_sparse_gpu.to_dense().T  # (vocab_size x corpus_size)

    def retrieve_top_k_multi_gpu(
        self,
        query_matrix: csr_matrix,
        top_k: int = 30,
        batch_size: int = 4000
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Distributes query retrieval across dual GPUs (cuda:0 and cuda:1).
        """
        logger.info(f"Executing Multi-GPU PyTorch Cosine Retrieval on {len(self.devices)} GPUs...")
        num_queries = query_matrix.shape[0]
        results = []
        
        # Split query batch across available GPUs
        num_gpus = max(1, len(self.devices))
        chunk_size = (num_queries + num_gpus - 1) // num_gpus
        
        for gpu_idx, dev_id in enumerate(self.devices):
            device_str = f"cuda:{dev_id}"
            start_q = gpu_idx * chunk_size
            end_q = min(start_q + chunk_size, num_queries)
            if start_q >= num_queries:
                break
                
            q_chunk_csr = query_matrix[start_q:end_q]
            coo = q_chunk_csr.tocoo()
            indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            shape = torch.Size(coo.shape)
            
            q_tensor_gpu = torch.sparse_coo_tensor(indices, values, shape).to(device_str).to_dense()
            corpus_T_gpu = self.corpus_dense_T_gpu.to(device_str)
            
            with torch.no_grad():
                # GPU Batch Dot Product Matrix Multiplication
                scores_gpu = torch.matmul(q_tensor_gpu, corpus_T_gpu) # (batch_size x corpus_size)
                
                # GPU Top-K calculation using torch.topk
                top_scores, top_indices = torch.topk(scores_gpu, k=top_k, dim=1)
                
                top_scores_cpu = top_scores.cpu().numpy()
                top_indices_cpu = top_indices.cpu().numpy()
                
            for i in range(len(top_scores_cpu)):
                row_res = []
                for rank, (score, idx) in enumerate(zip(top_scores_cpu[i], top_indices_cpu[i]), start=1):
                    if score <= 0.0:
                        break
                    row_res.append((self.s1_ids[idx], float(score), rank))
                results.append(row_res)
                
        return results
