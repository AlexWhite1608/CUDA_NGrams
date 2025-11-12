import numpy as np
import cupy as cp
from typing import Dict
import os
from src.utils.text_processing import text_to_bytes

class CharNgramGPU:
    
    def __init__(self, algorithm: str = "v1", max_private_hists: int = 256):
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
        
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        self.kernel_v1_cupy = cp.RawKernel(kernel_code, "char_ngram_kernel")
        self.kernel_v2_private_cupy = cp.RawKernel(kernel_code, "char_ngram_kernel_private")
        self.kernel_v2_reduce_cupy = cp.RawKernel(kernel_code, "reduce_histograms")
        self.kernel_map_cupy = cp.RawKernel(kernel_code, "char_ngram_map_kernel")

        self.algorithm = algorithm
        self.max_private_hists = max_private_hists
        
        if algorithm not in ["v1", "v2", "B"]:
             raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def compute_char_ngrams_gpu(self, text: str, n: int) -> Dict[str, int]:
        if self.algorithm == "v1":
            return self._compute_v1(text, n)
        elif self.algorithm == "v2":
            return self._compute_v2(text, n)
        elif self.algorithm == "B":
            return self._compute_B(text, n)
    
    def _compute_v1(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        try:
            hist_size = 256 ** n
            text_gpu = cp.asarray(text_bytes, dtype=cp.uint8)
            histogram_gpu = cp.zeros(hist_size, dtype=cp.uint32)
        except Exception as e:
            print(f"[Algo-V1] Fallimento allocazione per n={n}. Dimensione: {256**n * 4 / (1024**3):.2f} GB")
            raise e
            
        threads_per_block = 256
        num_ngrams = text_length - n + 1
        blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        self.kernel_v1_cupy(
            (blocks,), 
            (threads_per_block,),
            (
                text_gpu,                   
                np.uint32(text_length),
                np.uint32(n),
                histogram_gpu,              
                np.uint32(hist_size)
            )
        )
        
        cp.cuda.Stream.null.synchronize()
        histogram_cpu = cp.asnumpy(histogram_gpu)
        
        return self._build_result_dict(histogram_cpu, n)
    
    def _compute_v2(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        hist_size = 256 ** n
        
        text_gpu = cp.asarray(text_bytes, dtype=cp.uint8)
        
        threads_per_block = 256
        num_ngrams = text_length - n + 1
        num_blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        MAX_PRIVATE_HISTS = self.max_private_hists 
        num_private_hists = min(num_blocks, MAX_PRIVATE_HISTS)
        blocks_per_hist = (num_blocks + num_private_hists - 1) // num_private_hists
        
        print(f"[A-v2 Memory] num_blocks={num_blocks}, private_hists={num_private_hists}, blocks_per_hist={blocks_per_hist}")
        
        total_mem_mb = num_private_hists * hist_size * 4 / (1024**2)
        print(f"[A-v2 Memory] Allocating {total_mem_mb:.2f} MB")
        
        if total_mem_mb > 20000: # limit for security
             raise MemoryError(f"Aborting V2: Allocation too large ({total_mem_mb:.2f} MB)")

        try:
            private_histograms_gpu = cp.zeros(
                num_private_hists * hist_size, 
                dtype=np.uint32
            )
        except Exception as e:
            print(f"[Algo-V2] Fallimento allocazione per n={n}. Dimensione: {total_mem_mb:.2f} MB")
            raise e
        
        self.kernel_v2_private_cupy(
            (num_blocks,), 
            (threads_per_block,),
            (
                text_gpu,
                np.uint32(text_length),
                np.uint32(n),
                private_histograms_gpu,
                np.uint32(hist_size),
                np.uint32(num_private_hists)
            )
        )
        
        cp.cuda.Stream.null.synchronize()
        
        global_histogram_gpu = cp.zeros(hist_size, dtype=np.uint32)
        
        reduce_threads = 256
        reduce_blocks = (hist_size + reduce_threads - 1) // reduce_threads
        
        self.kernel_v2_reduce_cupy(
            (reduce_blocks,), 
            (reduce_threads,),
            (
                private_histograms_gpu,
                global_histogram_gpu,
                np.uint32(num_private_hists),  
                np.uint32(hist_size)
            )
        )
        
        cp.cuda.Stream.null.synchronize()
        histogram_cpu = cp.asnumpy(global_histogram_gpu)
        
        return self._build_result_dict(histogram_cpu, n)
    
    def _compute_B(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        num_ngrams = text_length - n + 1
        
        text_gpu = cp.asarray(text_bytes, dtype=cp.uint8)
        
        print(f"[Algo-B] Phase 1: MAP - Generating {num_ngrams} n-gram IDs (n={n})")
        ngram_ids_gpu = cp.empty(num_ngrams, dtype=cp.uint64)
        
        threads_per_block = 256
        num_blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        self.kernel_map_cupy(
            (num_blocks,),
            (threads_per_block,),
            (
                text_gpu,
                np.uint32(text_length),
                np.uint32(n),
                ngram_ids_gpu
            )
        )
        cp.cuda.Stream.null.synchronize()
        
        print(f"[Algo-B] Phase 2: SORT - Sorting {num_ngrams} IDs")
        sorted_ngram_ids_gpu = cp.sort(ngram_ids_gpu)
        del ngram_ids_gpu 
        cp.cuda.Stream.null.synchronize()

        print(f"[Algo-B] Phase 3: REDUCE - Counting unique n-grams (CuPy)")
        
        unique_ngrams_cp, counts_cp = cp.unique(sorted_ngram_ids_gpu, return_counts=True)
        
        del sorted_ngram_ids_gpu 
        
        num_unique = unique_ngrams_cp.size
        print(f"[Algo-B] Found {num_unique} unique n-grams")
        
        if num_unique == 0:
            return {}
        
        unique_ngrams_cpu = cp.asnumpy(unique_ngrams_cp)
        counts_cpu = cp.asnumpy(counts_cp)
        
        del unique_ngrams_cp
        del counts_cp
        cp.get_default_memory_pool().free_all_blocks()
        
        result = {}
        for i in range(num_unique):
            flat_idx = int(unique_ngrams_cpu[i])
            count = int(counts_cpu[i])
            
            ngram_chars = []
            temp_idx = flat_idx
            for _ in range(n):
                ngram_chars.append(chr(temp_idx % 256))
                temp_idx //= 256
            ngram = ''.join(reversed(ngram_chars))
            result[ngram] = count
        
        return result

    def _build_result_dict(self, histogram_cpu: np.ndarray, n: int) -> Dict[str, int]:
        result = {}
        non_zero_indices = np.nonzero(histogram_cpu)[0]
        
        for flat_idx in non_zero_indices:
            count = int(histogram_cpu[flat_idx])
            ngram_chars = []
            temp_idx = int(flat_idx)
            for _ in range(n):
                ngram_chars.append(chr(temp_idx % 256))
                temp_idx //= 256
            ngram = ''.join(reversed(ngram_chars))
            result[ngram] = count
        
        return result