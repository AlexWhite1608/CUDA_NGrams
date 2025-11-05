import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from typing import Dict
import os
from src.utils.load_cuda_kernels import load_cuda_kernels
from src.utils.text_processing import text_to_bytes


class CharNgramGPU:
    def __init__(self, algorithm: str = "v1"):
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
        self.mod = load_cuda_kernels(kernel_path)
        self.algorithm = algorithm
        
        if algorithm == "v1":   # v1: Baseline global atomic
            self.kernel = self.mod.get_function("char_ngram_kernel")
        elif algorithm == "v2":  # v2: Private histograms + reduce
            self.kernel_private = self.mod.get_function("char_ngram_kernel_private")
            self.kernel_reduce = self.mod.get_function("reduce_histograms")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def compute_char_ngrams_gpu(self, text: str, n: int) -> Dict[str, int]:
        if self.algorithm == "v1":
            return self._compute_v1(text, n)
        elif self.algorithm == "v2":
            return self._compute_v2(text, n)
    
    # A-v1: Baseline with global atomic updates
    def _compute_v1(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        hist_size = 256 ** n
        
        text_gpu = gpuarray.to_gpu(text_bytes)
        histogram_gpu = gpuarray.zeros(hist_size, dtype=np.uint32)
        
        threads_per_block = 256
        num_ngrams = text_length - n + 1
        blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        self.kernel(
            text_gpu.gpudata,
            np.uint32(text_length),
            np.uint32(n),
            histogram_gpu.gpudata,
            np.uint32(hist_size),
            block=(threads_per_block, 1, 1),
            grid=(blocks, 1)
        )
        
        cuda.Context.synchronize()
        histogram_cpu = histogram_gpu.get()
        
        return self._build_result_dict(histogram_cpu, n)
    
    # A-v2: Private histograms + reduce
    def _compute_v2(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        hist_size = 256 ** n
        
        text_gpu = gpuarray.to_gpu(text_bytes)
        
        threads_per_block = 256
        num_ngrams = text_length - n + 1
        num_blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        # Limit number of private histograms by grouping multiple blocks to share same private histogram
        MAX_PRIVATE_HISTS = 256  # Adjustable based on GPU memory
        num_private_hists = min(num_blocks, MAX_PRIVATE_HISTS)
        
        # Calculate how many CUDA blocks share each private histogram
        blocks_per_hist = (num_blocks + num_private_hists - 1) // num_private_hists
        
        print(f"[A-v2 Memory] num_blocks={num_blocks}, private_hists={num_private_hists}, blocks_per_hist={blocks_per_hist}")
        print(f"[A-v2 Memory] Allocating {num_private_hists * hist_size * 4 / (1024**2):.2f} MB")
        
        # Calculate private histograms
        private_histograms_gpu = gpuarray.zeros(
            num_private_hists * hist_size, 
            dtype=np.uint32
        )
        
        self.kernel_private(
            text_gpu.gpudata,
            np.uint32(text_length),
            np.uint32(n),
            private_histograms_gpu.gpudata,
            np.uint32(hist_size),
            np.uint32(num_private_hists),  
            block=(threads_per_block, 1, 1),
            grid=(num_blocks, 1)
        )
        
        cuda.Context.synchronize()
        
        # Reduce operation
        global_histogram_gpu = gpuarray.zeros(hist_size, dtype=np.uint32)
        
        reduce_threads = 256
        reduce_blocks = (hist_size + reduce_threads - 1) // reduce_threads
        
        self.kernel_reduce(
            private_histograms_gpu.gpudata,
            global_histogram_gpu.gpudata,
            np.uint32(num_private_hists),  
            np.uint32(hist_size),
            block=(reduce_threads, 1, 1),
            grid=(reduce_blocks, 1)
        )
        
        cuda.Context.synchronize()
        histogram_cpu = global_histogram_gpu.get()
        
        return self._build_result_dict(histogram_cpu, n)
    
    #FIXME: Shared method to build result dictionary from histogram
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