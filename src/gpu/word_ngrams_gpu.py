import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from typing import Dict, Tuple
import os
from src.utils.load_cuda_kernels import load_cuda_kernels
from itertools import groupby


class WordNgramGPU:
    def __init__(self):
        # loads the kernel from the .cu file
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
        self.mod = load_cuda_kernels(kernel_path)
        self.map_kernel = self.mod.get_function("word_ngram_map_kernel")
        self.reduce_kernel = self.mod.get_function("word_ngram_reduce_kernel")
    
    # Compute word N-grams using CUDA with Map-Sort-Reduce strategy. Returns a dictionary with N-gram counts.
    def compute_word_ngrams_gpu(
        self, 
        word_ids: np.ndarray, 
        n: int, 
        vocab_size: int,
        id_to_word: Dict[int, str]
    ) -> Dict[Tuple[str, ...], int]:

        num_tokens = len(word_ids)
        
        if num_tokens < n:
            return {}
        
        num_ngrams = num_tokens - n + 1
        
        # allocate and transfer data to GPU
        word_ids_gpu = gpuarray.to_gpu(word_ids)
        ngram_ids_gpu = gpuarray.empty(num_ngrams, dtype=np.uint64)
        
        # FIXME: configure grid and block
        threads_per_block = 256
        blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        # FIXME: launch kernel
        self.map_kernel(
            word_ids_gpu.gpudata,
            np.uint32(num_tokens),
            np.uint32(n),
            np.uint32(vocab_size),
            ngram_ids_gpu.gpudata,
            block=(threads_per_block, 1, 1),
            grid=(blocks, 1)
        )
        
        # synchronize
        cuda.Context.synchronize()
        
        # sort n-gram IDs on CPU
        ngram_ids_cpu = ngram_ids_gpu.get()
        ngram_ids_sorted = np.sort(ngram_ids_cpu)
        
        # counts unique n-grams
        result_counts = {}
        for ngram_id, group in groupby(ngram_ids_sorted):
            count = len(list(group))
            
            # reconstruct n-gram words from ngram_id
            ngram_word_ids = []
            temp_id = ngram_id
            for _ in range(n):
                ngram_word_ids.append(int(temp_id % vocab_size))
                temp_id //= vocab_size
            ngram_word_ids.reverse()
            
            ngram_words = tuple(id_to_word[wid] for wid in ngram_word_ids)
            result_counts[ngram_words] = count
        
        return result_counts