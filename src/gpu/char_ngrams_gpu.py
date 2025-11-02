import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from typing import Dict
import os
from utils.load_cuda_kernels import load_cuda_kernels
from utils.text_processing import text_to_bytes


class CharNgramGPU:
    def __init__(self):
        
        # loads the kernel from the .cu file
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cu')
        self.mod = load_cuda_kernels(kernel_path)
        self.kernel = self.mod.get_function("char_ngram_kernel")
    
    # Compute character N-grams using CUDA. Returns a dictionary with N-gram counts.
    def compute_char_ngrams_gpu(self, text: str, n: int) -> Dict[str, int]:

        # convert text to numpy array of bytes
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        # histogram size: 256^n for n-grams of size n. Number of possible n-grams
        hist_size = 256 ** n
        
        # gpu memory allocation
        text_gpu = gpuarray.to_gpu(text_bytes)
        histogram_gpu = gpuarray.zeros(hist_size, dtype=np.uint32)
        
        # FIXME: configure grid and block
        threads_per_block = 256
        num_ngrams = text_length - n + 1
        blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        # FIXME: launch kernel
        self.kernel(
            text_gpu.gpudata,
            np.uint32(text_length),
            np.uint32(n),
            histogram_gpu.gpudata,
            np.uint32(hist_size),
            block=(threads_per_block, 1, 1),
            grid=(blocks, 1)
        )
        
        # synchronize
        cuda.Context.synchronize()
        
        # copy histogram back to host
        histogram_cpu = histogram_gpu.get()
        
        # build result dictionary
        result = {}
        for flat_idx in range(hist_size):
            if histogram_cpu[flat_idx] > 0:
                # reconstruct n-gram from flat index
                ngram_chars = []
                temp_idx = flat_idx
                for _ in range(n):
                    ngram_chars.append(chr(temp_idx % 256))
                    temp_idx //= 256
                ngram = ''.join(reversed(ngram_chars))
                result[ngram] = int(histogram_cpu[flat_idx])
        
        return result