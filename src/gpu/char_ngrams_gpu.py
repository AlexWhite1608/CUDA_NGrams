import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from typing import Dict
import os
from src.utils.load_cuda_kernels import load_cuda_kernels
from src.utils.text_processing import text_to_bytes
import cupy as cp


class CharNgramGPU:
    def __init__(self, algorithm: str = "v1"):
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
        self.mod = load_cuda_kernels(kernel_path)
        self.algorithm = algorithm
        
        if algorithm == "v1":
            self.kernel = self.mod.get_function("char_ngram_kernel")
        elif algorithm == "v2":
            self.kernel_private = self.mod.get_function("char_ngram_kernel_private")
            self.kernel_reduce = self.mod.get_function("reduce_histograms")
        elif algorithm == "B":
            self.kernel_map = self.mod.get_function("char_ngram_map_kernel")
        else:
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
        
        MAX_PRIVATE_HISTS = 256
        num_private_hists = min(num_blocks, MAX_PRIVATE_HISTS)
        blocks_per_hist = (num_blocks + num_private_hists - 1) // num_private_hists
        
        print(f"[A-v2 Memory] num_blocks={num_blocks}, private_hists={num_private_hists}, blocks_per_hist={blocks_per_hist}")
        print(f"[A-v2 Memory] Allocating {num_private_hists * hist_size * 4 / (1024**2):.2f} MB")
        
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
    
    def _compute_B(self, text: str, n: int) -> Dict[str, int]:
        text_bytes = text_to_bytes(text)
        text_length = len(text_bytes)
        
        if text_length < n:
            return {}
        
        num_ngrams = text_length - n + 1
        
        text_gpu = gpuarray.to_gpu(text_bytes)
        
        # ===== PHASE 1: MAP =====
        print(f"[Algo-B] Phase 1: MAP - Generating {num_ngrams} n-gram IDs")
        ngram_ids_gpu = gpuarray.empty(num_ngrams, dtype=np.uint64)
        
        threads_per_block = 256
        num_blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
        
        self.kernel_map(
            text_gpu.gpudata,
            np.uint32(text_length),
            np.uint32(n),
            ngram_ids_gpu.gpudata,
            block=(threads_per_block, 1, 1),
            grid=(num_blocks, 1)
        )
        cuda.Context.synchronize()
        
        # ===== PHASE 2: SORT =====
        print(f"[Algo-B] Phase 2: SORT - Sorting {num_ngrams} IDs")
        sorted_ngram_ids_gpu = self._sort_ngram_ids_cupy(ngram_ids_gpu, num_ngrams)
        cuda.Context.synchronize()
        
        # ===== PHASE 3: REDUCE =====
        print(f"[Algo-B] Phase 3: REDUCE - Counting unique n-grams (CuPy)")
        result = self._reduce_with_cupy(sorted_ngram_ids_gpu, num_ngrams, n)
        
        return result
    
    def _sort_ngram_ids_cupy(self, ngram_ids_gpu: gpuarray.GPUArray, num_ngrams: int) -> gpuarray.GPUArray:
        """Sort n-gram IDs using CuPy - copia i dati per evitare conflitti di contesto."""
        try:
            print(f"[Algo-B] Sorting with CuPy ({num_ngrams} elements)...")
            
            # Sincronizza PyCUDA
            cuda.Context.synchronize()
            
            # STRATEGIA: Copia D->D da PyCUDA a CuPy
            # 1. Alloca array CuPy nativo
            ngram_ids_cupy = cp.empty(num_ngrams, dtype=cp.uint64)
            
            # 2. Copia device-to-device (veloce, ~2ms per 12M elementi)
            cuda.memcpy_dtod(
                ngram_ids_cupy.data.ptr,
                ngram_ids_gpu.gpudata,
                ngram_ids_gpu.nbytes
            )
            
            # 3. Ordina con CuPy (array nativo CuPy, nessun problema di contesto)
            sorted_cupy = cp.sort(ngram_ids_cupy)
            
            # 4. Alloca array PyCUDA per il risultato
            sorted_gpu = gpuarray.empty(num_ngrams, dtype=np.uint64)
            
            # 5. Copia device-to-device da CuPy a PyCUDA
            cuda.memcpy_dtod(
                sorted_gpu.gpudata,
                sorted_cupy.data.ptr,
                sorted_cupy.nbytes
            )
            
            # 6. Pulisci memoria CuPy
            del ngram_ids_cupy
            del sorted_cupy
            cp.get_default_memory_pool().free_all_blocks()
            
            print("[Algo-B] ✓ CuPy sort successful")
            cuda.Context.synchronize()
            
            return sorted_gpu
            
        except Exception as e:
            print(f"[Algo-B] CuPy sort failed: {type(e).__name__}: {e}")
            print(f"[Algo-B] Falling back to CPU NumPy sort...")
            
            # Fallback su CPU
            import time
            ngram_ids_cpu = ngram_ids_gpu.get()
            t0 = time.perf_counter()
            sorted_cpu = np.sort(ngram_ids_cpu)
            sort_time = time.perf_counter() - t0
            print(f"[Algo-B] CPU sort completed in {sort_time:.3f}s")
            
            return gpuarray.to_gpu(sorted_cpu)
    
    def _reduce_with_cupy(self, sorted_ngram_ids_gpu: gpuarray.GPUArray, num_ngrams: int, n: int) -> Dict[str, int]:
        """Esegui la riduzione (unique + counts) usando CuPy."""
        try:
            # Sincronizza PyCUDA
            cuda.Context.synchronize()
            
            # STRATEGIA: Copia D->D da PyCUDA a CuPy per evitare conflitti
            # 1. Alloca array CuPy nativo
            sorted_cupy = cp.empty(num_ngrams, dtype=cp.uint64)
            
            # 2. Copia device-to-device da PyCUDA a CuPy
            cuda.memcpy_dtod(
                sorted_cupy.data.ptr,
                sorted_ngram_ids_gpu.gpudata,
                sorted_ngram_ids_gpu.nbytes
            )
            
            # 3. Esegui unique su CuPy (array nativo CuPy)
            unique_ngrams_cp, counts_cp = cp.unique(sorted_cupy, return_counts=True)
            
            num_unique = unique_ngrams_cp.size
            print(f"[Algo-B] Found {num_unique} unique n-grams")
            
            if num_unique == 0:
                del sorted_cupy
                cp.get_default_memory_pool().free_all_blocks()
                return {}
            
            # 4. Trasferisci risultati alla CPU (sono piccoli)
            unique_ngrams_cpu = cp.asnumpy(unique_ngrams_cp)
            counts_cpu = cp.asnumpy(counts_cp)
            
            # 5. Pulisci memoria CuPy
            del sorted_cupy
            del unique_ngrams_cp
            del counts_cp
            cp.get_default_memory_pool().free_all_blocks()
            
            # Sincronizza
            cuda.Context.synchronize()
            
            # 6. Costruisci dizionario risultato
            result = {}
            for i in range(num_unique):
                flat_idx = int(unique_ngrams_cpu[i])
                count = int(counts_cpu[i])
                
                # Decodifica l'ID in n-gram
                ngram_chars = []
                temp_idx = flat_idx
                for _ in range(n):
                    ngram_chars.append(chr(temp_idx % 256))
                    temp_idx //= 256
                ngram = ''.join(reversed(ngram_chars))
                result[ngram] = count
            
            return result
            
        except Exception as e:
            print(f"[Algo-B] Error during CuPy reduce: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
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