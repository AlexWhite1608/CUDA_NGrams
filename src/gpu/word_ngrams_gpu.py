# In word_ngrams_gpu.py

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from typing import Dict, Tuple
import os
from src.utils.load_cuda_kernels import load_cuda_kernels

# === IMPORTA NUMBA ===
try:
    from numba import cuda as numba_cuda
    HAS_NUMBA = True
except ImportError:
    print("ATTENZIONE: numba non trovato. Il sort su GPU non funzionerà.")
    print("Installa con: conda install numba")
    HAS_NUMBA = False

class WordNgramGPU:
    def __init__(self):
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
        self.mod = load_cuda_kernels(kernel_path)
        self.map_kernel = self.mod.get_function("word_ngram_map_kernel")
        self.reduce_kernel = self.mod.get_function("word_ngram_reduce_kernel")
        
        # Prova a caricare la funzione Thrust sort
        try:
            self.thrust_sort = self.mod.get_function("thrust_sort_uint64")
            self.has_thrust = True
            print("Debug: Thrust sort available")
        except:
            self.has_thrust = False
            print("Debug: Thrust sort not available, will use fallback")
    
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
        
        # Fallback a CPU se numba non disponibile
        if not HAS_NUMBA:
            print("Numba non disponibile, usando fallback CPU per il sort")
            return self._compute_cpu_fallback(word_ids, n, vocab_size, id_to_word)
        
        num_ngrams = num_tokens - n + 1
        
        try:
            # 1. Allocazione e Map (PyCUDA)
            print(f"Debug: Allocating GPU memory for {num_ngrams} ngrams")
            word_ids_gpu = gpuarray.to_gpu(word_ids.astype(np.uint32))
            ngram_ids_gpu = gpuarray.zeros(num_ngrams, dtype=np.uint64)
            
            threads_per_block = 256
            blocks = (num_ngrams + threads_per_block - 1) // threads_per_block
            
            print(f"Debug: Running map kernel with {blocks} blocks, {threads_per_block} threads")
            self.map_kernel(
                word_ids_gpu.gpudata,
                np.uint32(num_tokens),
                np.uint32(n),
                np.uint32(vocab_size),
                ngram_ids_gpu.gpudata,
                block=(threads_per_block, 1, 1),
                grid=(blocks, 1)
            )
            cuda.Context.synchronize()
            
            # Verifica che il map abbia prodotto dati validi
            print("Debug: Map kernel completed, transferring to CPU for sort")
            ngram_ids_cpu = ngram_ids_gpu.get()
            
            del word_ids_gpu
            del ngram_ids_gpu
            
            # Verifica dati
            if np.all(ngram_ids_cpu == 0):
                print("Warning: All ngram_ids are zero after map!")
                return {}
            
            print(f"Debug: Sample ngram_ids: {ngram_ids_cpu[:min(5, len(ngram_ids_cpu))]}")
            
            # === 2. SORT (scegli il metodo più affidabile) ===
            # Per array grandi (>1M elementi), il sort CPU è più affidabile
            if num_ngrams > 1_000_000:
                print(f"Debug: Large array ({num_ngrams} elements), sorting on CPU")
                ngram_ids_sorted_cpu = np.sort(ngram_ids_cpu)
            else:
                # Per array piccoli, prova Numba GPU
                try:
                    print("Debug: Sorting on GPU with Numba")
                    ngram_ids_numba = numba_cuda.to_device(ngram_ids_cpu)
                    ngram_ids_numba.sort()
                    numba_cuda.synchronize()
                    ngram_ids_sorted_cpu = ngram_ids_numba.copy_to_host()
                    del ngram_ids_numba
                except Exception as e:
                    print(f"Debug: GPU sort failed ({e}), falling back to CPU sort")
                    ngram_ids_sorted_cpu = np.sort(ngram_ids_cpu)
            
            print(f"Debug: Sample sorted ngram_ids: {ngram_ids_sorted_cpu[:min(5, len(ngram_ids_sorted_cpu))]}")
            
            # === 3. REDUCE SU GPU (PyCUDA) ===
            print("Debug: Running reduce kernel")
            ngram_ids_sorted_gpu = gpuarray.to_gpu(ngram_ids_sorted_cpu)
            
            num_unique_gpu = gpuarray.zeros(1, dtype=np.uint32)
            unique_ngrams_gpu = gpuarray.zeros(num_ngrams, dtype=np.uint64)
            counts_gpu = gpuarray.zeros(num_ngrams, dtype=np.uint32)
            
            threads_per_block_reduce = 256
            blocks_reduce = (num_ngrams + threads_per_block_reduce - 1) // threads_per_block_reduce
            
            self.reduce_kernel(
                ngram_ids_sorted_gpu.gpudata,
                np.uint32(num_ngrams),
                unique_ngrams_gpu.gpudata,
                counts_gpu.gpudata,
                num_unique_gpu.gpudata,
                block=(threads_per_block_reduce, 1, 1),
                grid=(blocks_reduce, 1)
            )
            
            cuda.Context.synchronize()
            
            # === 4. TRASFERIMENTO RISULTATI ===
            num_unique = int(num_unique_gpu.get()[0])
            print(f"Debug: Found {num_unique} unique ngrams")
            
            if num_unique == 0:
                return {}
            
            # Limita a num_unique per sicurezza
            num_unique = min(num_unique, num_ngrams)
                
            unique_ngrams_cpu = unique_ngrams_gpu.get()[:num_unique]
            counts_cpu = counts_gpu.get()[:num_unique]
            
            # Cleanup
            del ngram_ids_sorted_gpu
            del num_unique_gpu
            del unique_ngrams_gpu
            del counts_gpu
            
            # Ricostruzione dizionario
            result_counts = {}
            for i in range(num_unique):
                ngram_id = int(unique_ngrams_cpu[i])
                count = int(counts_cpu[i])
                
                if count == 0:
                    continue
                
                ngram_word_ids = []
                temp_id = ngram_id
                for _ in range(n):
                    word_id = int(temp_id % vocab_size)
                    if word_id not in id_to_word:
                        break
                    ngram_word_ids.append(word_id)
                    temp_id //= vocab_size
                
                if len(ngram_word_ids) != n:
                    continue
                    
                ngram_word_ids.reverse()
                
                try:
                    ngram_words = tuple(id_to_word[wid] for wid in ngram_word_ids)
                    result_counts[ngram_words] = count
                except KeyError:
                    continue
            
            print(f"Debug: Returning {len(result_counts)} ngrams")
            return result_counts
            
        except Exception as e:
            print(f"Error in GPU computation: {e}")
            print("Falling back to CPU computation")
            return self._compute_cpu_fallback(word_ids, n, vocab_size, id_to_word)
    
    def _compute_cpu_fallback(self, word_ids, n, vocab_size, id_to_word):
        """Fallback CPU implementation"""
        from collections import Counter
        
        ngrams = []
        for i in range(len(word_ids) - n + 1):
            ngram = tuple(id_to_word[word_ids[i + j]] for j in range(n))
            ngrams.append(ngram)
        
        return dict(Counter(ngrams))