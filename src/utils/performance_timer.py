import time
from typing import Dict
from contextlib import contextmanager

class PerformanceTimer:
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start_time
    
    def get_elapsed(self) -> float:
        return self.elapsed_time if self.elapsed_time is not None else 0.0

def verify_results(cpu_result: Dict, gpu_result: Dict, tolerance: float = 1e-6) -> bool:

    if len(cpu_result) != len(gpu_result):
        print(f"The dimensions are different CPU={len(cpu_result)}, GPU={len(gpu_result)}")
        return False
    
    for key in cpu_result:
        if key not in gpu_result:
            print(f"Key is missing in GPU result: {key}")
            return False
        
        cpu_val = cpu_result[key]
        gpu_val = gpu_result[key]
        
        if abs(cpu_val - gpu_val) > tolerance:
            print(f"Different value for {key}: CPU={cpu_val}, GPU={gpu_val}")
            return False
    
    return True


def calculate_speedup(cpu_time: float, gpu_time: float) -> float:

    if gpu_time == 0:
        return float('inf')
    return cpu_time / gpu_time

def print_benchmark_report(
    title: str,
    corpus_size_mb: float,
    cpu_time: float,
    gpu_time: float,
    verification_passed: bool,
    top_ngrams_cpu: Dict = None,
    top_ngrams_gpu: Dict = None,
    n_top: int = 5
):

    speedup = calculate_speedup(cpu_time, gpu_time)
    
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"Corpus Size:            {corpus_size_mb:.2f} MB")
    print(f"Sequential Time (CPU):  {cpu_time:.4f} seconds")
    print(f"Parallel Time (GPU):    {gpu_time:.4f} seconds")
    print(f"Verification:           {'PASS' if verification_passed else 'FAIL'}")
    print(f"Speedup (CPU/GPU):      {speedup:.2f}x")
    
    if top_ngrams_cpu and n_top > 0:
        print(f"\nTop {n_top} N-grams (CPU):")
        sorted_cpu = sorted(top_ngrams_cpu.items(), key=lambda x: x[1], reverse=True)[:n_top]
        for ngram, count in sorted_cpu:
            print(f"  {ngram}: {count}")
    
    if top_ngrams_gpu and n_top > 0:
        print(f"\nTop {n_top} N-grams (GPU):")
        sorted_gpu = sorted(top_ngrams_gpu.items(), key=lambda x: x[1], reverse=True)[:n_top]
        for ngram, count in sorted_gpu:
            print(f"  {ngram}: {count}")
    
    print("=" * 70 + "\n")