import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.text_processing import amplify_corpus, tokenize, create_vocabulary, tokens_to_ids
from src.utils.performance_timer import PerformanceTimer, verify_results
from src.sequential.sequential import compute_char_ngrams_cpu
from src.gpu.char_ngrams_gpu import CharNgramGPU
from src.utils.logger import setup_logger

logger = setup_logger("benchmark")


class NgramBenchmark:
    
    def __init__(self, corpus_path: str, amplification_factor: int = 10, algorithm: str = "v1", max_private_hists: int = 256):
        self.corpus_path = corpus_path
        self.amplification_factor = amplification_factor
        self.algorithm = algorithm
        self.max_private_hists = max_private_hists 
        self.corpus_text = None
        self.tokens = None
        self.word_ids = None
        self.word_to_id = None
        self.id_to_word = None
        self.vocab_size = 0
        
        self.char_gpu = CharNgramGPU(algorithm=algorithm, max_private_hists=self.max_private_hists)
        
        self.detailed_runs = []
    
    def setup(self):
        self.corpus_text = amplify_corpus(self.corpus_path, self.amplification_factor)
        self.tokens = tokenize(self.corpus_text)
        self.word_to_id, self.id_to_word = create_vocabulary(self.tokens)
        self.vocab_size = len(self.word_to_id)
        self.word_ids = tokens_to_ids(self.tokens, self.word_to_id)
        
        corpus_size_mb = len(self.corpus_text.encode('utf-8')) / (1024 * 1024)
        logger.info(f"Corpus size: {corpus_size_mb:.1f} MB ({self.amplification_factor}x amplification)")
    
    def _warmup(self):
        logger.debug("GPU warmup...")
        try:
            _ = self.char_gpu.compute_char_ngrams_gpu("warmup", 2)
        except:
            pass

    def benchmark_char_ngrams_single(self, n: int):
        
        # CPU
        with PerformanceTimer("CPU") as timer_cpu:
            result_cpu = compute_char_ngrams_cpu(self.corpus_text, n)
        cpu_time = timer_cpu.get_elapsed()
        
        # GPU
        try:
            with PerformanceTimer("GPU") as timer_gpu:
                result_gpu = self.char_gpu.compute_char_ngrams_gpu(self.corpus_text, n)
            gpu_time = timer_gpu.get_elapsed()
            
            verification_passed = verify_results(result_cpu, result_gpu)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        except Exception as e:
            logger.error(f"GPU execution failed: {e}")
            gpu_time = float('inf')
            verification_passed = False
            speedup = 0.0

        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'verification_passed': verification_passed
        }

    def benchmark_char_ngrams(self, n: int, num_runs: int = 1, experiment_name: str = ""):
        cpu_times = []
        gpu_times = []
        speedups = []
        all_verified = True
        
        logger.info(f"Running n={n} ({num_runs} iterations)...")
        
        for run_idx in range(num_runs):
            logger.debug(f"  Iteration {run_idx + 1}/{num_runs}")
            
            single_result = self.benchmark_char_ngrams_single(n)
            
            cpu_times.append(single_result['cpu_time'])
            gpu_times.append(single_result['gpu_time'])
            speedups.append(single_result['speedup'])
            all_verified = all_verified and single_result['verification_passed']
            
            self.detailed_runs.append({
                'experiment': experiment_name,
                'algorithm': self.algorithm,
                'n': n,
                'amplification_factor': self.amplification_factor,
                'run_number': run_idx + 1,
                'cpu_time': single_result['cpu_time'],
                'gpu_time': single_result['gpu_time'],
                'speedup': single_result['speedup'],
                'verification_passed': single_result['verification_passed']
            })
        
        # Compute statistics
        cpu_mean = np.mean(cpu_times)
        cpu_std = np.std(cpu_times, ddof=1) if num_runs > 1 else 0.0
        
        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(gpu_times, ddof=1) if num_runs > 1 else 0.0
        
        speedup_mean = np.mean(speedups)
        speedup_std = np.std(speedups, ddof=1) if num_runs > 1 else 0.0
        
        status = "PASS" if all_verified else "FAIL"
        logger.info(f"Results: CPU={cpu_mean:.3f}±{cpu_std:.3f}s, GPU={gpu_mean:.3f}±{gpu_std:.3f}s, Speedup={speedup_mean:.1f}±{speedup_std:.1f}x [{status}]")
        
        return {
            "algorithm": self.algorithm,
            "n": n,
            "amplification_factor": self.amplification_factor,
            "max_private_hists": self.max_private_hists if self.algorithm == "v2" else 0,
            "corpus_size_mb": len(self.corpus_text.encode('utf-8')) / (1024 * 1024),
            "num_runs": num_runs,
            "cpu_time_mean": cpu_mean,
            "cpu_time_std": cpu_std,
            "gpu_time_mean": gpu_mean,
            "gpu_time_std": gpu_std,
            "speedup_mean": speedup_mean,
            "speedup_std": speedup_std,
            "verification_passed": all_verified
        }
    
    def run_all_benchmarks(self, n_values: list = [2, 3], num_runs: int = 1, experiment_name: str = ""):
        self.setup()
        self._warmup()
        
        self.detailed_runs = []
        
        results = []
        for n in n_values:
            results.append(self.benchmark_char_ngrams(n=n, num_runs=num_runs, experiment_name=experiment_name))
        
        return results
    
    def get_detailed_runs(self):
        return self.detailed_runs