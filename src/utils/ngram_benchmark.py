import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.text_processing import amplify_corpus, tokenize, create_vocabulary, tokens_to_ids
from src.utils.performance_timer import PerformanceTimer, verify_results, print_benchmark_report
from src.sequential.sequential import compute_char_ngrams_cpu
from src.gpu.char_ngrams_gpu import CharNgramGPU

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
    
    def setup(self):
        print("\n" + "=" * 70)
        print("  SETUP: Loading and preprocessing corpus")
        print("=" * 70)
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Amplification Factor: {self.amplification_factor}x")
        if self.algorithm == "v2":
            print(f"Max Private Hists: {self.max_private_hists}")
        
        self.corpus_text = amplify_corpus(self.corpus_path, self.amplification_factor)
        self.tokens = tokenize(self.corpus_text)
        self.word_to_id, self.id_to_word = create_vocabulary(self.tokens)
        self.vocab_size = len(self.word_to_id)
        self.word_ids = tokens_to_ids(self.tokens, self.word_to_id)
        
        print(f"Corpus Size: {len(self.corpus_text.encode('utf-8')) / (1024 * 1024):.2f} MB")
        print("Setup completed\n")
    
    def _warmup(self):
        print("-" * 70)
        try:
            _ = self.char_gpu.compute_char_ngrams_gpu("warmup", 2)
            print("  Warm-up completed.")
        except Exception as e:
            print(f"  Warm-up failed: {e}")
        print("-" * 70)

    # Runs benchmark for character n-grams
    def benchmark_char_ngrams(self, n: int):
        
        title_map = {1: "1-grams", 2: "Bigrams", 3: "Trigrams", 4: "4-grams"}
        n_name = title_map.get(n, f"{n}-grams")
        title = f"{n_name} of Characters [{self.algorithm.upper()}] (Amp: {self.amplification_factor}x)"
        
        corpus_size_mb = len(self.corpus_text.encode('utf-8')) / (1024 * 1024)
        
        # CPU
        print(f"Running CPU benchmark for n={n}...")
        with PerformanceTimer("CPU Char N-grams") as timer_cpu:
            result_cpu = compute_char_ngrams_cpu(self.corpus_text, n)
        cpu_time = timer_cpu.get_elapsed()
        
        # GPU
        print(f"Running GPU benchmark for n={n}...")
        try:
            with PerformanceTimer(f"GPU Char N-grams ({self.algorithm})") as timer_gpu:
                result_gpu = self.char_gpu.compute_char_ngrams_gpu(self.corpus_text, n)
            gpu_time = timer_gpu.get_elapsed()
            
            verification_passed = verify_results(result_cpu, result_gpu)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        except Exception as e:
            print(f"\n[ERROR] GPU execution failed for n={n}: {e}")
            gpu_time = float('inf')
            verification_passed = False
            result_gpu = {}
            speedup = 0.0

        print_benchmark_report(
            title=f"Benchmark: {title}",
            corpus_size_mb=corpus_size_mb,
            cpu_time=cpu_time,
            gpu_time=gpu_time,
            verification_passed=verification_passed,
            top_ngrams_cpu=result_cpu,
            top_ngrams_gpu=result_gpu,
            n_top=5
        )
        
        return {
            "algorithm": self.algorithm,
            "n": n,
            "amplification_factor": self.amplification_factor,
            "max_private_hists": self.max_private_hists if self.algorithm == "v2" else 0,
            "corpus_size_mb": corpus_size_mb,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "verification_passed": verification_passed,
            "speedup": speedup
        }
    
    def run_all_benchmarks(self, n_values: list = [2, 3]):
        self.setup()
        
        self._warmup()
        
        print("\n" + "=" * 70)
        print(f"  STARTING BENCHMARKS (N = {n_values})")
        print("=" * 70 + "\n")
        
        results = []
        
        for n in n_values:
            results.append(self.benchmark_char_ngrams(n=n))
        
        print("\n" + "=" * 70)
        print("  BENCHMARKS COMPLETED")
        print("=" * 70 + "\n")
        
        return results