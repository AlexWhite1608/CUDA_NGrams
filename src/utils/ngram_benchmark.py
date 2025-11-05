import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.text_processing import amplify_corpus, tokenize, create_vocabulary, tokens_to_ids
from src.utils.performance_timer import PerformanceTimer, verify_results, print_benchmark_report
from src.sequential.sequential import compute_char_ngrams_cpu
from src.gpu.char_ngrams_gpu import CharNgramGPU

class NgramBenchmark:
    
    def __init__(self, corpus_path: str, amplification_factor: int = 10, algorithm: str = "v1"):
        self.corpus_path = corpus_path
        self.amplification_factor = amplification_factor
        self.algorithm = algorithm
        self.corpus_text = None
        self.tokens = None
        self.word_ids = None
        self.word_to_id = None
        self.id_to_word = None
        self.vocab_size = 0
        
        self.char_gpu = CharNgramGPU(algorithm=algorithm)
    
    def setup(self):
        print("\n" + "=" * 70)
        print("  SETUP: Loading and preprocessing corpus")
        print("=" * 70)
        print(f"Algorithm: {self.algorithm.upper()}")
        
        # loads and amplifies corpus
        self.corpus_text = amplify_corpus(self.corpus_path, self.amplification_factor)
        
        # tokenizes text
        self.tokens = tokenize(self.corpus_text)
        
        # creates vocabulary
        self.word_to_id, self.id_to_word = create_vocabulary(self.tokens)
        self.vocab_size = len(self.word_to_id)
        
        # converts tokens to IDs
        self.word_ids = tokens_to_ids(self.tokens, self.word_to_id)
        
        print("Setup completed\n")
    
    # Runs benchmark for character n-grams
    def benchmark_char_ngrams(self, n: int):
        title = f"{'Bigrams' if n == 2 else 'Trigrams'} of Characters [{self.algorithm.upper()}]"
        corpus_size_mb = len(self.corpus_text.encode('utf-8')) / (1024 * 1024)
        
        # CPU
        with PerformanceTimer("CPU Char N-grams") as timer_cpu:
            result_cpu = compute_char_ngrams_cpu(self.corpus_text, n)
        cpu_time = timer_cpu.get_elapsed()
        
        # GPU
        with PerformanceTimer(f"GPU Char N-grams ({self.algorithm})") as timer_gpu:
            result_gpu = self.char_gpu.compute_char_ngrams_gpu(self.corpus_text, n)
        gpu_time = timer_gpu.get_elapsed()
        
        # Verification
        verification_passed = verify_results(result_cpu, result_gpu)
        
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
            "corpus_size_mb": corpus_size_mb,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "verification_passed": verification_passed,
            "speedup": cpu_time / gpu_time if gpu_time > 0 else float('inf')
        }
    
    # Runs all benchmarks
    def run_all_benchmarks(self):
        self.setup()
        
        print("\n" + "=" * 70)
        print("  STARTING ALL BENCHMARKS")
        print("=" * 70 + "\n")
        
        results = []
        
        # bigrams of characters
        results.append(self.benchmark_char_ngrams(n=2))
        
        # trigrams of characters
        results.append(self.benchmark_char_ngrams(n=3))
        
        print("\n" + "=" * 70)
        print("  ALL BENCHMARKS COMPLETED")
        print("=" * 70 + "\n")
        
        return results