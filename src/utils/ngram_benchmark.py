import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.text_processing import amplify_corpus, tokenize, create_vocabulary, tokens_to_ids
from src.utils.performance_timer import PerformanceTimer, verify_results, print_benchmark_report
from src.sequential.sequential import compute_char_ngrams_cpu
from src.sequential.sequential import compute_word_ngrams_cpu
from src.gpu.char_ngrams_gpu import CharNgramGPU
from src.gpu.word_ngrams_gpu import WordNgramGPU

# NgramBenchmark class to run all benchmarks
class NgramBenchmark:
    
    # Constructor, gets corpus path and amplification factor
    def __init__(self, corpus_path: str, amplification_factor: int = 10):
        self.corpus_path = corpus_path
        self.amplification_factor = amplification_factor
        self.corpus_text = None
        self.tokens = None
        self.word_ids = None
        self.word_to_id = None
        self.id_to_word = None
        self.vocab_size = 0
        
        # initialize GPU n-gramm objects
        self.char_gpu = CharNgramGPU()
        self.word_gpu = WordNgramGPU()
    
    def setup(self):
        print("\n" + "=" * 70)
        print("  SETUP: Loading and preprocessing corpus")
        print("=" * 70)
        
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
        title = f"{'Bigrams' if n == 2 else 'Trigrams'} of Characters"
        corpus_size_mb = len(self.corpus_text.encode('utf-8')) / (1024 * 1024)
        
        # CPU
        with PerformanceTimer("CPU Char N-grams") as timer_cpu:
            result_cpu = compute_char_ngrams_cpu(self.corpus_text, n)
        cpu_time = timer_cpu.get_elapsed()
        
        # GPU
        with PerformanceTimer("GPU Char N-grams") as timer_gpu:
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
    # Runs benchmark for word n-grams
    def benchmark_word_ngrams(self, n: int):
        title = f"{'Bigrams' if n == 2 else 'Trigrams'} of Words"
        corpus_size_mb = len(self.corpus_text.encode('utf-8')) / (1024 * 1024)
        
        # CPU
        with PerformanceTimer("CPU Word N-grams") as timer_cpu:
            result_cpu = compute_word_ngrams_cpu(self.tokens, n)
        cpu_time = timer_cpu.get_elapsed()
        
        # GPU
        with PerformanceTimer("GPU Word N-grams") as timer_gpu:
            result_gpu = self.word_gpu.compute_word_ngrams_gpu(
                self.word_ids, n, self.vocab_size, self.id_to_word
            )
        gpu_time = timer_gpu.get_elapsed()
        
        # verification
        verification_passed = verify_results(result_cpu, result_gpu)
        
        # Report
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
    
    # Runs all benchmarks
    def run_all_benchmarks(self):
        self.setup()
        
        print("\n" + "=" * 70)
        print("  STARTING ALL BENCHMARKS")
        print("=" * 70 + "\n")
        
        # bigrams of characters
        self.benchmark_char_ngrams(n=2)
        
        # trigrams of characters
        self.benchmark_char_ngrams(n=3)
        
        # bigrams of words
        self.benchmark_word_ngrams(n=2)
        
        # trigrams of words
        self.benchmark_word_ngrams(n=3)
        
        print("\n" + "=" * 70)
        print("  ALL BENCHMARKS COMPLETED")
        print("=" * 70 + "\n")