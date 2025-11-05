import sys
import os

from src.utils.ngram_benchmark import NgramBenchmark

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

CORPUS_PATH = "data/data.txt"
AMPLIFY_FACTOR = 10
MODE = "all"  
ALGORITHM = "compare"  # v1, v2, or 'compare' to run both

def main():
    """Main execution function."""
    corpus = CORPUS_PATH
    amplify = AMPLIFY_FACTOR
    mode = MODE
    algorithm = ALGORITHM

    if not os.path.exists(corpus):
        print(f"Error: Corpus file '{corpus}' not found.")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("  N-GRAM GPU ANALYZER")
    print("  Comparing Sequential (CPU) vs Parallel (CUDA) Performance")
    print("=" * 70)
    print(f"Corpus file: {corpus}")
    print(f"Amplification factor: {amplify}")
    print(f"Mode: {mode}")
    print(f"Algorithm: {algorithm}")
    print("=" * 70 + "\n")
    
    # Compare algorithms mode
    if mode == "compare-algorithms" or algorithm == "compare":
        print("\n" + "=" * 70)
        print("  ALGORITHM COMPARISON MODE")
        print("=" * 70 + "\n")
        
        all_results = []
        
        for alg in ["v1", "v2"]:
            print(f"\n{'='*70}")
            print(f"  Testing Algorithm: {alg.upper()}")
            print(f"{'='*70}\n")
            
            benchmark = NgramBenchmark(corpus, amplify, algorithm=alg)
            results = benchmark.run_all_benchmarks()
            all_results.extend(results)
        
        # Print comparison summary
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Algorithm':<12} {'N':<5} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Verified'}")
        print("-" * 70)
        for r in all_results:
            verified = "✓" if r['verification_passed'] else "✗"
            print(f"{r['algorithm'].upper():<12} {r['n']:<5} {r['cpu_time']:<12.4f} {r['gpu_time']:<12.4f} {r['speedup']:<10.2f} {verified}")
        print("=" * 70 + "\n")
        
        return
    
    # Single algorithm mode
    benchmark = NgramBenchmark(corpus, amplify, algorithm=algorithm)
    
    # Execute based on mode
    if mode == 'all':
        benchmark.run_all_benchmarks()
    else:
        benchmark.setup()
        
        if mode == 'char' or mode == 'char-bigram':
            benchmark.benchmark_char_ngrams(n=2)
        
        if mode == 'char' or mode == 'char-trigram':
            benchmark.benchmark_char_ngrams(n=3)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)