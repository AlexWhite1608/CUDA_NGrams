import sys
import os
import argparse

from utils.ngram_benchmark import NgramBenchmark

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='N-gram GPU Analyzer - Compare CPU vs GPU performance'
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        required=True,
        help='Path to the input text file (e.g., a book from Project Gutenberg)'
    )
    
    parser.add_argument(
        '--amplify',
        type=int,
        default=10,
        help='Amplification factor for the corpus (default: 10)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'char', 'word', 'char-bigram', 'char-trigram', 'word-bigram', 'word-trigram'],
        default='all',
        help='Benchmark mode to run (default: all)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate corpus file exists
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file '{args.corpus}' not found.")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("  N-GRAM GPU ANALYZER")
    print("  Comparing Sequential (CPU) vs Parallel (CUDA) Performance")
    print("=" * 70)
    print(f"Corpus file: {args.corpus}")
    print(f"Amplification factor: {args.amplify}")
    print(f"Mode: {args.mode}")
    print("=" * 70 + "\n")
    
    # Initialize benchmark
    benchmark = NgramBenchmark(args.corpus, args.amplify)
    
    # Execute based on mode
    if args.mode == 'all':
        benchmark.run_all_benchmarks()
    else:
        benchmark.setup()
        
        if args.mode == 'char' or args.mode == 'char-bigram':
            benchmark.benchmark_char_ngrams(n=2)
        
        if args.mode == 'char' or args.mode == 'char-trigram':
            benchmark.benchmark_char_ngrams(n=3)
        
        if args.mode == 'word' or args.mode == 'word-bigram':
            benchmark.benchmark_word_ngrams(n=2)
        
        if args.mode == 'word' or args.mode == 'word-trigram':
            benchmark.benchmark_word_ngrams(n=3)
    
    print("\n✓ All tasks completed successfully!\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)