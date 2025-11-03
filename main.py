import sys
import os

from src.utils.ngram_benchmark import NgramBenchmark

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# === CONFIGURAZIONE MANUALE ===
CORPUS_PATH = "data/data.txt"         # Modifica qui il percorso del file
AMPLIFY_FACTOR = 10                     # Modifica qui il fattore di amplificazione
MODE = "all"                            # Scegli tra: all, char, word, char-bigram, char-trigram, word-bigram, word-trigram

def main():
    """Main execution function."""
    # Usa le variabili definite sopra
    corpus = CORPUS_PATH
    amplify = AMPLIFY_FACTOR
    mode = MODE

    # Validate corpus file exists
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
    print("=" * 70 + "\n")
    
    # Initialize benchmark
    benchmark = NgramBenchmark(corpus, amplify)
    
    # Execute based on mode
    if mode == 'all':
        benchmark.run_all_benchmarks()
    else:
        benchmark.setup()
        
        if mode == 'char' or mode == 'char-bigram':
            benchmark.benchmark_char_ngrams(n=2)
        
        if mode == 'char' or mode == 'char-trigram':
            benchmark.benchmark_char_ngrams(n=3)
        
        if mode == 'word' or mode == 'word-bigram':
            benchmark.benchmark_word_ngrams(n=2)
        
        if mode == 'word' or mode == 'word-trigram':
            benchmark.benchmark_word_ngrams(n=3)
    
    print("\n✓ All tasks completed successfully!\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)