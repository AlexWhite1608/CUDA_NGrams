import sys
import os

from src.utils.ngram_benchmark import NgramBenchmark

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

CORPUS_PATH = "data/data.txt"
HISTS_V2_SWEETSPOT = 128 

# Experiment A: Scalabilità al variare di n (Complessità)
EXP_A_AMPLIFY = 1
EXP_A_N_VALUES = [2]

# Experiments B: Scalabilità al variare di N (Dimensione Dati)
EXP_B_AMPLIFY_FACTORS = [5, 10, 20, 50]
EXP_B_N_VALUE = [3] 

# print summary table function
def print_summary_table(title: str, results: list):
    
    print("\n" + "=" * 78)
    print(f"  RECAP: {title}")
    print("=" * 78)
    
    if not results:
        print("=" * 78 + "\n")
        return

    show_hists = any(r['max_private_hists'] > 0 for r in results)
    
    if show_hists:
        print(f"{'Amplify':<8} {'N':<3} {'Alg':<5} {'Hists':<6} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Verified'}")
        print("-" * 78)
    else:
        print(f"{'Amplify':<8} {'N':<3} {'Alg':<5} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Verified'}")
        print("-" * 70)
    
    for r in results:
        verified = "✓" if r['verification_passed'] else "✗"
        amp_str = str(r['amplification_factor']) + 'x'
        
        if show_hists:
            hists_str = str(r['max_private_hists']) if r['max_private_hists'] > 0 else "-"
            print(f"{amp_str:<8} {r['n']:<3} {r['algorithm'].upper():<5} {hists_str:<6} {r['cpu_time']:<12.4f} {r['gpu_time']:<12.4f} {r['speedup']:<10.2f} {verified}")
        else:
            print(f"{amp_str:<8} {r['n']:<3} {r['algorithm'].upper():<5} {r['cpu_time']:<12.4f} {r['gpu_time']:<12.4f} {r['speedup']:<10.2f} {verified}")
            
    print("=" * (78 if show_hists else 70) + "\n")


def run_experiment_A_scalability_n(corpus_path: str):
    print("\n" + "#" * 78)
    print(f"##  Experiment A")
    print(f"##  Corpus: {EXP_A_AMPLIFY}x, n: {EXP_A_N_VALUES}")
    print("#" * 78 + "\n")
    
    results_A = []
    
    for alg in ["v1", "v2", "B"]:
        
        hists = HISTS_V2_SWEETSPOT if alg == "v2" else 0
        
        print(f"--- Testing Alg: {alg.upper()} ---")
        
        try:
            benchmark = NgramBenchmark(
                corpus_path, 
                EXP_A_AMPLIFY, 
                alg, 
                max_private_hists=hists
            )
            results = benchmark.run_all_benchmarks(n_values=EXP_A_N_VALUES)
            results_A.extend(results)
        except Exception as e:
            print(f"Error in Exp A (Alg: {alg}): {e}")

    print_summary_table("Experiment A", results_A)
    return results_A

def run_experiment_B_scalability_N(corpus_path: str):
    print("\n" + "#" * 78)
    print(f"##  Experiment B")
    print(f"##  n: {EXP_B_N_VALUE}, Corpus: {EXP_B_AMPLIFY_FACTORS}")
    print("#" * 78 + "\n")
    
    results_B = []
    
    for alg in ["v1", "v2", "B"]:
        for amplify_factor in EXP_B_AMPLIFY_FACTORS:
            
            hists = HISTS_V2_SWEETSPOT if alg == "v2" else 0
            
            print(f"--- Testing Alg: {alg.upper()} (Amplify: {amplify_factor}x) ---")
            
            try:
                benchmark = NgramBenchmark(
                    corpus_path, 
                    amplify_factor, 
                    alg, 
                    max_private_hists=hists
                )
                results = benchmark.run_all_benchmarks(n_values=EXP_B_N_VALUE)
                results_B.extend(results)
            except Exception as e:
                print(f"Error in Exp B (Alg: {alg}, Amp: {amplify_factor}x): {e}")

    print_summary_table("Experiment B", results_B)
    return results_B

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus file '{CORPUS_PATH}' not found.")
        sys.exit(1)

    # A
    run_experiment_A_scalability_n(CORPUS_PATH)
    
    # B
    run_experiment_B_scalability_N(CORPUS_PATH)

    print("\n" + "=" * 78)
    print("  All experiments completed.")
    print("=" * 78 + "\n")

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