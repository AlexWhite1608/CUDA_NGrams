import sys
import os
from src.utils.ngram_benchmark import NgramBenchmark
from src.utils.logger import setup_logger
from src.utils.csv_exporter import BenchmarkCSVExporter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logger = setup_logger("main", level="INFO")
csv_exporter = BenchmarkCSVExporter(output_dir="results")

CORPUS_PATH = "data/data.txt"
HISTS_V2_SWEETSPOT = 128 
NUM_RUNS = 3  

# Experiment A: Scalability vs n-gram size
EXP_A_AMPLIFY = 10
EXP_A_N_VALUES = [2]    # 2, 3, 4, 5

# Experiment B: Scalability vs corpus size
EXP_B_AMPLIFY_FACTORS = [5, 10, 20, 50]
EXP_B_N_VALUE = [2] # 2, 3



def print_summary_table(title: str, results: list):
    if not results:
        logger.warning(f"No results for {title}")
        return

    logger.info("")
    logger.info("=" * 90)
    logger.info(f"SUMMARY: {title}")
    logger.info("=" * 90)
    
    show_hists = any(r['max_private_hists'] > 0 for r in results)
    
    if show_hists:
        header = f"{'Amplify':<8} {'N':<3} {'Alg':<5} {'Hists':<6} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<12} {'Status'}"
        logger.info(header)
        logger.info("-" * 90)
    else:
        header = f"{'Amplify':<8} {'N':<3} {'Alg':<5} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<12} {'Status'}"
        logger.info(header)
        logger.info("-" * 82)
    
    for r in results:
        verified = "PASS" if r['verification_passed'] else "FAIL"
        amp_str = str(r['amplification_factor']) + 'x'
        
        cpu_str = f"{r['cpu_time_mean']:.3f}±{r['cpu_time_std']:.3f}s"
        gpu_str = f"{r['gpu_time_mean']:.3f}±{r['gpu_time_std']:.3f}s"
        speedup_str = f"{r['speedup_mean']:.1f}±{r['speedup_std']:.1f}x"
        
        if show_hists:
            hists_str = str(r['max_private_hists']) if r['max_private_hists'] > 0 else "-"
            row = f"{amp_str:<8} {r['n']:<3} {r['algorithm'].upper():<5} {hists_str:<6} {cpu_str:<15} {gpu_str:<15} {speedup_str:<12} {verified}"
        else:
            row = f"{amp_str:<8} {r['n']:<3} {r['algorithm'].upper():<5} {cpu_str:<15} {gpu_str:<15} {speedup_str:<12} {verified}"
        
        logger.info(row)
    
    logger.info("=" * (90 if show_hists else 82))
    logger.info("")


def run_experiment_A_scalability_n(corpus_path: str):
    logger.info("")
    logger.info("=" * 90)
    logger.info("EXPERIMENT A")
    logger.info(f"Configuration: {EXP_A_AMPLIFY}x corpus, n={EXP_A_N_VALUES}, {NUM_RUNS} runs per test")
    logger.info("=" * 90)
    
    results_A = []
    all_detailed_runs = []
    
    for alg in ["v1", "v2", "B"]:
        hists = HISTS_V2_SWEETSPOT if alg == "v2" else 0
        
        logger.info("")
        if alg == "v2":
            logger.info(f"Testing Algorithm: {alg.upper()} (private_histograms={hists})")
        else:
            logger.info(f"Testing Algorithm: {alg.upper()}")
        
        try:
            benchmark = NgramBenchmark(
                corpus_path, 
                EXP_A_AMPLIFY, 
                alg, 
                max_private_hists=hists
            )
            results = benchmark.run_all_benchmarks(
                n_values=EXP_A_N_VALUES, 
                num_runs=NUM_RUNS,
                experiment_name="experiment_A"
            )
            results_A.extend(results)
            all_detailed_runs.extend(benchmark.get_detailed_runs())
        except Exception as e:
            logger.error(f"Experiment A failed for algorithm {alg.upper()}: {e}")

    print_summary_table("Experiment A", results_A)
    
    # Export to CSV
    if results_A:
        config = {
            'corpus_amplification': EXP_A_AMPLIFY,
            'n_values': str(EXP_A_N_VALUES),
            'num_runs': NUM_RUNS,
            'v2_private_hists': HISTS_V2_SWEETSPOT
        }
        summary_file = csv_exporter.export_results("experiment_A", results_A, config)
        detailed_file = csv_exporter.export_all_runs("experiment_A", all_detailed_runs)
        logger.info(f"Results exported to: {summary_file}")
        logger.info(f"Detailed runs exported to: {detailed_file}")
    
    return results_A


def run_experiment_B_scalability_N(corpus_path: str):
    logger.info("")
    logger.info("=" * 90)
    logger.info("EXPERIMENT B")
    logger.info(f"Configuration: n={EXP_B_N_VALUE}, amplification={EXP_B_AMPLIFY_FACTORS}, {NUM_RUNS} runs per test")
    logger.info("=" * 90)
    
    results_B = []
    all_detailed_runs = []
    
    for alg in ["v1", "v2", "B"]:
        for amplify_factor in EXP_B_AMPLIFY_FACTORS:
            hists = HISTS_V2_SWEETSPOT if alg == "v2" else 0
            
            logger.info("")
            if alg == "v2":
                logger.info(f"Testing {alg.upper()} with {amplify_factor}x corpus (private_histograms={hists})")
            else:
                logger.info(f"Testing {alg.upper()} with {amplify_factor}x corpus")
            
            try:
                benchmark = NgramBenchmark(
                    corpus_path, 
                    amplify_factor, 
                    alg, 
                    max_private_hists=hists
                )
                results = benchmark.run_all_benchmarks(
                    n_values=EXP_B_N_VALUE, 
                    num_runs=NUM_RUNS,
                    experiment_name="experiment_B"
                )
                results_B.extend(results)
                all_detailed_runs.extend(benchmark.get_detailed_runs())
            except Exception as e:
                logger.error(f"Experiment B failed for {alg.upper()} at {amplify_factor}x: {e}")

    print_summary_table("Experiment B", results_B)
    
    if results_B:
        config = {
            'n_value': str(EXP_B_N_VALUE),
            'amplify_factors': str(EXP_B_AMPLIFY_FACTORS),
            'num_runs': NUM_RUNS,
            'v2_private_hists': HISTS_V2_SWEETSPOT
        }
        summary_file = csv_exporter.export_results("experiment_B", results_B, config)
        detailed_file = csv_exporter.export_all_runs("experiment_B", all_detailed_runs)
        logger.info(f"Results exported to: {summary_file}")
        logger.info(f"Detailed runs exported to: {detailed_file}")
    
    return results_B


def main():
    if not os.path.exists(CORPUS_PATH):
        logger.error(f"Corpus file not found: {CORPUS_PATH}")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 90)
    logger.info("BENCHMARK CONFIGURATION")
    logger.info("=" * 90)
    logger.info(f"Runs per test: {NUM_RUNS}")
    logger.info(f"Result format: mean ± standard deviation")
    logger.info("=" * 90)

    # Run experiments
    run_experiment_A_scalability_n(CORPUS_PATH)
    run_experiment_B_scalability_N(CORPUS_PATH)

    logger.info("")
    logger.info("=" * 90)
    logger.info("All experiments completed successfully")
    logger.info("=" * 90)
    logger.info("")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)