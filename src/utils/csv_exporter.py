import csv
import os
from datetime import datetime
from typing import List, Dict


class BenchmarkCSVExporter:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_results(self, experiment_name: str, results: List[Dict], config: Dict = None):
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # CSV columns
        fieldnames = [
            'experiment',
            'algorithm',
            'n',
            'amplification_factor',
            'corpus_size_mb',
            'max_private_hists',
            'num_runs',
            'cpu_time_mean',
            'cpu_time_std',
            'gpu_time_mean',
            'gpu_time_std',
            'speedup_mean',
            'speedup_std',
            'verification_passed'
        ]
        
        if config:
            fieldnames = ['config_' + k for k in config.keys()] + fieldnames
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {'experiment': experiment_name}
                
                # columns
                if config:
                    for key, value in config.items():
                        row[f'config_{key}'] = value
                
                # data
                row.update({
                    'algorithm': result['algorithm'],
                    'n': result['n'],
                    'amplification_factor': result['amplification_factor'],
                    'corpus_size_mb': f"{result['corpus_size_mb']:.2f}",
                    'max_private_hists': result['max_private_hists'],
                    'num_runs': result['num_runs'],
                    'cpu_time_mean': f"{result['cpu_time_mean']:.6f}",
                    'cpu_time_std': f"{result['cpu_time_std']:.6f}",
                    'gpu_time_mean': f"{result['gpu_time_mean']:.6f}",
                    'gpu_time_std': f"{result['gpu_time_std']:.6f}",
                    'speedup_mean': f"{result['speedup_mean']:.2f}",
                    'speedup_std': f"{result['speedup_std']:.2f}",
                    'verification_passed': result['verification_passed']
                })
                
                writer.writerow(row)
        
        return filepath
    
    def export_all_runs(self, experiment_name: str, all_runs_data: List[Dict]):
        if not all_runs_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_detailed_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        fieldnames = [
            'experiment',
            'algorithm',
            'n',
            'amplification_factor',
            'run_number',
            'cpu_time',
            'gpu_time',
            'speedup',
            'verification_passed'
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for run_data in all_runs_data:
                writer.writerow(run_data)
        
        return filepath