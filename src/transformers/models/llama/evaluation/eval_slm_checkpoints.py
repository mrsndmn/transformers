import os
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import torch

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_acc_hellaswag, evaluate_acc_winogrande, evaluate_acc_piqa, evaluate_acc_siqa, evaluate_acc_openbookqa

class ContinuousEvaluator:
    def __init__(self):
        self.vanilla_checkpoints_dir = "./vanilla_slm2_1.7B_pretrain_w_0.000_l_-_BM2DG7DH"
        self.vanilla_16L_checkpoints_dir = "./vanilla_slm2_1.7B_pretrain_16L_w_0.000_l_-_X3NTEOHS"
        self.adaptive_checkpoints_dir = "./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_K6WC5X8F"

        self.model_to_checkpoints = [
            ("vanilla", LlamaForCausalLM, self.vanilla_checkpoints_dir),
            ("vanilla_16L", LlamaForCausalLM, self.vanilla_16L_checkpoints_dir),
            ("adaptive", AdaptiveLlamaForCausalLM, self.adaptive_checkpoints_dir),
        ]

        self.benchmarks = [
            ("hellaswag", evaluate_acc_hellaswag),
            ("winogrande", evaluate_acc_winogrande),
            ("piqa", evaluate_acc_piqa),
            ("siqa", evaluate_acc_siqa),
            ("openbookqa", evaluate_acc_openbookqa),
        ]

        self.results_file = "slm_checkpoints_benchmarks_results.json"
        self.results = self.load_results()
        self.evaluated_combinations = self.get_evaluated_combinations()

    def load_results(self):
        """Load existing results from JSON file"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Could not load {self.results_file}, starting fresh")
        return []

    def save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.results_file}")

    def get_evaluated_combinations(self):
        """Get set of already evaluated (model, checkpoint, benchmark) combinations"""
        evaluated = set()
        for result in self.results:
            if len(result) >= 4:
                model_name, checkpoint, benchmark_name, _ = result[:4]
                evaluated.add((model_name, checkpoint, benchmark_name))
        return evaluated

    def get_checkpoint_number(self, checkpoint_dir):
        """Extract checkpoint number from directory name"""
        if checkpoint_dir.startswith("checkpoint-"):
            try:
                return int(checkpoint_dir.split("-")[1])
            except (IndexError, ValueError):
                return 0
        return 0

    def get_new_checkpoints(self):
        """Find new checkpoints that haven't been evaluated yet"""
        new_checkpoints = []
        
        for model_name, model_class, checkpoints_dir in self.model_to_checkpoints:
            if not os.path.exists(checkpoints_dir):
                print(f"Warning: Directory {checkpoints_dir} does not exist")
                continue
                
            for dir_name in sorted(os.listdir(checkpoints_dir), key=self.get_checkpoint_number):
                if dir_name.startswith("checkpoint-"):
                    checkpoint_dir = os.path.join(checkpoints_dir, dir_name)
                    
                    # Check which benchmarks need to be evaluated for this checkpoint
                    pending_benchmarks = []
                    for benchmark_name, benchmark_fn in self.benchmarks:
                        if (model_name, dir_name, benchmark_name) not in self.evaluated_combinations:
                            pending_benchmarks.append((benchmark_name, benchmark_fn))
                    
                    if pending_benchmarks:
                        new_checkpoints.append((model_name, model_class, checkpoint_dir, dir_name, pending_benchmarks))
        
        return new_checkpoints

    def evaluate_checkpoint(self, model_name, model_class, checkpoint_dir, dir_name, pending_benchmarks):
        """Evaluate a single checkpoint on pending benchmarks"""
        print(f"\n=== Evaluating {model_name} checkpoint: {dir_name} ===")
        
        try:
            # Load model
            model = model_class.from_pretrained(checkpoint_dir)
            model.eval()
            model.to("cuda")
            
            # Evaluate each pending benchmark
            for benchmark_name, benchmark_fn in pending_benchmarks:
                print(f"  Running {benchmark_name}...")
                
                try:
                    bench_results = benchmark_fn(model)
                    acc_norm = bench_results['acc_norm']
                    
                    # Add result
                    current_result = [model_name, dir_name, benchmark_name, acc_norm]
                    self.results.append(current_result)
                    
                    # Update evaluated combinations
                    self.evaluated_combinations.add((model_name, dir_name, benchmark_name))
                    
                    # Save results immediately
                    self.save_results()
                    
                    # Update plots
                    self.update_plots()
                    
                    print(f"    {benchmark_name}: {acc_norm:.4f}")
                    
                except Exception as e:
                    print(f"    Error evaluating {benchmark_name}: {e}")
                    continue
            
            # Clean up model from GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error loading model {checkpoint_dir}: {e}")

    def update_plots(self):
        """Create/update plots for each benchmark"""
        if not self.results:
            return
            
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            if len(result) >= 4:
                model_name, checkpoint, benchmark_name, acc_norm = result[:4]
                checkpoint_num = self.get_checkpoint_number(checkpoint)
                df_data.append({
                    'model': model_name,
                    'checkpoint': checkpoint,
                    'checkpoint_num': checkpoint_num,
                    'benchmark': benchmark_name,
                    'acc_norm': acc_norm
                })
        
        if not df_data:
            return
            
        df = pd.DataFrame(df_data)
        
        # Create plots for each benchmark
        for benchmark_name in df['benchmark'].unique():
            benchmark_df = df[df['benchmark'] == benchmark_name]
            
            plt.figure(figsize=(12, 8))
            
            # Plot for each model
            for model_name in benchmark_df['model'].unique():
                model_df = benchmark_df[benchmark_df['model'] == model_name]
                model_df = model_df.sort_values('checkpoint_num')
                
                plt.plot(model_df['checkpoint_num'], model_df['acc_norm'], 
                        marker='o', label=model_name, linewidth=2, markersize=6)
            
            plt.xlabel('Checkpoint Number')
            plt.ylabel('Accuracy (Normalized)')
            plt.title(f'{benchmark_name.upper()} Performance Over Checkpoints')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{benchmark_name}_performance.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved: {plot_filename}")

    def run_continuous_evaluation(self, sleep_interval=60):
        """Run continuous evaluation loop"""
        print("Starting continuous evaluation...")
        print(f"Monitoring directories:")
        for _, _, dir_path in self.model_to_checkpoints:
            print(f"  - {dir_path}")
        print(f"Results will be saved to: {self.results_file}")
        print(f"Sleep interval: {sleep_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Find new checkpoints
                new_checkpoints = self.get_new_checkpoints()
                
                if new_checkpoints:
                    print(f"Found {len(new_checkpoints)} new checkpoint(s) to evaluate")
                    
                    # Evaluate each new checkpoint
                    for model_name, model_class, checkpoint_dir, dir_name, pending_benchmarks in new_checkpoints:
                        self.evaluate_checkpoint(model_name, model_class, checkpoint_dir, dir_name, pending_benchmarks)
                else:
                    print(f"No new checkpoints found. Sleeping for {sleep_interval} seconds...")
                
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            print("\nStopping continuous evaluation...")
            print(f"Final results saved to: {self.results_file}")

if __name__ == "__main__":
    evaluator = ContinuousEvaluator()
    evaluator.run_continuous_evaluation()
