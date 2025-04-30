import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
import logging
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def compute_bias_approximation(per_token_embeddings, token_ids=None, min_occurrences=5):
    """
    Compute bias vector approximation for token embedding changes.
    
    For each token, we compute a single bias vector that approximates 
    the direction of change between consecutive layers.
    """
    if token_ids is None:
        # Filter tokens with sufficient occurrences
        token_ids = [token_id for token_id, data in per_token_embeddings.items() 
                    if data["count"] >= min_occurrences]
    
    approximations = {}
    
    for token_id in tqdm(token_ids, desc="Computing bias approximations"):
        data = per_token_embeddings[token_id]
        if data["count"] < min_occurrences:
            continue
        
        # Get mean embeddings for each layer
        mean_embeddings = data["mean_embeddings"]
        num_layers = len(mean_embeddings)
        
        # Compute average change direction across all layer transitions
        all_diffs = []
        for i in range(num_layers - 1):
            diff = mean_embeddings[i+1] - mean_embeddings[i]
            all_diffs.append(diff)
        
        # Compute the average difference vector
        avg_diff = torch.stack(all_diffs).mean(dim=0)
        
        # Store the approximation
        approximations[token_id] = {
            "bias_vector": avg_diff,
            "original_diffs": all_diffs,
            "count": data["count"]
        }
    
    return approximations

def evaluate_approximation(per_token_embeddings, approximations, token_ids=None):
    """
    Evaluate how well the bias vector approximates actual embedding changes.
    
    Computes metrics like MSE, cosine similarity, and R^2 for each token.
    """
    if token_ids is None:
        token_ids = list(approximations.keys())
    
    results = {}
    
    for token_id in tqdm(token_ids, desc="Evaluating approximations"):
        if token_id not in approximations:
            continue
        
        data = per_token_embeddings[token_id]
        approx = approximations[token_id]
        
        # Get the original differences and the bias vector
        original_diffs = approx["original_diffs"]
        bias_vector = approx["bias_vector"].float()
        
        # Metrics for each layer transition
        mse_scores = []
        cosine_sim = []
        r2_scores = []
        
        # Compare the actual diff with the approximation for each layer transition
        for i, diff in enumerate(original_diffs):
            diff = diff.float()
            # MSE between actual diff and bias vector
            mse = torch.mean((diff - bias_vector) ** 2).item()
            mse_scores.append(mse)
            
            # Cosine similarity between actual diff and bias vector
            cos_sim = torch.nn.functional.cosine_similarity(
                diff.unsqueeze(0), bias_vector.unsqueeze(0)
            ).item()
            cosine_sim.append(cos_sim)
            
            # R^2 score
            diff_np = diff.float().numpy()
            bias_np = bias_vector.float().numpy()
            r2 = r2_score(diff_np, bias_np * np.ones_like(diff_np))
            r2_scores.append(r2)
        
        # Store results
        results[token_id] = {
            "mse": np.mean(mse_scores),
            "mse_per_layer": mse_scores,
            "cosine_similarity": np.mean(cosine_sim),
            "cosine_per_layer": cosine_sim,
            "r2": np.mean(r2_scores),
            "r2_per_layer": r2_scores,
            "count": data["count"]
        }
    
    return results

def compare_embeddings_with_approximation(per_token_embeddings, approximations, token_id, output_dir, num_layers=None):
    """
    Visualize the comparison between actual embedding changes and the bias approximation.
    """
    if token_id not in approximations:
        logger.warning(f"Token ID {token_id} not found in approximations")
        return
    
    data = per_token_embeddings[token_id]
    approx = approximations[token_id]
    
    # Get mean embeddings and the bias vector
    mean_embeddings = data["mean_embeddings"]
    bias_vector = approx["bias_vector"]
    
    if num_layers is None:
        num_layers = len(mean_embeddings)
    
    # Compute approximated embeddings by adding the bias vector
    approx_embeddings = [mean_embeddings[0]]  # Start with the first layer embedding
    for i in range(1, num_layers):
        approx_embeddings.append(approx_embeddings[i-1] + bias_vector)
    
    # Compute PCA to visualize in 2D
    all_embeddings = torch.stack(mean_embeddings[:num_layers] + approx_embeddings[:num_layers])
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings.float().numpy())
    
    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    
    # Original embeddings
    orig_x = embeddings_2d[:num_layers, 0]
    orig_y = embeddings_2d[:num_layers, 1]
    plt.scatter(orig_x, orig_y, c=np.arange(num_layers), cmap='viridis', 
                marker='o', s=100, label='Original Embeddings')
    
    # Connect original embeddings with lines
    for i in range(num_layers-1):
        plt.plot([orig_x[i], orig_x[i+1]], [orig_y[i], orig_y[i+1]], 'b-', alpha=0.7)
    
    # Approximated embeddings
    approx_x = embeddings_2d[num_layers:, 0]
    approx_y = embeddings_2d[num_layers:, 1]
    plt.scatter(approx_x, approx_y, c=np.arange(num_layers), cmap='viridis', 
                marker='x', s=100, label='Approximated Embeddings')
    
    # Connect approximated embeddings with lines
    for i in range(num_layers-1):
        plt.plot([approx_x[i], approx_x[i+1]], [approx_y[i], approx_y[i+1]], 'r-', alpha=0.7)
    
    plt.colorbar(label='Layer Index')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Original vs. Approximated Embeddings for Token {token_id}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"token_{token_id}_approx_comparison.png"))
    plt.close()

def plot_approximation_metrics(results, metric_name, output_dir):
    """
    Plot the distribution of approximation metrics.
    """
    metric_values = [results[token_id][metric_name] for token_id in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(metric_values, bins=50)
    plt.xlabel(metric_name.replace('_', ' ').title())
    plt.ylabel('Number of Tokens')
    plt.title(f'Distribution of {metric_name.replace("_", " ").title()} for Bias Approximation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}_distribution.png"))
    plt.close()
    
    return np.mean(metric_values), np.median(metric_values)

def plot_metrics_by_frequency(results, metric_name, output_dir):
    """
    Plot how approximation metrics vary with token frequency.
    """
    frequencies = [results[token_id]["count"] for token_id in results]
    metric_values = [results[token_id][metric_name] for token_id in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(frequencies, metric_values, alpha=0.5)
    plt.xscale('log')
    plt.xlabel('Token Frequency (log scale)')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} vs. Token Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}_vs_frequency.png"))
    plt.close()
    
    # Calculate correlation
    corr, p_value = spearmanr(frequencies, metric_values)
    return corr, p_value

def analyze_similarity_matrix(per_token_embeddings, token_ids, layer_idx=0, output_dir=None):
    """
    Create a correlation matrix of embedding changes across tokens.
    """
    # Extract embedding differences at the specified layer transition
    token_diffs = []
    for token_id in token_ids:
        data = per_token_embeddings[token_id]
        if layer_idx < len(data["embedding_diffs"]):
            token_diffs.append(data["embedding_diffs"][layer_idx])
    
    # Calculate correlation matrix
    n_tokens = len(token_diffs)
    corr_matrix = torch.zeros((n_tokens, n_tokens))
    
    for i in range(n_tokens):
        for j in range(n_tokens):
            cos_sim = torch.nn.functional.cosine_similarity(
                token_diffs[i].unsqueeze(0), token_diffs[j].unsqueeze(0)
            ).item()
            corr_matrix[i, j] = cos_sim
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.float().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Cosine Similarity Between Token Embedding Changes (Layer {layer_idx} to {layer_idx+1})')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"token_changes_correlation_layer_{layer_idx}.png"))
        plt.close()
    
    return corr_matrix

def analyze_token_consistency_with_bias(per_token_embeddings, approximations, token_ids, output_dir):
    """
    Analyze the consistency of changes across tokens with original vs. bias-approximated embeddings.
    """
    
    assert len(token_ids) > 0, "No tokens provided for analysis"

    # Get the number of layers
    num_layers = len(next(iter(per_token_embeddings.values()))["mean_embeddings"]) - 1
    
    # For each layer transition, compute correlation matrices
    for layer_idx in range(num_layers - 1):
        # Original embedding differences
        original_diffs = []
        token_labels = []
        valid_token_ids = []
        
        for token_id in token_ids:
            if token_id not in per_token_embeddings:
                continue
                
            data = per_token_embeddings[token_id]
            if layer_idx < len(data["embedding_diffs"]):
                original_diffs.append(data["embedding_diffs"][layer_idx])
                token_labels.append(str(token_id))
                valid_token_ids.append(token_id)
        
        if not original_diffs:
            logger.warning(f"No valid embedding differences found for layer transition {layer_idx} to {layer_idx+1}")
            breakpoint()
            continue
            
        # Bias vector approximations - use only tokens that have valid original diffs
        approx_diffs = []
        for token_id in valid_token_ids:
            if token_id in approximations:
                approx_diffs.append(approximations[token_id]["bias_vector"])
            else:
                # This should not happen, but let's be safe
                logger.warning(f"Token {token_id} found in original data but not in approximations")
                # Use a placeholder to maintain alignment
                if original_diffs:
                    approx_diffs.append(torch.zeros_like(original_diffs[0]))
        
        if not approx_diffs:
            logger.warning(f"No valid approximation vectors found for layer transition {layer_idx} to {layer_idx+1}")
            continue
            
        # Check if vectors have the same size
        if len(original_diffs) != len(approx_diffs):
            logger.error(f"Mismatch between original diffs ({len(original_diffs)}) and approx diffs ({len(approx_diffs)})")
            continue
            
        # Calculate correlation matrices
        n_tokens = len(original_diffs)
        
        logger.info(f"Computing correlation matrices for layer {layer_idx} with {n_tokens} tokens")
        
        # Original correlations
        orig_corr = torch.zeros((n_tokens, n_tokens))
        for i in range(n_tokens):
            for j in range(n_tokens):
                # Ensure tensors are on CPU and have the correct shape for cosine similarity
                vec1 = original_diffs[i].cpu().view(1, -1)
                vec2 = original_diffs[j].cpu().view(1, -1)
                
                try:
                    cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2).item()
                    orig_corr[i, j] = cos_sim
                except Exception as e:
                    logger.error(f"Error computing cosine similarity: {e}")
                    logger.error(f"Shapes: {vec1.shape}, {vec2.shape}")
                    orig_corr[i, j] = 0.0
        
        # Approximation correlations
        approx_corr = torch.zeros((n_tokens, n_tokens))
        for i in range(n_tokens):
            for j in range(n_tokens):
                # Ensure tensors are on CPU and have the correct shape for cosine similarity
                vec1 = approx_diffs[i].cpu().view(1, -1)
                vec2 = approx_diffs[j].cpu().view(1, -1)
                
                try:
                    cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2).item()
                    approx_corr[i, j] = cos_sim
                except Exception as e:
                    logger.error(f"Error computing cosine similarity: {e}")
                    logger.error(f"Shapes: {vec1.shape}, {vec2.shape}")
                    approx_corr[i, j] = 0.0
        
        # Plot correlation matrices side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Convert to numpy arrays for plotting
        orig_corr_np = orig_corr.cpu().float().numpy()
        approx_corr_np = approx_corr.cpu().float().numpy()
        
        # Check if the matrices have valid values
        if np.isnan(orig_corr_np).any() or np.isnan(approx_corr_np).any():
            logger.warning(f"NaN values found in correlation matrices for layer {layer_idx}")
            # Replace NaN with zeros
            orig_corr_np = np.nan_to_num(orig_corr_np)
            approx_corr_np = np.nan_to_num(approx_corr_np)
        
        # Check for empty matrices
        if orig_corr_np.size == 0 or approx_corr_np.size == 0:
            logger.warning(f"Empty correlation matrices for layer {layer_idx}")
            continue
            
        logger.info(f"Plotting correlation heatmaps for layer {layer_idx}")
        
        # Original correlations
        sns.heatmap(orig_corr_np, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title(f'Original Embedding Changes\nLayer {layer_idx} to {layer_idx+1}')
        
        # Approximation correlations
        sns.heatmap(approx_corr_np, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title(f'Bias-Approximated Changes')
        
        # Add token labels if not too many
        if n_tokens <= 50:
            axes[0].set_yticks(np.arange(n_tokens) + 0.5)
            axes[0].set_yticklabels(token_labels)
            axes[1].set_yticks(np.arange(n_tokens) + 0.5)
            axes[1].set_yticklabels(token_labels)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"correlation_comparison_layer_{layer_idx}.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved correlation heatmap to {output_path}")
        
        # Calculate Frobenius norm of the difference between matrices
        diff_norm = torch.norm(orig_corr - approx_corr, p='fro').item()
        
        # Calculate mean absolute difference
        mean_abs_diff = torch.abs(orig_corr - approx_corr).mean().item()
        
        # Return metrics for the final layer transition
        if layer_idx == num_layers - 2:
            return diff_norm, mean_abs_diff, orig_corr, approx_corr
    
    # Return default values if we didn't process any layers
    return 0.0, 0.0, torch.zeros((1, 1)), torch.zeros((1, 1))

def analyze_token_similarity_visualization(per_token_embeddings, approximations, token_ids, output_dir):
    """
    Create 2D projection of token bias vectors to visualize their relationships.
    """
    # Extract bias vectors for the tokens
    bias_vectors = []
    
    for token_id in token_ids:
        if token_id in approximations:
            bias_vectors.append(approximations[token_id]["bias_vector"])
    
    if len(bias_vectors) == 0:
        logger.warning("No bias vectors found for the provided token IDs")
        return
    
    # Stack bias vectors and apply PCA
    bias_matrix = torch.stack(bias_vectors)
    pca = PCA(n_components=2)
    bias_2d = pca.fit_transform(bias_matrix.float().numpy())
    
    # Create a scatter plot
    plt.figure(figsize=(12, 10))
    plt.scatter(bias_2d[:, 0], bias_2d[:, 1], s=100)
    
    # Add token ID labels
    for i, token_id in enumerate(token_ids):
        if token_id in approximations:
            plt.annotate(str(token_id), (bias_2d[i, 0], bias_2d[i, 1]), fontsize=8)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Projection of Token Bias Vectors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_bias_vectors_2d.png"))
    plt.close()

def compute_correlation_with_metric(per_token_embeddings, approximations, metric_values, metric_name, output_dir):
    """
    Analyze correlation between token bias vectors and a given metric.
    """
    # Calculate cosine similarity between each token's bias vector and the "average" token
    all_bias_vectors = [approximations[token_id]["bias_vector"] for token_id in approximations]
    avg_bias = torch.stack(all_bias_vectors).mean(dim=0)
    
    token_ids = list(approximations.keys())
    similarities = []
    
    for token_id in token_ids:
        bias_vector = approximations[token_id]["bias_vector"]
        cos_sim = torch.nn.functional.cosine_similarity(
            bias_vector.unsqueeze(0), avg_bias.unsqueeze(0)
        ).item()
        similarities.append(cos_sim)
    
    # Get metric values for these tokens
    metric_vals = [metric_values.get(token_id, float('nan')) for token_id in token_ids]
    
    # Remove NaN values
    valid_indices = ~np.isnan(metric_vals)
    similarities = np.array(similarities)[valid_indices]
    metric_vals = np.array(metric_vals)[valid_indices]
    
    # Calculate correlation
    corr, p_value = spearmanr(similarities, metric_vals)
    
    # Plot scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(similarities, metric_vals, alpha=0.5)
    plt.xlabel('Cosine Similarity to Average Bias Vector')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'Correlation: {corr:.4f} (p={p_value:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bias_similarity_vs_{metric_name}.png"))
    plt.close()
    
    return corr, p_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the token embeddings file (.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--min_occurrences", type=int, default=10, help="Minimum token occurrences to analyze")
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to analyze in detail")
    
    args = parser.parse_args()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Load embeddings data
    logger.info(f"Loading token embeddings from {args.embeddings_file}")
    per_token_embeddings = torch.load(args.embeddings_file)
    
    # Create output directory
    model_name = os.path.basename(args.embeddings_file).replace("token_embeddings_", "").replace(".pt", "")
    output_dir = os.path.join("results", "token_approximation", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing data for {len(per_token_embeddings)} unique tokens")
    
    # Filter tokens with sufficient occurrences
    frequent_tokens = [token_id for token_id, data in per_token_embeddings.items() if data["count"] >= args.min_occurrences]
    logger.info(f"Found {len(frequent_tokens)} tokens with at least {args.min_occurrences} occurrences")
    assert len(frequent_tokens) > 0, "No tokens found with sufficient occurrences"
    
    # Sort by frequency
    token_counts = {token_id: per_token_embeddings[token_id]["count"] for token_id in frequent_tokens}
    most_common_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:args.num_tokens]
    
    # Compute bias approximation
    logger.info("Computing bias vector approximations")
    approximations = compute_bias_approximation(per_token_embeddings, most_common_tokens, args.min_occurrences)
    
    # Evaluate approximation
    logger.info("Evaluating bias approximation quality")
    eval_results = evaluate_approximation(per_token_embeddings, approximations, most_common_tokens)
    
    # Visualize token embeddings with approximation
    logger.info("Generating visualizations")
    for token_id in tqdm(most_common_tokens[:10], desc="Plotting token visualizations"):
        token_str = tokenizer.decode([token_id])
        logger.info(f"Processing token {token_id} ('{token_str}')")
        compare_embeddings_with_approximation(per_token_embeddings, approximations, token_id, output_dir)
    
    # Analyze distribution of approximation metrics
    logger.info("Analyzing approximation metrics")
    metric_stats = {}
    for metric_name in ["mse", "cosine_similarity", "r2"]:
        mean_val, median_val = plot_approximation_metrics(eval_results, metric_name, output_dir)
        metric_stats[metric_name] = {
            "mean": mean_val,
            "median": median_val
        }
        
        # Analyze relationship with token frequency
        corr, p_value = plot_metrics_by_frequency(eval_results, metric_name, output_dir)
        metric_stats[metric_name]["frequency_correlation"] = corr
        metric_stats[metric_name]["frequency_p_value"] = p_value
    
    # Analyze token consistency with original vs approximated embeddings
    logger.info("Analyzing token consistency")
    diff_norm, mean_abs_diff, orig_corr, approx_corr = analyze_token_consistency_with_bias(
        per_token_embeddings, approximations, most_common_tokens[:50], output_dir
    )
    
    # Visualize token bias vectors in 2D
    analyze_token_similarity_visualization(per_token_embeddings, approximations, most_common_tokens[:100], output_dir)
    
    # Prepare summary report
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Token Embedding Approximation Analysis Summary\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tokens analyzed: {len(most_common_tokens)}\n")
        f.write(f"Minimum occurrences: {args.min_occurrences}\n\n")
        
        f.write("--- Approximation Quality Metrics ---\n")
        for metric_name, stats in metric_stats.items():
            f.write(f"{metric_name.replace('_', ' ').title()}:\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Median: {stats['median']:.6f}\n")
            f.write(f"  Correlation with frequency: {stats['frequency_correlation']:.6f} ")
            f.write(f"(p={stats['frequency_p_value']:.6f})\n\n")
        
        f.write("--- Token Consistency Analysis ---\n")
        f.write(f"Frobenius norm of correlation difference: {diff_norm:.6f}\n")
        f.write(f"Mean absolute difference between correlations: {mean_abs_diff:.6f}\n\n")
        
        f.write("--- Top Tokens by Approximation Quality ---\n")
        # Sort tokens by cosine similarity (higher is better)
        best_tokens = sorted(eval_results.keys(), key=lambda x: eval_results[x]["cosine_similarity"], reverse=True)[:10]
        worst_tokens = sorted(eval_results.keys(), key=lambda x: eval_results[x]["cosine_similarity"])[:10]
        
        f.write("Best approximated tokens:\n")
        for i, token_id in enumerate(best_tokens):
            token_str = tokenizer.decode([token_id])
            cosine_sim = eval_results[token_id]["cosine_similarity"]
            f.write(f"{i+1}. '{token_str}' (ID: {token_id}): {cosine_sim:.6f}\n")
        
        f.write("\nWorst approximated tokens:\n")
        for i, token_id in enumerate(worst_tokens):
            token_str = tokenizer.decode([token_id])
            cosine_sim = eval_results[token_id]["cosine_similarity"]
            f.write(f"{i+1}. '{token_str}' (ID: {token_id}): {cosine_sim:.6f}\n")
    
    logger.info(f"All analysis results saved to {output_dir}")
    
    # Save approximation results
    torch.save({
        "approximations": approximations,
        "eval_results": eval_results,
        "metric_stats": metric_stats
    }, os.path.join(output_dir, "approximation_results.pt"))
    
    logger.info("Analysis complete") 