import os
import torch
import random # Added
import pytest
import safetensors
import matplotlib.pyplot as plt
import numpy as np # Added
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone
import imageio  # Add imageio import
import io # Add io import


if __name__ == "__main__":
    torch.set_default_device('cuda')

    # checkpoint_base_path = "adaptive_hcg_slm2_360M_w_0.001_l_12_XY58F4YK"
    checkpoint_base_path = "adaptive_hcg_slm2_360M_w_0.0_l_12_test"
    # checkpoint_base_path = "adaptive_hcg_slm2_360M_w_0.01_l_12_80KF9UFM"
    # checkpoint_base_path = "adaptive_hcg_slm2_360M_w_0.001_l_12_XY58F4YK"
    print("checkpoint_base_path", checkpoint_base_path)
    checkpoints = os.listdir(checkpoint_base_path)
    checkpoints = [x for x in checkpoints if x.startswith('checkpoint')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

    if not checkpoints:
        print("No checkpoints found.")
        exit()

    # --- Initialization for Random Token Evolution ---
    num_random_tokens = 10
    random_token_indices = None
    random_token_strings = None
    token_prob_history = None # Will be {token_idx: []}
    evolution_images = [] # For the second animation
    # Define key upfront, try the standard one first
    hcg_log_a_key = 'model.adaptive_down.3.hcg.hcg_log_a'
    tokeniser = None
    vocab_size = None

    # Load tokenizer and select random tokens from the first checkpoint
    first_checkpoint_path = os.path.join(checkpoint_base_path, checkpoints[0])
    print(f"Loading tokenizer and initial state from: {checkpoints[0]}")
    try:
        tokeniser = AutoTokenizer.from_pretrained(first_checkpoint_path)
        first_pretrained_checkpoint = os.path.join(first_checkpoint_path, "model.safetensors")
        first_state_dict = safetensors.torch.load_file(first_pretrained_checkpoint)

        # Find hcg_log_a key in the first checkpoint
        if hcg_log_a_key not in first_state_dict:
            print(f"Warning: Default key '{hcg_log_a_key}' not found in first checkpoint. Searching...")
            potential_keys = [k for k in first_state_dict.keys() if 'hcg.hcg_log_a' in k]
            if not potential_keys:
                print(f"Error: Could not find any hcg_log_a key in the first checkpoint: {checkpoints[0]}")
                exit()
            hcg_log_a_key = potential_keys[0]
            print(f"Using alternative key found: {hcg_log_a_key}")
        else:
             print(f"Using default key: {hcg_log_a_key}")

        first_hcg_log_a = first_state_dict[hcg_log_a_key]
        vocab_size = first_hcg_log_a.shape[0]

        # Select random tokens
        if vocab_size >= num_random_tokens:
            random_token_indices = random.sample(range(vocab_size), num_random_tokens)
            random_token_strings = tokeniser.batch_decode(random_token_indices)
            token_prob_history = {idx: [] for idx in random_token_indices}
            print(f"Selected {num_random_tokens} random tokens (indices): {random_token_indices}")
            # print(f"Selected random tokens (strings): {random_token_strings}") # Optional: uncomment for debugging
        else:
            print(f"Warning: Vocab size ({vocab_size}) is smaller than requested random tokens ({num_random_tokens}). Skipping evolution plot.")
            random_token_indices = None # Disable tracking

    except Exception as e:
        print(f"Error during initialization with {checkpoints[0]}: {e}")
        exit()

    # --- Main Loop ---
    images = [] # List to store histogram plot images
    last_valid_hcg_log_a = None # Store the log_a from the last successfully processed checkpoint
    last_valid_checkpoint_index = -1

    for i, checkpoint in enumerate(checkpoints):
        print(f"Processing checkpoint: {checkpoint} ({i+1}/{len(checkpoints)})")
        checkpoint_path = os.path.join(checkpoint_base_path, checkpoint)

        pretrained_checkpoint = os.path.join(checkpoint_path, "model.safetensors")
        state_dict = None
        hcg_log_a = None
        passing_probs = None
        error_occurred = False

        try:
            state_dict = safetensors.torch.load_file(pretrained_checkpoint)
            # Use the key found during initialization
            if hcg_log_a_key not in state_dict:
                print(f"Warning: Key '{hcg_log_a_key}' not found in {checkpoint}. Skipping.")
                error_occurred = True
            else:
                hcg_log_a = state_dict[hcg_log_a_key]
                # Ensure it has the expected vocab size
                if hcg_log_a.shape[0] != vocab_size:
                    print(f"Warning: Vocab size mismatch in {checkpoint} (expected {vocab_size}, got {hcg_log_a.shape[0]}). Skipping.")
                    error_occurred = True
                else:
                    passing_probs = torch.sigmoid(hcg_log_a)
                    last_valid_hcg_log_a = hcg_log_a # Update last valid tensor
                    last_valid_checkpoint_index = i

        except Exception as e:
            print(f"Error loading or processing state dict for {checkpoint}: {e}. Skipping.")
            error_occurred = True

        # --- Store Random Token Probs (Handle errors) ---
        if random_token_indices is not None and token_prob_history is not None:
            if error_occurred or passing_probs is None:
                for token_idx in random_token_indices:
                    token_prob_history[token_idx].append(float('nan')) # Append NaN on error
            else:
                current_probs = passing_probs[random_token_indices]
                for token_idx, prob in zip(random_token_indices, current_probs):
                    token_prob_history[token_idx].append(prob.item())

        # Skip histogram plotting if error occurred before probability calculation
        if error_occurred or passing_probs is None:
            continue

        # --- Plot Histogram (Only if probs calculated successfully) ---
        passing_probs_np = passing_probs.cpu().numpy()

        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        try:
            hist_counts, _, _ = ax_hist.hist(passing_probs_np, bins=50, color='skyblue', edgecolor='black', range=(0, 1))
            # Ensure y-axis starts at 0 and accommodates the max count, with a minimum height
            max_count = hist_counts.max() if hist_counts.size > 0 else 0
            ax_hist.set_ylim(bottom=0, top=max(50000, max_count * 1.1))
        except Exception as hist_e:
            print(f"Warning: Could not plot histogram for {checkpoint}: {hist_e}")
            ax_hist.set_ylim(bottom=0, top=50000) # Default ylim on error

        ax_hist.set_title(f'Passing Prob Distribution Checkpoint: {checkpoint}')
        ax_hist.set_xlabel('Approximate Passing Probability')
        ax_hist.set_ylabel('Number of Tokens')
        ax_hist.grid(axis='y', alpha=0.75)

        # Save plot to a BytesIO object
        buf_hist = io.BytesIO()
        try:
            fig_hist.savefig(buf_hist, format='png')
            buf_hist.seek(0)
            images.append(imageio.imread(buf_hist))
        except Exception as save_e:
            print(f"Warning: Could not save histogram image for {checkpoint}: {save_e}")
        finally:
            plt.close(fig_hist) # Close the figure to free memory

    # --- Save Histogram Animation ---
    if images:
        print("Creating histogram animation...")
        hist_output_path = 'src/transformers/models/llama/interpretation/pruning_probs_animation.gif'
        try:
            imageio.mimsave(hist_output_path, images, fps=2, loop=1)
            print(f"Histogram animation saved to {hist_output_path}")
        except Exception as e:
            print(f"Error saving histogram animation: {e}")
    else:
        print("No histogram images were generated or saved successfully.")

    # --- Generate and Save Random Token Evolution Animation ---
    if random_token_indices is not None and token_prob_history is not None:
        print("Creating random token evolution animation...")
        num_checkpoints_processed = len(next(iter(token_prob_history.values()))) # Get actual processed length

        if num_checkpoints_processed > 0:
            # Determine a consistent color map for the tokens
            colors = plt.cm.tab10(np.linspace(0, 1, num_random_tokens))
            token_colors = {idx: colors[j] for j, idx in enumerate(random_token_indices)}

            for k in range(1, num_checkpoints_processed + 1): # Create a frame for each checkpoint processed
                fig_evol, ax_evol = plt.subplots(figsize=(12, 7))
                frame_checkpoint_name = checkpoints[k-1] if k-1 < len(checkpoints) else f"Index {k-1}"
                current_handles = {} # To store handles for unique legend

                for token_idx, token_str in zip(random_token_indices, random_token_strings):
                    # Plot only up to the current frame k
                    probs_to_plot = token_prob_history[token_idx][:k]
                    indices_to_plot = np.arange(k)

                    # Create mask for valid (non-NaN) points
                    valid_mask = ~np.isnan(probs_to_plot)

                    # Find segments of valid data to plot separately
                    segments = np.ma.flatnotmasked_contiguous(np.ma.masked_invalid(probs_to_plot))

                    label = f'{token_str} ({token_idx})'
                    token_color = token_colors[token_idx]
                    plotted_handle = None

                    for segment in segments:
                        seg_indices = indices_to_plot[segment]
                        seg_probs = np.array(probs_to_plot)[segment]
                        line, = ax_evol.plot(seg_indices, seg_probs, marker='.', linestyle='-', color=token_color)
                        if plotted_handle is None: # Store handle only once per token per frame
                             plotted_handle = line

                    # Store the handle for the legend using the full label
                    if plotted_handle is not None:
                        current_handles[label] = plotted_handle

                ax_evol.set_title(f'Passing Probability Evolution for {num_random_tokens} Random Tokens (Up to: {frame_checkpoint_name})')
                ax_evol.set_xlabel('Checkpoint Index')
                ax_evol.set_ylabel('Approximate Passing Probability (Sigmoid(log_a))')
                ax_evol.set_xlim(left=-0.5, right=num_checkpoints_processed - 0.5) # Keep x-axis fixed
                ax_evol.set_ylim(-0.05, 1.05) # Probability range slightly padded
                ax_evol.grid(True, alpha=0.5)

                # Use the collected handles for the legend
                if current_handles:
                    ax_evol.legend(current_handles.values(), current_handles.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
                else:
                    plt.tight_layout()

                # Save plot to buffer
                buf_evol = io.BytesIO()
                try:
                    fig_evol.savefig(buf_evol, format='png')
                    buf_evol.seek(0)
                    evolution_images.append(imageio.imread(buf_evol))
                except Exception as save_evol_e:
                    print(f"Warning: Could not save evolution frame {k}/{num_checkpoints_processed}: {save_evol_e}")
                finally:
                    plt.close(fig_evol)

            # Save evolution animation
            if evolution_images:
                evol_output_path = 'src/transformers/models/llama/interpretation/random_token_evolution_animation.gif'
                try:
                    imageio.mimsave(evol_output_path, evolution_images, fps=2, loop=1)
                    print(f"Random token evolution animation saved to {evol_output_path}")
                except Exception as e:
                    print(f"Error saving evolution animation: {e}")
            else:
                print("No evolution images were successfully generated or saved.")
        else:
            print("No checkpoints were successfully processed to create evolution animation.")
    else:
        print("Random token selection or history tracking was not initialized. Skipping evolution animation.")


    # --- Final Checkpoint Analysis (Using last valid data) ---
    print(f"--- Final Analysis Results (Based on last valid checkpoint: {checkpoints[last_valid_checkpoint_index] if last_valid_checkpoint_index != -1 else 'None'}) ---")
    if last_valid_hcg_log_a is not None and tokeniser is not None:
        top_passed = last_valid_hcg_log_a.argsort(dim=-1, descending=True)
        top_pruned = last_valid_hcg_log_a.argsort(dim=-1, descending=False)

        # The closest indexes to 0
        indices_close_to_zero = torch.where( (last_valid_hcg_log_a < 0.0001) * (last_valid_hcg_log_a > -0.0001) )[0]
        print(f"Indices close to zero: {indices_close_to_zero.shape}")
        print(f"Indices close to zero: {tokeniser.batch_decode(indices_close_to_zero)}")
        print("\n\n")


        print("--- Top 100 Most Likely to be Passed Tokens (Last Valid Checkpoint) ---")
        try:
            top_passed_tokens = tokeniser.batch_decode(top_passed[:100])
            print(top_passed_tokens)
        except Exception as e:
            print(f"Error decoding top passed tokens: {e}")

        print("--- Top 100 Most Likely to be Pruned Tokens (Last Valid Checkpoint) ---")
        try:
            top_pruned_tokens = tokeniser.batch_decode(top_pruned[:100])
            print(top_pruned_tokens)
        except Exception as e:
            print(f"Error decoding top pruned tokens: {e}")
    else:
        print("--- Could not load valid data from the last processed checkpoint for final token analysis.")

