"""
Q3 — Scaled Dot-Product Attention
Author: Mohan Vamsi Pusaala
Student ID: 700773458
Course: CS5760 – Natural Language Processing
University: University of Central Missouri
Semester: Fall 2025

This script implements the attention function:
    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V

It tests the implementation with random inputs and prints:
 - raw attention scores (QK^T)
 - softmax over raw scores (naive)
 - scaled scores and scaled softmax
 - numerically-stable softmax results (using max subtraction)
 - final output vectors (attention-weighted values)
 - a stability demonstration with artificially large inputs
All relevant outputs are printed and saved to results/console_output.txt
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

# Reproducible output
torch.manual_seed(42)

# Create results folder
os.makedirs("results", exist_ok=True)

def naive_softmax(x, dim=-1):
    """Naive softmax (calls torch.softmax). Behavior depends on implementation."""
    return torch.softmax(x, dim=dim)

def stable_softmax(x, dim=-1):
    """Numerically stable softmax: subtract max per row before exponentiating."""
    # x : (..., T)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    ex = torch.exp(x - x_max)
    return ex / ex.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, use_scaling=True):
    """
    Compute attention output and weights.

    Args:
        Q: (B, T_q, d_k)
        K: (B, T_k, d_k)
        V: (B, T_k, d_v)
        use_scaling: if True, divide scores by sqrt(d_k)
    Returns:
        out: (B, T_q, d_v)
        weights: (B, T_q, T_k)
        scores: raw scores before softmax
    """
    d_k = Q.size(-1)
    # (B, T_q, d_k) @ (B, d_k, T_k) -> (B, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)
    # Use numerically stable softmax
    weights = stable_softmax(scores, dim=-1)
    out = torch.matmul(weights, V)
    return out, weights, scores

def print_and_save(s, f):
    print(s)
    f.write(s + "\n")

def run_test(batch=1, T_q=3, T_k=3, d_k=4, d_v=4):
    """
    Run a set of tests:
    1) Small random inputs (normal)
    2) Naive softmax (on raw scores) vs scaled + stable softmax
    3) A stability demo with artificially large numbers
    """
    with open("results/console_output.txt", "w") as fout:
        # Create random Q, K, V
        Q = torch.randn(batch, T_q, d_k)
        K = torch.randn(batch, T_k, d_k)
        V = torch.randn(batch, T_k, d_v)

        print_and_save("=== Test 1: Random small inputs ===", fout)
        print_and_save(f"Q shape: {tuple(Q.shape)}, K shape: {tuple(K.shape)}, V shape: {tuple(V.shape)}", fout)

        # Compute raw scores (no scaling)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1))           # (B, T_q, T_k)
        print_and_save("Raw scores (Q K^T):", fout)
        print_and_save(np.array2string(raw_scores.numpy(), precision=4, suppress_small=True), fout)

        # Naive softmax on raw scores
        naive_weights = naive_softmax(raw_scores, dim=-1)
        print_and_save("\nNaive softmax over raw scores (may be unstable):", fout)
        print_and_save(np.array2string(naive_weights.numpy(), precision=4, suppress_small=True), fout)

        # Scaled scores + stable softmax
        scaled_scores = raw_scores / math.sqrt(d_k)
        stable_weights = stable_softmax(scaled_scores, dim=-1)
        print_and_save("\nScaled scores (raw / sqrt(d_k)):", fout)
        print_and_save(np.array2string(scaled_scores.numpy(), precision=4, suppress_small=True), fout)
        print_and_save("\nStable softmax over scaled scores:", fout)
        print_and_save(np.array2string(stable_weights.numpy(), precision=4, suppress_small=True), fout)

        # Output vectors (attention-weighted values) using scaled stable softmax
        out = torch.matmul(stable_weights, V)
        print_and_save("\nAttention output vectors (using scaled stable softmax):", fout)
        print_and_save(np.array2string(out.numpy(), precision=4, suppress_small=True), fout)

        # Softmax stability check: compare softmax before and after subtracting max
        print_and_save("\n=== Softmax stability check (manual) ===", fout)
        # compute naive softmax without stable trick (but torch.softmax is stable under-the-hood)
        # For demonstration we will show manual unstable computation using exp (without subtracting max)
        scores_np = raw_scores.numpy()
        try:
            manual_exp = np.exp(scores_np)              # may overflow if scores large
            manual_unstable = manual_exp / manual_exp.sum(axis=-1, keepdims=True)
            print_and_save("Manual (unstable) softmax on raw scores (numpy.exp):", fout)
            print_and_save(np.array2string(manual_unstable, precision=4, suppress_small=True), fout)
        except OverflowError:
            print_and_save("Manual unstable softmax overflowed (scores too large).", fout)

        # Now stable manual softmax for the same raw scores:
        scores_max = scores_np.max(axis=-1, keepdims=True)
        stable_manual = np.exp(scores_np - scores_max)
        stable_manual = stable_manual / stable_manual.sum(axis=-1, keepdims=True)
        print_and_save("\nManual stable softmax on raw scores (subtract row max):", fout)
        print_and_save(np.array2string(stable_manual, precision=4, suppress_small=True), fout)

        # Now demonstrate instability with artificially large Q,K
        print_and_save("\n=== Stability Demonstration with very large values ===", fout)
        large_factor = 1e6
        Q_large = Q * large_factor
        K_large = K * large_factor
        raw_large = torch.matmul(Q_large, K_large.transpose(-2, -1))

        print_and_save("Raw large scores (Q_large K_large^T) sample:", fout)
        # show a small slice
        print_and_save(np.array2string(raw_large.numpy(), precision=4, suppress_small=True), fout)

        # Try naive numpy exp (will overflow)
        try:
            raw_large_np = raw_large.numpy()
            bad = np.exp(raw_large_np)
            bad_sm = bad / bad.sum(axis=-1, keepdims=True)
            print_and_save("\nManual unstable softmax on large scores (should overflow or become NaN):", fout)
            print_and_save(np.array2string(bad_sm, precision=4, suppress_small=True), fout)
        except Exception as e:
            print_and_save(f"\nManual unstable softmax failed: {e}", fout)

        # Stable softmax on large scores (subtract max)
        raw_large_max = raw_large.max(dim=-1, keepdim=True).values
        stable_large = torch.exp(raw_large - raw_large_max)
        stable_large = stable_large / stable_large.sum(dim=-1, keepdim=True)
        print_and_save("\nStable softmax on large scores (torch, subtract max):", fout)
        print_and_save(np.array2string(stable_large.numpy(), precision=4, suppress_small=True), fout)

        # Scaled & stable attention on large values (divide by sqrt(d_k) first)
        scaled_large = raw_large / math.sqrt(d_k)
        stable_scaled_large = stable_softmax(scaled_large, dim=-1)
        out_large = torch.matmul(stable_scaled_large, V)
        print_and_save("\nScaled + Stable softmax weights on large scores:", fout)
        print_and_save(np.array2string(stable_scaled_large.numpy(), precision=4, suppress_small=True), fout)
        print_and_save("\nAttention output vectors for large inputs (using scaled+stable):", fout)
        print_and_save(np.array2string(out_large.numpy(), precision=4, suppress_small=True), fout)

        print_and_save("\n=== End of Q3 attention test ===", fout)

if __name__ == "__main__":
    run_test()
    print("Q3 attention test complete — results written to results/console_output.txt")
