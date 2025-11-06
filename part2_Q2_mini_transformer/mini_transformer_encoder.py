"""
Part 2 – Q2: Mini Transformer Encoder
Author: Mohan Vamsi Pusaala
Student ID: 700773458
Course: CS5760 – Natural Language Processing
University: University of Central Missouri
Semester: Fall 2025
"""

# ---------------------------------------------------------
# Import necessary libraries
# ---------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. Dataset Preparation
# ---------------------------------------------------------
# Define 10 simple sentences as our small dataset
sentences = [
    "i like cats",
    "she loves dogs",
    "they read books",
    "we eat apples",
    "i like reading",
    "dogs chase cats",
    "cats climb trees",
    "she reads novels",
    "we enjoy coffee",
    "they play football"
]

# Tokenize each sentence into a list of words
tokens_list = [s.split() for s in sentences]

# Build vocabulary mapping each unique word to an index
vocab = {"<pad>": 0, "<unk>": 1}  # add special tokens for padding and unknowns
for toks in tokens_list:
    for t in toks:
        if t not in vocab:
            vocab[t] = len(vocab)

# Create inverse vocabulary for decoding indices later
inv_vocab = {i: w for w, i in vocab.items()}

# Find maximum sentence length (for padding)
max_len = max(len(toks) for toks in tokens_list)

# Convert words to integer indices and pad shorter sentences
encoded = []
for toks in tokens_list:
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
    ids += [vocab["<pad>"]] * (max_len - len(ids))  # add <pad> tokens
    encoded.append(ids)

# Convert to PyTorch tensor
input_ids = torch.tensor(encoded, dtype=torch.long)

# ---------------------------------------------------------
# 2. Model Configuration
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)

# Model parameters
d_model = 64    # embedding dimension
num_heads = 2   # number of attention heads
d_ff = 128      # hidden layer size in feed-forward network

# ---------------------------------------------------------
# 3. Define Transformer Components
# ---------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to word embeddings.
    This helps the model capture token position information
    since transformers are position-agnostic by default.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create matrix of [max_len, d_model] with sinusoidal values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # store in buffer (not trainable)

    def forward(self, x):
        # Add positional encoding to embeddings
        return x + self.pe[:, :x.size(1)].to(x.device)

# Scaled Dot-Product Attention function
def scaled_dot_product_attention(Q, K, V):
    """
    Computes attention scores using the scaled dot-product formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    dk = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn

class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention:
    - Splits embeddings into multiple heads
    - Applies scaled dot-product attention per head
    - Concatenates results and applies a linear layer
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Linear layers for Q, K, V
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        # Final linear projection
        self.out_lin = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()
        # Project embeddings into Q, K, V
        Q = self.q_lin(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_lin(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_lin(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply scaled dot-product attention
        out, attn = scaled_dot_product_attention(Q, K, V)
        # Combine attention heads back into single representation
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_lin(out), attn

class PositionwiseFFN(nn.Module):
    """
    Two-layer feed-forward network applied to each token position independently.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    """
    A single encoder block that includes:
    - Multi-head self-attention
    - Add & Norm
    - Feed-forward layer
    - Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention sublayer
        attn_out, attn = self.mha(x)
        x = self.norm1(x + attn_out)
        # Feed-forward sublayer
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x, attn

class MiniTransformerEncoder(nn.Module):
    """
    Mini Transformer Encoder combining:
    - Embedding layer
    - Positional encoding
    - One encoder block (self-attention + FFN)
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.layer = EncoderLayer(d_model, num_heads, d_ff)

    def forward(self, ids):
        x = self.embed(ids)          # Token embedding
        x = self.pos(x)              # Add positional encoding
        x, attn = self.layer(x)      # Apply encoder block
        return x, attn

# ---------------------------------------------------------
# 4. Run Model
# ---------------------------------------------------------
# Initialize model
model = MiniTransformerEncoder(len(vocab), d_model, num_heads, d_ff, max_len).to(device)
model.eval()

# Run a forward pass through the model
with torch.no_grad():
    contextual_emb, attn_weights = model(input_ids)

# Create folder to save results
os.makedirs("results", exist_ok=True)

# Print and save summary info
print("Vocab size:", len(vocab))
print("Max seq len:", max_len)
print("Final contextual embeddings shape:", contextual_emb.shape)

with open("results/console_output.txt", "w") as f:
    f.write(f"Vocab size: {len(vocab)}\n")
    f.write(f"Max seq len: {max_len}\n")
    f.write(f"Final contextual embeddings shape: {contextual_emb.shape}\n\n")

# Save input tokens
with open("results/tokens.txt", "w") as f:
    for i, toks in enumerate(tokens_list, 1):
        f.write(f"{i:02d}. {toks}\n")

# Save embeddings for the first sentence
ce_np = contextual_emb.cpu().numpy()
with open("results/embeddings_sentence1.txt", "w") as f:
    f.write("Sentence 1: ['i', 'like', 'cats']\n")
    for t_idx, token_id in enumerate(encoded[0]):
        tok = inv_vocab[token_id]
        vec = ce_np[0, t_idx].tolist()
        f.write(f"token='{tok}': {vec}\n")

# ---------------------------------------------------------
# 5. Visualization - Attention Heatmap
# ---------------------------------------------------------
# Average attention across all heads for the first sentence
attn = attn_weights.cpu().numpy()
sent_idx = 0
avg = attn[sent_idx].mean(axis=0)
tokens_for_plot = tokens_list[sent_idx] + ["<pad>"] * (max_len - len(tokens_list[sent_idx]))

# Plot the attention heatmap
plt.figure(figsize=(5, 4))
plt.imshow(avg, aspect="auto")
plt.colorbar()
plt.xticks(range(max_len), tokens_for_plot, rotation=45)
plt.yticks(range(max_len), tokens_for_plot)
plt.title("Sentence #1 – Average Attention Across Heads")
plt.tight_layout()
plt.savefig("results/attention_heatmap.png", dpi=200)
plt.close()

print("All results saved in the 'results/' folder.")
