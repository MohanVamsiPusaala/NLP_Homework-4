"""
Part 2 – Q1: Character-Level RNN Language Model
Author: Mohan Vamsi Pusaala
Student ID: 700773458
Course: CS5760 – Natural Language Processing
University: University of Central Missouri
Semester: Fall 2025

This script trains a small character-level RNN (LSTM by default), saves:
 - training & validation loss curve (results/loss_curve.png)
 - three sample generations at temperatures 0.7, 1.0, 1.2 (results/gen_temp_*.txt)
 - console output (results/console_output.txt)

Configurable hyperparams are at the top of the file.
"""

import os
import random
import math
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# ----------------------------
# Configuration (change if needed)
# ----------------------------
DATA_DIR = "data"
TOY_FILE = os.path.join(DATA_DIR, "toy.txt")   # small toy data (required)
BOOK_FILE = os.path.join(DATA_DIR, "book.txt") # optional larger text (50-200 KB)
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparams
EMBED_SIZE = 32   # embedding dimension
HIDDEN_SIZE = 128 # hidden size (64-256 recommended)
NUM_LAYERS = 1    # RNN layers
RNN_TYPE = "LSTM" # "RNN", "GRU", or "LSTM"

# Training hyperparams
SEQ_LEN = 10      # lowered for toy text
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 5
TEACHER_FORCING = True


# Generation params
GENERATE_LEN = 300
TEMPS = [0.7, 1.0, 1.2]
SEED = 42

# Create results dir
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Utility: Load text and build vocab
# ----------------------------
def load_text(prefer_book=False):
    """
    If prefer_book True and book exists, use book.txt. Otherwise fall back to toy.txt.
    """
    path = BOOK_FILE if (prefer_book and os.path.exists(BOOK_FILE)) else TOY_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Create {TOY_FILE} with a few toy lines.")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # simple cleaning: replace CRLF, keep characters as-is
    text = text.replace("\r\n", "\n")
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

# ----------------------------
# Dataset utilities
# ----------------------------
def create_batches(text, stoi, seq_len, batch_size):
    """
    Create batches of integer sequences (next-character prediction)
    Return generator that yields input and target tensors.
    """
    # convert all text to indices
    data = [stoi[c] for c in text]
    # number of full sequences we can create
    num_seq = len(data) // seq_len
    if num_seq == 0:
        raise ValueError("Text too short for given seq_len. Lower seq_len or add more data.")
    data = data[:num_seq * seq_len]
    # reshape: (num_seq, seq_len)
    arr = np.array(data).reshape(num_seq, seq_len)
    # shuffle sequences
    np.random.shuffle(arr)
    # yield batches
    for i in range(0, num_seq, batch_size):
        batch = arr[i:i+batch_size]
        if batch.shape[0] == 0:
            continue
        x = torch.tensor(batch, dtype=torch.long, device=DEVICE)
        # target is next char — here we shift sequences by one; for last char we use next sequence's first or wrap-around
        # Simpler: targets are same shape: for each sequence, target[t] = input[t+1], last target = input[0] (wrap)
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        # wrap last char to first char of batch row (or to 0)
        y[:, -1] = x[:, 0]
        yield x, y

# ----------------------------
# Model
# ----------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, rnn_type="LSTM"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T)
        emb = self.embed(x)               # (B, T, E)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)             # (B, T, V)
        return logits, hidden

# ----------------------------
# Sampling helper
# ----------------------------
def sample(model, start_str, stoi, itos, length=200, temp=1.0):
    """
    Generate characters from the model using temperature sampling.
    """
    model.eval()
    indices = [stoi.get(c, None) for c in start_str]
    # if any char not in vocab, replace with random char from vocab
    if any(i is None for i in indices):
        indices = [random.choice(list(stoi.values())) for _ in start_str]
    input_seq = torch.tensor([indices], dtype=torch.long, device=DEVICE)
    hidden = None
    out_chars = list(start_str)
    with torch.no_grad():
        logits, hidden = model(input_seq, hidden)
        last = input_seq[:, -1:]
        for _ in range(length):
            logits, hidden = model(last, hidden)            # logits shape (1,1,V)
            logits = logits[:, -1, :] / max(temp, 1e-8)     # divide by temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy().ravel()
            next_idx = np.random.choice(len(probs), p=probs)
            out_chars.append(itos[next_idx])
            last = torch.tensor([[next_idx]], dtype=torch.long, device=DEVICE)
    return "".join(out_chars)

# ----------------------------
# Training loop
# ----------------------------
def train_model(text, stoi, itos, prefer_book=False):
    vocab_size = len(stoi)
    model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RNN_TYPE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # split into train/val indices by simple split
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        # create batches from train_text
        for x, y in create_batches(train_text, stoi, SEQ_LEN, BATCH_SIZE):
            optimizer.zero_grad()
            logits, _ = model(x)                   # (B, T, V)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        avg_train = epoch_loss / max(1, steps)
        train_losses.append(avg_train)

        # validation (no teacher forcing as we compute loss on real sequence)
        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for x, y in create_batches(val_text, stoi, SEQ_LEN, BATCH_SIZE):
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
                vsteps += 1
        avg_val = val_loss / max(1, vsteps)
        val_losses.append(avg_val)

        # print progress & save to console file
        msg = f"Epoch {epoch}/{EPOCHS}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}"
        print(msg)
        with open(os.path.join(RESULTS_DIR, "console_output.txt"), "a", encoding="utf-8") as cf:
            cf.write(msg + "\n")

    # Save loss curve
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label="train")
    plt.plot(range(1, EPOCHS+1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

    # Sample 3 temperature-controlled generations
    start = text[:min(30, len(text))] if len(text) > 0 else "hello"
    for t in TEMPS:
        gen = sample(model, start_str=start, stoi=stoi, itos=itos, length=GENERATE_LEN, temp=t)
        fname = os.path.join(RESULTS_DIR, f"gen_temp_{t:.1f}.txt")
        with open(fname, "w", encoding="utf-8") as gf:
            gf.write(f"Temperature {t}\n---\n")
            gf.write(gen)
        with open(os.path.join(RESULTS_DIR, "console_output.txt"), "a", encoding="utf-8") as cf:
            cf.write(f"\n=== Generation at temp={t} ===\n")
            cf.write(gen + "\n")

    # Save model (optional)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "char_rnn_model.pt"))
    print("Training complete. Results saved in 'results/'.")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Clear/prepare console_output
    open(os.path.join(RESULTS_DIR, "console_output.txt"), "w").close()

    # Load toy text by default; if you want to use book.txt set prefer_book=True
    try:
        text = load_text(prefer_book=False)  # change to True to use book.txt if present
    except FileNotFoundError as e:
        # create a basic toy file for you automatically
        toy_example = "\n".join(["hello", "help", "hello hello", "hell", "help me", "hello world"])
        with open(TOY_FILE, "w", encoding="utf-8") as f:
            f.write(toy_example)
        print(f"Created toy file at {TOY_FILE}. Re-run the script.")
        text = load_text(prefer_book=False)

    print(f"Loaded text length: {len(text)} characters.")
    # Build vocab and print info
    stoi, itos = build_vocab(text)
    print(f"Vocab size: {len(stoi)}; Sequence length: {SEQ_LEN}; Hidden size: {HIDDEN_SIZE}")

    # Train the model
    train_model(text, stoi, itos)
