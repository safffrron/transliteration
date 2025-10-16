import os
import json
import math
import random
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ================================
# HYPERPARAMETERS
# ================================
HP = {
    # Paths
    "train_path": "/kaggle/input/hindi-translit/hin_train_sampled.jsonl",
    "valid_path": "/kaggle/input/hindi-translit/hin_valid.json",
    "test_path": "/kaggle/input/hindi-translit/hin_test.json",
    
    # Model architecture
    "d_model": 512,           # Model dimension
    "nhead": 8,               # Number of attention heads
    "num_encoder_layers": 2,  # Max 2 layers as per constraint
    "num_decoder_layers": 2,  # Max 2 layers as per constraint
    "dim_feedforward": 2048,  # FFN hidden dimension
    "dropout": 0.1,
    "activation": "relu",
    "max_seq_length": 128,
    
    # Training
    "batch_size": 128,
    "learning_rate": 5e-4,    # Lower LR for transformers
    "num_epochs": 20,
    "warmup_steps": 4000,     # LR warmup for better convergence
    "label_smoothing": 0.1,   # Helps with overconfidence
    "max_target_len": 64,
    "grad_clip": 1.0,
    
    # Inference
    "beam_size": 5,
    "length_penalty": 0.6,
    
    # Other
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "/kaggle/working/best_transformer_model.pt",
    "seed": 42,
}

# Enforce max 2 layers
assert HP["num_encoder_layers"] <= 2, "num_encoder_layers must be <= 2"
assert HP["num_decoder_layers"] <= 2, "num_decoder_layers must be <= 2"

# Reproducibility
torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
random.seed(HP["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(HP["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# POSITIONAL ENCODING
# ================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ================================
# TRANSFORMER MODEL
# ================================
class TransformerTransliterator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048,
                 dropout=0.1, activation="relu", max_seq_length=128):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        src_key_padding_mask: (batch, src_len) - True for padding
        tgt_key_padding_mask: (batch, tgt_len) - True for padding
        """
        # Embeddings with scaling
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Create target mask (causal mask for autoregressive generation)
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(output)
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

def beam_search_decode(model, src_ids, inv_tgt, hp):
    """Beam search decoding for a single source sequence."""
    device = hp["device"]
    beam_size = hp["beam_size"]
    alpha = hp["length_penalty"]
    SOS, EOS = 1, 2
    max_len = hp["max_target_len"]
    
    model.eval()
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = (src_tensor == 0)
    
    with torch.no_grad():
        # Initialize beam with SOS token
        beams = [(0.0, [SOS])]
        completed = []
        
        for step in range(max_len):
            candidates = []
            
            for score, seq in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue
                
                # Prepare decoder input
                dec_input = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_pad_mask = (dec_input == 0)
                
                # Forward pass
                logits = model(src_tensor, dec_input,
                             src_key_padding_mask=src_pad_mask,
                             tgt_key_padding_mask=tgt_pad_mask)
                
                # Get probabilities for next token
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                
                # Get top-k candidates
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq))
            
            # Select top beam_size candidates with length normalization
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            
            if not beams:
                break
        
        # Select best sequence
        all_hyps = completed + beams
        if not all_hyps:
            return ""
        
        best_seq = max(all_hyps, key=lambda x: x[0] / (len(x[1]) ** alpha))[1]
        
        # Convert to string
        pred_chars = []
        for pid in best_seq[1:]:  # Skip SOS
            if pid == EOS:
                break
            ch = inv_tgt.get(pid, "")
            if ch not in ("<pad>", "<sos>", "<eos>"):
                pred_chars.append(ch if ch != "<unk>" else "")
        
        return "".join(pred_chars)


def infer_word(word: str, model, src_vocab, inv_tgt, hp) -> str:
    """Inference on a single word using beam search."""
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower())]
    return beam_search_decode(model, ids, inv_tgt, hp)


hp = HP
device = torch.device(hp["device"])
ckpt = torch.load("transformer_checkpoint/v1.pt", map_location=device)

src_vocab = ckpt["src_vocab"]
tgt_vocab = ckpt["tgt_vocab"]
inv_tgt = ckpt["inv_tgt"]

model = TransformerTransliterator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        num_decoder_layers=hp["num_decoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        activation=hp["activation"],
        max_seq_length=hp["max_seq_length"]
    ).to(device)
model.load_state_dict(ckpt["model_state"])




examples = input("Enter a sentence \n").strip().split()
for word in examples:
    pred = infer_word(word, model, src_vocab, inv_tgt, hp)
    print(f"{word:15s} -> {pred}")