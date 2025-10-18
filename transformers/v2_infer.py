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
# HYPERPARAMETERS - OPTIMIZED FOR STABILITY
# ================================
HP = {
    # Paths
    "train_path": "/kaggle/input/hindi-translit/hin_train_sampled.jsonl",
    "valid_path": "/kaggle/input/hindi-translit/hin_valid.json",
    "test_path": "/kaggle/input/hindi-translit/hin_test.json",
    
    # Model architecture - REDUCED FOR STABILITY
    "d_model": 512,              # Reduced from 768
    "nhead": 8,                  # Reduced from 12
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 2048,     # Reduced from 3072
    "dropout": 0.1,              # Reduced from 0.15
    "activation": "relu",        # Changed back to relu
    "max_seq_length": 128,
    "layer_norm_eps": 1e-5,      # Changed from 1e-6
    
    # Training - MORE CONSERVATIVE
    "batch_size": 128,           # Increased from 96
    "learning_rate": 3e-4,       # REDUCED from 8e-4 (key fix)
    "num_epochs": 25,
    "warmup_epochs": 5,          # Longer warmup from 3
    "min_lr": 5e-6,              # Higher minimum
    "label_smoothing": 0.1,      # Reduced from 0.15
    "max_target_len": 64,
    "grad_clip": 1.0,            # Increased from 0.5
    "weight_decay": 0.0001,      # Reduced from 0.01
    
    # Inference
    "beam_size": 5,              # Reduced from 8
    "length_penalty": 0.6,       # Reduced from 0.8
    
    # Other
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "/kaggle/working/best_transformer_v2.pt",
    "seed": 42,
}

assert HP["num_encoder_layers"] <= 2 and HP["num_decoder_layers"] <= 2

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
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ================================
# TRANSFORMER MODEL
# ================================

class ImprovedTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768, nhead=12,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=3072,
                 dropout=0.15, activation="gelu", max_seq_length=128, layer_norm_eps=1e-6):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False  # Changed to False for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False  # Changed to False for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        # Output projection with weight tying
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.output_proj.weight = self.tgt_embed.weight
        
        self._init_parameters()
    
    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embed' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                else:
                    nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_key_padding_mask=None, 
                tgt_key_padding_mask=None):
        # Embeddings with scaling
        src_emb = self.embed_dropout(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.embed_dropout(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Create causal mask
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Decode
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(output)
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

def beam_search_decode(model, src_ids, inv_tgt, hp):
    device = hp["device"]
    beam_size = hp["beam_size"]
    alpha = hp["length_penalty"]
    SOS, EOS = 1, 2
    max_len = hp["max_target_len"]
    
    model.eval()
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = (src_tensor == 0)
    
    with torch.no_grad():
        beams = [(0.0, [SOS])]
        completed = []
        
        for step in range(max_len):
            candidates = []
            
            for score, seq in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue
                
                dec_input = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_pad_mask = (dec_input == 0)
                
                logits = model(src_tensor, dec_input,
                             src_key_padding_mask=src_pad_mask,
                             tgt_key_padding_mask=tgt_pad_mask)
                
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_ids = log_probs.topk(min(beam_size * 2, log_probs.size(0)))
                
                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq))
            
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            
            if not beams or (completed and len(completed) >= beam_size):
                break
        
        all_hyps = completed + beams
        if not all_hyps:
            return ""
        
        best_seq = max(all_hyps, key=lambda x: x[0] / (len(x[1]) ** alpha))[1]
        
        pred_chars = []
        for pid in best_seq[1:]:
            if pid == EOS:
                break
            ch = inv_tgt.get(pid, "")
            if ch not in ("<pad>", "<sos>", "<eos>"):
                pred_chars.append(ch if ch != "<unk>" else "")
        
        return "".join(pred_chars)

def infer_word(word: str, model, src_vocab, inv_tgt, hp) -> str:
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower())]
    return beam_search_decode(model, ids, inv_tgt, hp)




import warnings 
# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')
warnings.filterwarnings('ignore', message='.*nested tensors.*')
warnings.filterwarnings('ignore', message='.*key_padding_mask.*')



hp=HP
device = device = torch.device(hp["device"])
ckpt = torch.load("transformer_checkpoint/v2.pt", map_location=device)


src_vocab = ckpt["src_vocab"]
tgt_vocab = ckpt["tgt_vocab"]
inv_tgt = ckpt["inv_tgt"]


model = ImprovedTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        num_decoder_layers=hp["num_decoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        activation=hp["activation"],
        max_seq_length=hp["max_seq_length"],
        layer_norm_eps=hp["layer_norm_eps"]
    ).to(device)


model.load_state_dict(ckpt["model_state"])

if __name__ == "__main__":
    examples = input("Enter the sentence \n").strip().split()
    for word in examples:
        pred = infer_word(word, model, src_vocab, inv_tgt, hp)
        print(f"{word:15s} -> {pred}")