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
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 2,
    "bidirectional_encoder": True,
    "dropout": 0.3,
    "attention_type": "bahdanau",  # bahdanau or none
    
    # Training
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "teacher_forcing_start": 0.8,
    "teacher_forcing_end": 0.1,
    "max_target_len": 64,
    "grad_clip": 1.0,
    
    # Inference
    "beam_size": 5,
    "length_penalty": 0.6,  # alpha for length normalization
    
    # Other
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "/kaggle/working/best_attention_model.pt",
    "seed": 42,
}

# Enforce max 2 layers
assert HP["num_layers"] <= 2, "num_layers must be <= 2"

# Reproducibility
torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
random.seed(HP["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(HP["seed"])


# ================================
# MODEL: ENCODER
# ================================
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, 
                 dropout=0.1, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(self, src, src_lens):
        embedded = self.embed(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h_n, c_n)

# ================================
# MODEL: ATTENTION
# ================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(encoder_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, src_lens):
        """
        decoder_hidden: (batch, hidden_dim)
        encoder_outputs: (batch, src_len, encoder_dim)
        src_lens: (batch,)
        """
        batch_size, src_len, _ = encoder_outputs.size()
        
        # Expand decoder hidden to match encoder outputs
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        
        # Compute attention scores
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))
        scores = self.V(energy).squeeze(2)  # (batch, src_len)
        
        # Create mask for padding
        mask = torch.arange(src_len, device=scores.device).unsqueeze(0) >= src_lens.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e10)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, src_len)
        
        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_dim)
        
        return context, attn_weights

# ================================
# MODEL: DECODER WITH ATTENTION
# ================================
class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, encoder_dim,
                 num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        
        # LSTM input is embedding + context
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output projection
        self.out = nn.Linear(hidden_dim + encoder_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_in, hidden, encoder_outputs, src_lens):
        """
        dec_in: (batch, seq_len)
        hidden: tuple of (h, c) each (num_layers, batch, hidden_dim)
        encoder_outputs: (batch, src_len, encoder_dim)
        """
        embedded = self.dropout(self.embed(dec_in))  # (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = embedded.size()
        
        outputs = []
        h, c = hidden
        
        for t in range(seq_len):
            emb_t = embedded[:, t:t+1, :]  # (batch, 1, embed_dim)
            
            # Get attention context using previous hidden state
            h_prev = h[-1]  # Use last layer hidden state (batch, hidden_dim)
            context, attn_weights = self.attention(h_prev, encoder_outputs, src_lens)
            context = context.unsqueeze(1)  # (batch, 1, encoder_dim)
            
            # Concatenate embedding with context
            lstm_input = torch.cat([emb_t, context], dim=2)  # (batch, 1, embed+encoder_dim)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Output projection: concatenate lstm_out, context, and embedding
            combined = torch.cat([lstm_out, context, emb_t], dim=2)
            output = self.out(combined)  # (batch, 1, output_dim)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        return outputs, (h, c)

def prepare_decoder_hidden(h_n, c_n, enc, dec, device):
    """Prepare decoder initial hidden state from encoder."""
    batch_size = h_n.size(1)
    num_layers = enc.num_layers
    
    if enc.bidirectional:
        # Reshape: (num_layers*2, batch, hidden) -> (num_layers, 2, batch, hidden)
        h_n = h_n.view(num_layers, 2, batch_size, enc.hidden_dim)
        c_n = c_n.view(num_layers, 2, batch_size, enc.hidden_dim)
        
        # Concatenate forward and backward
        h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)
        c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)
        
        # Project if dimensions don't match
        if h_n.size(2) != dec.hidden_dim:
            proj_h = nn.Linear(h_n.size(2), dec.hidden_dim).to(device)
            proj_c = nn.Linear(c_n.size(2), dec.hidden_dim).to(device)
            h_0 = proj_h(h_n)
            c_0 = proj_c(c_n)
        else:
            h_0, c_0 = h_n, c_n
    else:
        h_0, c_0 = h_n, c_n
    
    return h_0.contiguous(), c_0.contiguous()


def beam_search_decode(enc, dec, src_ids, src_len, inv_tgt, hp):
    """Beam search decoding for a single source sequence."""
    device = hp["device"]
    beam_size = hp["beam_size"]
    alpha = hp["length_penalty"]
    SOS, EOS = 1, 2
    max_len = hp["max_target_len"]
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len_tensor = torch.tensor([src_len], dtype=torch.long, device=device)
    
    with torch.no_grad():
        enc_out, (h_n, c_n) = enc(src_tensor, src_len_tensor)
        h_0, c_0 = prepare_decoder_hidden(h_n, c_n, enc, dec, device)
        hidden = (h_0, c_0)
        
        # Initialize beam
        beams = [(0.0, [SOS], hidden)]
        completed = []
        
        for step in range(max_len):
            candidates = []
            
            for score, seq, h in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue
                
                input_t = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                out, h_next = dec(input_t, h, enc_out, src_len_tensor)
                logits = out[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq, h_next))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            
            if not beams:
                break
        
        # Select best sequence
        all_hyps = completed + [(score, seq, None) for score, seq, _ in beams]
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

def infer_word(word: str, enc, dec, src_vocab, inv_tgt, hp) -> str:
    """Inference on a single word using beam search."""
    enc.eval()
    dec.eval()
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower())]
    return beam_search_decode(enc, dec, ids, len(ids), inv_tgt, hp)

def infer_sentence(sentence: str, enc, dec, src_vocab, inv_tgt, hp) -> str:
    """Inference on a sentence (space-separated words)."""
    tokens = sentence.strip().split()
    preds = [infer_word(tok, enc, dec, src_vocab, inv_tgt, hp) for tok in tokens]
    return " ".join(preds)



hp=HP
device = torch.device(hp["device"])

ckpt = torch.load("lstm_checkpoint/v2.pt", map_location=device)
src_vocab = ckpt["src_vocab"]
tgt_vocab = ckpt["tgt_vocab"]
inv_tgt = ckpt["inv_tgt"]
input_dim = len(src_vocab)
output_dim = len(tgt_vocab)
encoder_output_dim = hp["hidden_size"] * (2 if hp["bidirectional_encoder"] else 1)

enc = Encoder(
    input_dim=input_dim,
    embed_dim=hp["embed_size"],
    hidden_dim=hp["hidden_size"],
    num_layers=hp["num_layers"],
    dropout=hp["dropout"],
    bidirectional=hp["bidirectional_encoder"]
).to(device)

dec = AttentionDecoder(
    output_dim=output_dim,
    embed_dim=hp["embed_size"],
    hidden_dim=hp["hidden_size"] * (2 if hp["bidirectional_encoder"] else 1),
    encoder_dim=encoder_output_dim,
    num_layers=hp["num_layers"],
    dropout=hp["dropout"]
).to(device)


enc.load_state_dict(ckpt["enc_state"])
dec.load_state_dict(ckpt["dec_state"])


optim_enc = torch.optim.Adam(enc.parameters(), lr=hp["learning_rate"])
optim_dec = torch.optim.Adam(dec.parameters(), lr=hp["learning_rate"])
criterion = nn.CrossEntropyLoss(ignore_index=0)



# sent = input("Enter a sentence \n")
# pred = infer_sentence(sent, enc, dec, src_vocab, inv_tgt, hp)
# print(f"{sent:20s} -> {pred}")


sent = input("Enter a sentence \n").strip().split()
for word in sent:
    pred = infer_word(word, enc, dec, src_vocab, inv_tgt, hp)
    print(f"{word:15s} -> {pred}")