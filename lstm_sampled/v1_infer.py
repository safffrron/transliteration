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
from torch.utils.data import Dataset, DataLoader



# ---------------------------
# HYPERPARAMETERS 
# ---------------------------
HP = {
    "train_path": "/kaggle/input/hindi-translit/hin_train_sampled.jsonl",
    "valid_path": "/kaggle/input/hindi-translit/hin_valid.json",
    "test_path":  "/kaggle/input/hindi-translit/hin_test.json",
    "batch_size": 256,
    "embed_size": 128,
    "hidden_size": 256,
    "num_layers": 2,           # MUST be <= 2 (enforced)
    "bidirectional_encoder": True,
    "dropout": 0.4,
    "learning_rate": 1e-3,
    "num_epochs": 12,
    "teacher_forcing_ratio": 0.5,
    "max_target_len": 64,      
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "/kaggle/working/sampled_lstm_model.pt",
    "seed": 42,
}
# Enforce max of 2 layers
if HP["num_layers"] > 2:
    raise ValueError("num_layers must be <= 2 per your constraint.")

# Reproducibility
torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
random.seed(HP["seed"])
hp = HP
device = torch.device(hp["device"])

# ---------------------------
# MODEL (Encoder-Decoder LSTM)
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, src, src_lens):
        # src: (batch, seq_len)
        embedded = self.embed(src)  # (batch, seq_len, embed_dim)
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq, hid*directions)
        return out, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, dec_in, hidden):
        # dec_in: (batch, seq_len)  already token ids (including SOS)
        embedded = self.embed(dec_in)  # (batch, seq_len, embed_dim)
        outputs, hidden = self.lstm(embedded, hidden)  # outputs: (batch, seq_len, hidden_dim)
        logits = self.out(outputs)  # (batch, seq_len, vocab)
        return logits, hidden

def load_model(path):
        ckpt = torch.load(path, map_location=device)
        # Recreate model objects with same architecture
        # For simplicity assume same hp as before
        enc_model = Encoder(input_dim=len(ckpt["src_vocab"]),
                            embed_dim=hp["embed_size"],
                            hidden_dim=hp["hidden_size"],
                            num_layers=hp["num_layers"],
                            dropout=hp["dropout"],
                            bidirectional=hp["bidirectional_encoder"]).to(device)
        dec_model = Decoder(output_dim=len(ckpt["tgt_vocab"]),
                            embed_dim=hp["embed_size"],
                            hidden_dim=hp["hidden_size"] * (2 if hp["bidirectional_encoder"] else 1),
                            num_layers=hp["num_layers"],
                            dropout=hp["dropout"]).to(device)
        enc_model.load_state_dict(ckpt["enc_state"])
        dec_model.load_state_dict(ckpt["dec_state"])
        return enc_model.eval(), dec_model.eval(), ckpt["src_vocab"], ckpt["tgt_vocab"], ckpt["inv_tgt"]

enc_model, dec_model, saved_src_vocab, saved_tgt_vocab, saved_inv_tgt = load_model("lstm_checkpoint/v1.pt")

def infer_word(word: str, enc_model, dec_model, src_vocab, inv_tgt, hp) -> str:
    enc_model.eval()
    dec_model.eval()
    device = hp["device"]
    # encode word
    ids = [ src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower()) ]
    src_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    src_len = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        enc_out, (h_n, c_n) = enc_model(src_tensor, src_len)
        # prepare initial hidden similar to training
        if hp["bidirectional_encoder"]:
            num_layers = hp["num_layers"]
            batch_size = 1
            h_n = h_n.view(num_layers, 2, batch_size, hp["hidden_size"])
            c_n = c_n.view(num_layers, 2, batch_size, hp["hidden_size"])
            h_n = torch.cat([h_n[:,0,:,:], h_n[:,1,:,:]], dim=2)
            c_n = torch.cat([c_n[:,0,:,:], c_n[:,1,:,:]], dim=2)
            if dec_model.lstm.hidden_size != h_n.size(2):
                proj_h = nn.Linear(h_n.size(2), dec_model.lstm.hidden_size).to(device)
                proj_c = nn.Linear(c_n.size(2), dec_model.lstm.hidden_size).to(device)
                h_0 = proj_h(h_n)
                c_0 = proj_c(c_n)
            else:
                h_0, c_0 = h_n, c_n
        else:
            h_0, c_0 = h_n, c_n

        hidden = (h_0.contiguous(), c_0.contiguous())
        SOS = 1
        EOS = 2
        input_t = torch.tensor([[SOS]], dtype=torch.long, device=device)
        pred_chars = []
        for _ in range(hp["max_target_len"]):
            out, hidden = dec_model(input_t, hidden)  # (1,1,vocab)
            logits = out.squeeze(1)
            top1 = logits.argmax(dim=1).item()
            if top1 == EOS:
                break
            ch = inv_tgt.get(top1, "")
            if ch in ("<pad>", "<sos>", "<eos>"):
                if ch == "<unk>":
                    pred_chars.append("�")
                # else ignore
            else:
                pred_chars.append(ch)
            input_t = torch.tensor([[top1]], dtype=torch.long, device=device)
        return "".join(pred_chars)

def infer_sentence(sentence: str, enc_model, dec_model, src_vocab, inv_tgt, hp) -> str:
    tokens = sentence.strip().split()
    preds = []
    for tok in tokens:
        preds.append(infer_word(tok, enc_model, dec_model, src_vocab, inv_tgt, hp))
    return " ".join(preds)




# Example usage:
examples = ["namaste", "bharat", "kumar", "gargling"]
for w in examples:
    pred = infer_word(w, enc_model, dec_model, saved_src_vocab, saved_inv_tgt, hp)
    print(f"{w}  ->  {pred}")