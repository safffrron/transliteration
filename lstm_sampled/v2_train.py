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
# UTILITIES
# ================================
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j - 1] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, replace)
        prev = cur
    return prev[lb]

# ================================
# DATASET
# ================================
class TranslitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_vocab: Dict[str,int], 
                 tgt_vocab: Dict[str,int], max_tgt_len: int):
        self.srcs = df["english word"].astype(str).tolist()
        self.tgts = df["native word"].astype(str).tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.srcs)

    def encode_src(self, s: str) -> List[int]:
        return [self.src_vocab.get(ch, self.src_vocab["<unk>"]) 
                for ch in list(s.lower())]

    def encode_tgt(self, s: str) -> List[int]:
        chars = list(s.strip())[:self.max_tgt_len - 1]
        return [self.tgt_vocab.get(ch, self.tgt_vocab["<unk>"]) 
                for ch in chars]

    def __getitem__(self, idx):
        return {
            "src_raw": self.srcs[idx],
            "tgt_raw": self.tgts[idx],
            "src": torch.tensor(self.encode_src(self.srcs[idx]), dtype=torch.long),
            "tgt": torch.tensor(self.encode_tgt(self.tgts[idx]), dtype=torch.long),
        }

def collate_fn(batch):
    """Collate batch with padding."""
    PAD, SOS, EOS = 0, 1, 2
    
    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    
    max_src = max(src_lens)
    max_tgt = max(tgt_lens) + 1
    
    src_padded = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    dec_in_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    dec_target_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :s.size(0)] = s
        dec_in_padded[i, 0] = SOS
        dec_in_padded[i, 1:1+t.size(0)] = t
        dec_target_padded[i, :t.size(0)] = t
        dec_target_padded[i, t.size(0)] = EOS
    
    return {
        "src": src_padded,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "dec_in": dec_in_padded,
        "dec_target": dec_target_padded,
        "src_raws": [b["src_raw"] for b in batch],
        "tgt_raws": [b["tgt_raw"] for b in batch],
    }

# ================================
# BUILD VOCABULARIES
# ================================
def build_vocabs(train_path, valid_path, test_path):
    """Build character vocabularies from all data splits."""
    src_chars = set()
    tgt_chars = set()
    
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_json(path, lines=True)
        except:
            df = pd.read_json(path)
        
        for s in df["english word"].astype(str):
            src_chars.update(list(s.lower()))
        for t in df["native word"].astype(str):
            tgt_chars.update(list(t))
    
    def make_vocab(chars):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        for i, ch in enumerate(sorted(chars), start=4):
            vocab[ch] = i
        inv_vocab = {v: k for k, v in vocab.items()}
        return vocab, inv_vocab
    
    src_vocab, inv_src = make_vocab(src_chars)
    tgt_vocab, inv_tgt = make_vocab(tgt_chars)
    return src_vocab, inv_src, tgt_vocab, inv_tgt

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
    

# ================================
# TRAINING
# ================================
def train_epoch(enc, dec, dataloader, optim_enc, optim_dec, criterion, hp, epoch):
    enc.train()
    dec.train()
    device = hp["device"]
    total_loss = 0.0
    total_tokens = 0
    
    # Scheduled sampling: decay teacher forcing
    progress = (epoch - 1) / max(1, hp["num_epochs"] - 1)
    tf_ratio = hp["teacher_forcing_start"] + (hp["teacher_forcing_end"] - hp["teacher_forcing_start"]) * progress
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        src = batch["src"].to(device)
        src_lens = batch["src_lens"].to(device)
        dec_in = batch["dec_in"].to(device)
        dec_target = batch["dec_target"].to(device)
        batch_size = src.size(0)
        
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        
        # Encode
        enc_out, (h_n, c_n) = enc(src, src_lens)
        
        # Prepare decoder initial hidden state
        h_0, c_0 = prepare_decoder_hidden(h_n, c_n, enc, dec, device)
        
        # Decode with teacher forcing
        use_tf = random.random() < tf_ratio
        if use_tf:
            logits, _ = dec(dec_in, (h_0, c_0), enc_out, src_lens)
            loss = criterion(logits.view(-1, logits.size(-1)), dec_target.view(-1))
        else:
            # Scheduled sampling: use own predictions
            seq_len = dec_in.size(1)
            logits_list = []
            hidden = (h_0, c_0)
            input_t = dec_in[:, 0:1]
            
            for t in range(1, seq_len):
                out_t, hidden = dec(input_t, hidden, enc_out, src_lens)
                logits_list.append(out_t)
                input_t = out_t.argmax(dim=-1)
            
            if logits_list:
                logits = torch.cat(logits_list, dim=1)
                loss = criterion(logits.view(-1, logits.size(-1)), 
                               dec_target[:, :seq_len-1].contiguous().view(-1))
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), 
            max_norm=hp["grad_clip"]
        )
        optim_enc.step()
        optim_dec.step()
        
        total_loss += loss.item() * batch_size
        total_tokens += batch_size
    
    return total_loss / max(1, total_tokens), tf_ratio

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

# ================================
# EVALUATION
# ================================
def evaluate(enc, dec, dataloader, inv_tgt, hp):
    """Evaluate model on validation/test set."""
    enc.eval()
    dec.eval()
    device = hp["device"]
    
    n_samples = 0
    n_word_correct = 0
    total_lev = 0
    total_lev_sim = 0.0
    total_ref_chars = 0
    total_matching_chars = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch["src"].to(device)
            src_lens = batch["src_lens"].to(device)
            tgt_raws = batch["tgt_raws"]
            batch_size = src.size(0)
            
            # Encode
            enc_out, (h_n, c_n) = enc(src, src_lens)
            h_0, c_0 = prepare_decoder_hidden(h_n, c_n, enc, dec, device)
            
            # Greedy decode
            SOS, EOS = 1, 2
            max_len = hp["max_target_len"]
            input_t = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
            hidden = (h_0, c_0)
            decoded_ids = [[] for _ in range(batch_size)]
            
            for _ in range(max_len):
                out, hidden = dec(input_t, hidden, enc_out, src_lens)
                logits = out[:, -1, :]
                top1 = logits.argmax(dim=1)
                input_t = top1.unsqueeze(1)
                
                for i, token in enumerate(top1.cpu().numpy()):
                    decoded_ids[i].append(int(token))
            
            # Convert to strings
            for i in range(batch_size):
                pred_ids = decoded_ids[i]
                if EOS in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(EOS)]
                
                pred_chars = []
                for pid in pred_ids:
                    ch = inv_tgt.get(pid, "")
                    if ch not in ("<pad>", "<sos>", "<eos>"):
                        pred_chars.append(ch if ch != "<unk>" else "")
                
                pred = "".join(pred_chars)
                ref = tgt_raws[i]
                
                n_samples += 1
                if pred == ref:
                    n_word_correct += 1
                
                # Levenshtein
                d = levenshtein(pred, ref)
                total_lev += d
                maxlen = max(1, max(len(pred), len(ref)))
                total_lev_sim += 1.0 - (d / maxlen)
                
                # Character-level matching
                match_chars = sum(1 for a, b in zip(pred, ref) if a == b)
                total_matching_chars += match_chars
                total_ref_chars += len(ref)
    
    word_acc = n_word_correct / max(1, n_samples)
    avg_lev_sim = total_lev_sim / max(1, n_samples)
    char_match_rate = total_matching_chars / max(1, total_ref_chars)
    cer = total_lev / max(1, total_ref_chars)
    
    return word_acc, avg_lev_sim, char_match_rate, cer

# ================================
# BEAM SEARCH INFERENCE
# ================================
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


# ================================
# MAIN TRAINING LOOP
# ================================
if __name__ == "__main__":
    hp = HP
    device = torch.device(hp["device"])
    print(f"Device: {device}")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}\n")
    
    # Load data
    def read_json_or_jsonl(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        try:
            return pd.read_json(path, lines=True)
        except:
            return pd.read_json(path)
    
    print("Loading datasets...")
    df_train = read_json_or_jsonl(hp["train_path"])
    df_valid = read_json_or_jsonl(hp["valid_path"])
    df_test = read_json_or_jsonl(hp["test_path"])
    print(f"Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}\n")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab, inv_src, tgt_vocab, inv_tgt = build_vocabs(
        hp["train_path"], hp["valid_path"], hp["test_path"]
    )
    print(f"Source vocab: {len(src_vocab):,} | Target vocab: {len(tgt_vocab):,}\n")
    
    # Create datasets and dataloaders
    train_ds = TranslitDataset(df_train, src_vocab, tgt_vocab, hp["max_target_len"])
    valid_ds = TranslitDataset(df_valid, src_vocab, tgt_vocab, hp["max_target_len"])
    test_ds = TranslitDataset(df_test, src_vocab, tgt_vocab, hp["max_target_len"])
    
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], 
                             shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=hp["batch_size"], 
                             shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=hp["batch_size"], 
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Build models
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
    
    optim_enc = torch.optim.Adam(enc.parameters(), lr=hp["learning_rate"])
    optim_dec = torch.optim.Adam(dec.parameters(), lr=hp["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler
    scheduler_enc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_enc, mode='max', factor=0.5, patience=2, verbose=True
    )
    scheduler_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_dec, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    best_valid_metric = -1.0
    patience_counter = 0
    max_patience = 4
    
    print("Starting training...\n")
    for epoch in range(1, hp["num_epochs"] + 1):
        train_loss, tf_ratio = train_epoch(
            enc, dec, train_loader, optim_enc, optim_dec, criterion, hp, epoch
        )
        print(f"\nEpoch {epoch}/{hp['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | TF Ratio: {tf_ratio:.3f}")
        
        # Validation
        word_acc, lev_sim, char_match, cer = evaluate(enc, dec, valid_loader, inv_tgt, hp)
        print(f"Valid -> Word Acc: {word_acc:.4f} | LevSim: {lev_sim:.4f} | "
              f"CharMatch: {char_match:.4f} | CER: {cer:.4f}")
        
        # Update learning rate
        valid_metric = word_acc + lev_sim
        scheduler_enc.step(valid_metric)
        scheduler_dec.step(valid_metric)
        
        # Save best model
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            patience_counter = 0
            torch.save({
                "enc_state": enc.state_dict(),
                "dec_state": dec.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "inv_tgt": inv_tgt,
                "hp": hp
            }, hp["save_path"])
            print(f"✓ Saved best model (metric={valid_metric:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # ================================
    # FINAL TEST EVALUATION
    # ================================
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    ckpt = torch.load(hp["save_path"], map_location=device)
    enc.load_state_dict(ckpt["enc_state"])
    dec.load_state_dict(ckpt["dec_state"])
    
    word_acc, lev_sim, char_match, cer = evaluate(enc, dec, test_loader, inv_tgt, hp)
    print(f"\nTest Results:")
    print(f"  Word Accuracy: {word_acc:.4f} ({word_acc*100:.2f}%)")
    print(f"  Levenshtein Similarity: {lev_sim:.4f}")
    print(f"  Character Match Rate: {char_match:.4f}")
    print(f"  Character Error Rate (CER): {cer:.4f}")
    print(f"  1 - CER: {1-cer:.4f}")
    
    # ================================
    # INFERENCE EXAMPLES
    # ================================
    print("\n" + "="*60)
    print("INFERENCE EXAMPLES (Beam Search)")
    print("="*60)
    
    examples = [
        "namaste",
        "bharat",
        "kumar",
        "gargling",
        "delhi",
        "maharashtra",
        "rajasthan",
        "bengaluru",
        "sanskrit",
        "himalaya"
    ]
    
    for word in examples:
        pred = infer_word(word, enc, dec, src_vocab, inv_tgt, hp)
        print(f"{word:15s} -> {pred}")
    
    print("\n" + "="*60)
    print("SENTENCE INFERENCE")
    print("="*60)
    
    sentences = [
        "namaste bharat",
        "jai hind",
        "kumar singh"
    ]
    
    for sent in sentences:
        pred = infer_sentence(sent, enc, dec, src_vocab, inv_tgt, hp)
        print(f"{sent:20s} -> {pred}")
    
    print("\n" + "="*60)
    print("Training complete! Best model saved to:", hp["save_path"])
    print("="*60)