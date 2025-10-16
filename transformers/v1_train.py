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

# ================================
# LEARNING RATE SCHEDULER
# ================================
class TransformerLRScheduler:
    """Learning rate scheduler with warmup for transformers."""
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
    
    def get_last_lr(self):
        return [self._get_lr()]
    

# ================================
# TRAINING
# ================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, hp, epoch):
    model.train()
    device = hp["device"]
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        src = batch["src"].to(device)
        src_lens = batch["src_lens"]
        dec_in = batch["dec_in"].to(device)
        dec_target = batch["dec_target"].to(device)
        
        # Create padding masks
        src_pad_mask = (src == 0)
        tgt_pad_mask = (dec_in == 0)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(src, dec_in, 
                      src_key_padding_mask=src_pad_mask,
                      tgt_key_padding_mask=tgt_pad_mask)
        
        # Compute loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            dec_target.reshape(-1)
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp["grad_clip"])
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * src.size(0)
        total_tokens += src.size(0)
    
    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, scheduler.get_last_lr()[0]

# ================================
# EVALUATION
# ================================
def evaluate(model, dataloader, inv_tgt, hp):
    """Evaluate model on validation/test set."""
    model.eval()
    device = hp["device"]
    
    n_samples = 0
    n_word_correct = 0
    total_lev = 0
    total_lev_sim = 0.0
    total_ref_chars = 0
    total_matching_chars = 0
    
    PAD, SOS, EOS = 0, 1, 2
    max_len = hp["max_target_len"]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch["src"].to(device)
            tgt_raws = batch["tgt_raws"]
            batch_size = src.size(0)
            
            # Greedy decode
            src_pad_mask = (src == 0)
            
            # Initialize decoder input with SOS
            dec_input = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
            
            for _ in range(max_len):
                tgt_pad_mask = (dec_input == 0)
                
                # Forward pass
                logits = model(src, dec_input,
                             src_key_padding_mask=src_pad_mask,
                             tgt_key_padding_mask=tgt_pad_mask)
                
                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                dec_input = torch.cat([dec_input, next_token], dim=1)
                
                # Check if all sequences have generated EOS
                if (next_token == EOS).all():
                    break
            
            # Convert predictions to strings
            for i in range(batch_size):
                pred_ids = dec_input[i, 1:].cpu().numpy()  # Skip SOS
                
                # Stop at EOS
                if EOS in pred_ids:
                    pred_ids = pred_ids[:list(pred_ids).index(EOS)]
                
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

def infer_sentence(sentence: str, model, src_vocab, inv_tgt, hp) -> str:
    """Inference on a sentence (space-separated words)."""
    tokens = sentence.strip().split()
    preds = [infer_word(tok, model, src_vocab, inv_tgt, hp) for tok in tokens]
    return " ".join(preds)

# ================================
# MAIN TRAINING LOOP
# ================================
if __name__ == "__main__":
    hp = HP
    device = torch.device(hp["device"])
    print(f"Device: {device}")
    print(f"Hyperparameters:\n{json.dumps(hp, indent=2)}\n")
    
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
    
    # Build model
    print("Building Transformer model...")
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
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], 
                                betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, hp["d_model"], hp["warmup_steps"])
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=hp["label_smoothing"])
    
    # Training loop
    best_valid_metric = -1.0
    patience_counter = 0
    max_patience = 5
    
    print("Starting training...\n")
    for epoch in range(1, hp["num_epochs"] + 1):
        train_loss, current_lr = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, hp, epoch
        )
        print(f"\nEpoch {epoch}/{hp['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
        
        # Validation every epoch
        word_acc, lev_sim, char_match, cer = evaluate(model, valid_loader, inv_tgt, hp)
        print(f"Valid -> Word Acc: {word_acc:.4f} | LevSim: {lev_sim:.4f} | "
              f"CharMatch: {char_match:.4f} | CER: {cer:.4f}")
        
        # Save best model
        valid_metric = word_acc + lev_sim
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
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
    model.load_state_dict(ckpt["model_state"])
    
    word_acc, lev_sim, char_match, cer = evaluate(model, test_loader, inv_tgt, hp)
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
        "himalaya",
        "gandhi",
        "nehru",
        "india",
        "mumbai",
        "chennai"
    ]
    
    for word in examples:
        pred = infer_word(word, model, src_vocab, inv_tgt, hp)
        print(f"{word:15s} -> {pred}")
    
    print("\n" + "="*60)
    print("SENTENCE INFERENCE")
    print("="*60)
    
    sentences = [
        "namaste bharat",
        "jai hind",
        "kumar singh",
        "delhi mumbai",
        "ram krishna"
    ]
    
    for sent in sentences:
        pred = infer_sentence(sent, model, src_vocab, inv_tgt, hp)
        print(f"{sent:20s} -> {pred}")
    
    print("\n" + "="*60)
    print("Training complete! Best model saved to:", hp["save_path"])
    print("="*60)