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


# ---------------------------
# UTIL: Levenshtein (for char-level eval)
# ---------------------------
def levenshtein(a: str, b: str) -> int:
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

# ---------------------------
# DATASET
# ---------------------------
class TranslitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_vocab: Dict[str,int], tgt_vocab: Dict[str,int], max_tgt_len:int):
        """
        df must have columns: 'english word' and 'native word'
        """
        self.srcs = df["english word"].astype(str).tolist()
        self.tgts = df["native word"].astype(str).tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.srcs)

    def encode_src(self, s: str) -> List[int]:
        # lowercase roman input
        return [ self.src_vocab.get(ch, self.src_vocab["<unk>"]) for ch in list(s.lower()) ]

    def encode_tgt(self, s: str) -> List[int]:
        # target is Devanagari; preserve characters as-is
        chars = list(s.strip())
        chars = chars[: (self.max_tgt_len - 1)]  # reserve space for <eos>
        token_ids = [ self.tgt_vocab.get(ch, self.tgt_vocab["<unk>"]) for ch in chars ]
        return token_ids

    def __getitem__(self, idx):
        src_raw = self.srcs[idx]
        tgt_raw = self.tgts[idx]

        src_ids = self.encode_src(src_raw)
        tgt_ids = self.encode_tgt(tgt_raw)
        return {
            "src_raw": src_raw,
            "tgt_raw": tgt_raw,
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }

def collate_fn(batch):
    # Pads sequences and creates decoder inputs and targets
    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    src_raws = [b["src_raw"] for b in batch]
    tgt_raws = [b["tgt_raw"] for b in batch]

    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens) + 1  # +1 for EOS in target sequence

    PAD = 0
    SOS = 1
    EOS = 2

    src_padded = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    # decoder input starts with SOS
    dec_in_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    dec_target_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :s.size(0)] = s
        # decoder input: <sos> t[0..]
        dec_in_padded[i, 0] = SOS
        dec_in_padded[i, 1:1+t.size(0)] = t
        # decoder target: t[0..] <eos>
        dec_target_padded[i, :t.size(0)] = t
        dec_target_padded[i, t.size(0)] = EOS

    return {
        "src": src_padded,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "dec_in": dec_in_padded,
        "dec_target": dec_target_padded,
        "src_raws": src_raws,
        "tgt_raws": tgt_raws,
    }


# ---------------------------
# BUILD VOCABS (from train+valid+test to reduce OOV)
# ---------------------------
def build_vocabs(train_path, valid_path, test_path):
    def collect_chars(paths):
        chars = set()
        for p in paths:
            if not os.path.exists(p):
                continue
            # try jsonl or json
            try:
                df = pd.read_json(p, lines=True)
            except Exception:
                df = pd.read_json(p)
            # english roman
            for s in df["english word"].astype(str).tolist():
                chars.update(list(s.lower()))
            # devanagari
            for t in df["native word"].astype(str).tolist():
                chars.update(list(t))
        return chars

    # Source (roman) chars
    src_chars = set()
    tgt_chars = set()
    for p in [train_path, valid_path, test_path]:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_json(p, lines=True)
        except Exception:
            df = pd.read_json(p)
        for s in df["english word"].astype(str).tolist():
            src_chars.update(list(s.lower()))
        for t in df["native word"].astype(str).tolist():
            tgt_chars.update(list(t))

    # build maps
    # Special tokens: PAD=0, SOS=1, EOS=2, UNK=3
    def make_vocab(chars):
        idx = 4
        vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        for ch in sorted(chars):
            if ch in vocab:
                continue
            vocab[ch] = idx
            idx += 1
        inv = {v:k for k,v in vocab.items()}
        return vocab, inv

    src_vocab, inv_src = make_vocab(src_chars)
    tgt_vocab, inv_tgt = make_vocab(tgt_chars)
    return src_vocab, inv_src, tgt_vocab, inv_tgt


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
    

# ---------------------------
# TRAIN / EVAL FUNCTIONS
# ---------------------------
def train_epoch(model_enc, model_dec, dataloader, optim_enc, optim_dec, criterion, hp):
    model_enc.train()
    model_dec.train()
    device = hp["device"]
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        src = batch["src"].to(device)
        src_lens = batch["src_lens"].to(device)
        dec_in = batch["dec_in"].to(device)
        dec_target = batch["dec_target"].to(device)
        batch_size = src.size(0)

        optim_enc.zero_grad()
        optim_dec.zero_grad()

        enc_out, (h_n, c_n) = model_enc(src, src_lens)

        # Prepare initial decoder hidden state:
        # If encoder is bidirectional, combine forward & backward states:
        if hp["bidirectional_encoder"]:
            # h_n: (num_layers*2, batch, hidden)
            # we need to reduce to (num_layers, batch, hidden) to feed decoder.
            num_layers = hp["num_layers"]
            # split and concat
            h_n = h_n.view(num_layers, 2, batch_size, hp["hidden_size"])
            c_n = c_n.view(num_layers, 2, batch_size, hp["hidden_size"])
            # concat forward & backward along hidden dim
            h_n = torch.cat([h_n[:,0,:,:], h_n[:,1,:,:]], dim=2)  # (num_layers, batch, hidden*2)
            c_n = torch.cat([c_n[:,0,:,:], c_n[:,1,:,:]], dim=2)
            # if decoder hidden_size != hidden_size*2 we project
            if model_dec.lstm.hidden_size != h_n.size(2):
                # project to decoder hidden size
                proj_h = nn.Linear(h_n.size(2), model_dec.lstm.hidden_size).to(device)
                proj_c = nn.Linear(c_n.size(2), model_dec.lstm.hidden_size).to(device)
                h_0 = proj_h(h_n)
                c_0 = proj_c(c_n)
            else:
                h_0, c_0 = h_n, c_n
        else:
            # encoder not bidir => shapes match
            h_0, c_0 = h_n, c_n

        # Teacher forcing step-by-step:
        seq_len = dec_in.size(1)
        logits_all = []
        hidden = (h_0.contiguous(), c_0.contiguous())

        use_teacher_forcing = (random.random() < hp["teacher_forcing_ratio"])

        if use_teacher_forcing:
            # compute logits for whole decoder input at once
            logits, _ = model_dec(dec_in, hidden)  # (batch, seq, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), dec_target.view(-1))
        else:
            # greedy decode one step at a time feeding previous prediction
            batch_logits = []
            input_t = dec_in[:, 0].unsqueeze(1)  # initial = SOS
            for t in range(1, seq_len):
                out_t, hidden = model_dec(input_t, hidden)  # out_t: (batch, 1, vocab)
                logits_t = out_t.squeeze(1)
                batch_logits.append(logits_t)
                # greedy next token
                top1 = logits_t.argmax(dim=1)
                input_t = top1.unsqueeze(1)
            # stack logits (seq_len-1 steps); we need to align with dec_target[:,:seq_len-1]
            if len(batch_logits) == 0:
                # degenerate case (seq len 1)
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                logits_cat = torch.stack(batch_logits, dim=1)  # (batch, seq-1, vocab)
                # pad to seq_len with zeros at front to match dec_target alignment with EOS at position t
                # We will compare logits_cat to dec_target[:, :seq_len-1]
                loss = criterion(logits_cat.view(-1, logits_cat.size(-1)), dec_target[:, :seq_len-1].contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model_enc.parameters()) + list(model_dec.parameters()), max_norm=1.0)
        optim_enc.step()
        optim_dec.step()

        total_loss += loss.item() * batch["src"].size(0)
        total_tokens += batch["src"].size(0)

    return total_loss / max(1, total_tokens)

def evaluate(model_enc, model_dec, dataloader, hp):
    # Returns: word_acc, char_levenshtein_score_mean, char_match_rate
    model_enc.eval()
    model_dec.eval()
    device = hp["device"]

    n_samples = 0
    n_word_correct = 0
    total_lev_similarity = 0.0
    total_ref_chars = 0
    total_matching_chars = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            src = batch["src"].to(device)
            src_lens = batch["src_lens"].to(device)
            dec_in = batch["dec_in"].to(device)
            dec_target = batch["dec_target"].to(device)
            src_raws = batch["src_raws"]
            tgt_raws = batch["tgt_raws"]
            batch_size = src.size(0)

            enc_out, (h_n, c_n) = model_enc(src, src_lens)

            # prepare initial decoder hidden similar to training
            if hp["bidirectional_encoder"]:
                num_layers = hp["num_layers"]
                h_n = h_n.view(num_layers, 2, batch_size, hp["hidden_size"])
                c_n = c_n.view(num_layers, 2, batch_size, hp["hidden_size"])
                h_n = torch.cat([h_n[:,0,:,:], h_n[:,1,:,:]], dim=2)
                c_n = torch.cat([c_n[:,0,:,:], c_n[:,1,:,:]], dim=2)
                if model_dec.lstm.hidden_size != h_n.size(2):
                    proj_h = nn.Linear(h_n.size(2), model_dec.lstm.hidden_size).to(device)
                    proj_c = nn.Linear(c_n.size(2), model_dec.lstm.hidden_size).to(device)
                    h_0 = proj_h(h_n)
                    c_0 = proj_c(c_n)
                else:
                    h_0, c_0 = h_n, c_n
            else:
                h_0, c_0 = h_n, c_n

            hidden = (h_0.contiguous(), c_0.contiguous())

            # Greedy decode until EOS or max length
            SOS = 1
            EOS = 2
            max_out_len = hp["max_target_len"]
            input_t = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
            decoded_ids = [[] for _ in range(batch_size)]

            for t in range(max_out_len):
                out, hidden = model_dec(input_t, hidden)  # out: (batch, 1, vocab)
                logits = out.squeeze(1)  # (batch, vocab)
                top1 = logits.argmax(dim=1)  # (batch,)
                input_t = top1.unsqueeze(1)
                for i, token in enumerate(top1.cpu().numpy().tolist()):
                    decoded_ids[i].append(int(token))

            # convert ids to strings using inverse vocab (we will pass inv_tgt from outer scope)
            # We'll return later; for now collect predicted and ref pairs
            for i in range(batch_size):
                # detach predicted until EOS
                pred_ids = decoded_ids[i]
                # cut at EOS if present
                if EOS in pred_ids:
                    eos_idx = pred_ids.index(EOS)
                    pred_ids = pred_ids[:eos_idx]
                # map ids to chars via global inv_tgt, defined later by closure (okay)
                # We'll append to lists below
                n_samples += 1

                # raw reference:
                ref = tgt_raws[i]

                # convert pred ids to string (we will fill inv_tgt in outer scope)
                # to be replaced by actual text at the end when inv_tgt is available
                # store as tuple for now
                batch["pred_ids_{}".format(i)] = pred_ids

            # Now convert batched predictions using inv_tgt mapping
            # (we cheat a bit: use inv_tgt available in outer scope)
            for i in range(batch_size):
                pred_ids = batch["pred_ids_{}".format(i)]
                pred_chars = []
                for pid in pred_ids:
                    ch = inv_tgt.get(pid, "")
                    # ignore special tokens or PAD if any
                    if ch in ("<pad>", "<sos>", "<eos>", "<unk>") :
                        # if <unk>, we can leave blank or mark as '?'
                        if ch == "<unk>":
                            pred_chars.append("�")
                        continue
                    pred_chars.append(ch)
                pred = "".join(pred_chars)
                ref = batch["tgt_raws"][i]

                # word accuracy
                if pred == ref:
                    n_word_correct += 1

                # levenshtein similarity normalized
                d = levenshtein(pred, ref)
                maxlen = max(1, max(len(pred), len(ref)))
                sim = 1.0 - (d / maxlen)
                total_lev_similarity += sim

                # per-character matching count (align naive by position)
                match_chars = sum(1 for a,b in zip(pred, ref) if a==b)
                total_matching_chars += match_chars
                total_ref_chars += len(ref)

    word_acc = n_word_correct / max(1, n_samples)
    avg_lev_sim = total_lev_similarity / max(1, n_samples)
    char_match_rate = total_matching_chars / max(1, total_ref_chars) if total_ref_chars>0 else 0.0

    return word_acc, avg_lev_sim, char_match_rate



# ---------------------------
# MAIN: load data, build vocabs, train
# ---------------------------
if __name__ == "__main__":
    hp = HP
    device = torch.device(hp["device"])
    print("Device:", device)

    # Load dataframes
    def read_maybe_jsonl(p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")
        try:
            return pd.read_json(p, lines=True)
        except Exception:
            return pd.read_json(p)

    print("Loading datasets...")
    df_train = read_maybe_jsonl(hp["train_path"])
    df_valid = read_maybe_jsonl(hp["valid_path"])
    df_test = read_maybe_jsonl(hp["test_path"])
    # Keep only required columns
    for df in (df_train, df_valid, df_test):
        if "english word" not in df.columns or "native word" not in df.columns:
            raise KeyError("Input JSONs must contain 'english word' and 'native word' columns")

    print(f"Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")

    print("Building vocabularies (this uses chars from train+valid+test)...")
    src_vocab, inv_src, tgt_vocab, inv_tgt = build_vocabs(hp["train_path"], hp["valid_path"], hp["test_path"])
    print(f"Source vocab size: {len(src_vocab):,} ; Target vocab size: {len(tgt_vocab):,}")

    # Expose inv_tgt to eval closure scope
    globals()["inv_tgt"] = inv_tgt

    # Create datasets
    train_ds = TranslitDataset(df_train, src_vocab, tgt_vocab, max_tgt_len=hp["max_target_len"])
    valid_ds = TranslitDataset(df_valid, src_vocab, tgt_vocab, max_tgt_len=hp["max_target_len"])
    test_ds  = TranslitDataset(df_test,  src_vocab, tgt_vocab, max_tgt_len=hp["max_target_len"])

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=hp["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=hp["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Instantiate models
    input_dim = len(src_vocab)
    output_dim = len(tgt_vocab)
    enc = Encoder(input_dim=input_dim,
                  embed_dim=hp["embed_size"],
                  hidden_dim=hp["hidden_size"],
                  num_layers=hp["num_layers"],
                  dropout=hp["dropout"],
                  bidirectional=hp["bidirectional_encoder"]).to(device)

    # If encoder is bidirectional, decoder hidden_size must match encoder hidden*2 (we project inside train)
    dec_hidden_size = hp["hidden_size"] * (2 if hp["bidirectional_encoder"] else 1)
    # But to keep decoder capacity reasonable, we set decoder.hidden_size = hp["hidden_size"] * (2 if bidir else 1)
    # and projection will happen if shapes mismatch.
    dec = Decoder(output_dim=output_dim,
                  embed_dim=hp["embed_size"],
                  hidden_dim=hp["hidden_size"] * (2 if hp["bidirectional_encoder"] else 1),
                  num_layers=hp["num_layers"],
                  dropout=hp["dropout"]).to(device)

    optim_enc = torch.optim.Adam(enc.parameters(), lr=hp["learning_rate"])
    optim_dec = torch.optim.Adam(dec.parameters(), lr=hp["learning_rate"])
    # ignore PAD (0) when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop with validation & model saving
    best_valid_metric = -1.0
    for epoch in range(1, hp["num_epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{hp['num_epochs']} =====")
        train_loss = train_epoch(enc, dec, train_loader, optim_enc, optim_dec, criterion, hp)
        print(f"Train loss: {train_loss:.4f}")

        print("Valid evaluation...")
        word_acc, avg_lev_sim, char_match_rate = evaluate(enc, dec, valid_loader, hp)
        print(f"Valid Word Acc: {word_acc:.4f}  |  Valid Avg LevSim: {avg_lev_sim:.4f}  |  CharMatchRate: {char_match_rate:.4f}")

        # choose best by word accuracy + levsim (simple sum)
        valid_metric = word_acc + avg_lev_sim
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            torch.save({
                "enc_state": enc.state_dict(),
                "dec_state": dec.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "inv_tgt": inv_tgt,
                "hp": hp
            }, hp["save_path"])
            print(f"Saved best model to {hp['save_path']} (metric={valid_metric:.4f})")

    # Final evaluation on test set
    print("\n=== Loading best model and evaluating on test set ===")
    ckpt = torch.load(hp["save_path"], map_location=device)
    enc.load_state_dict(ckpt["enc_state"])
    dec.load_state_dict(ckpt["dec_state"])
    print("Running evaluation on test set ...")
    word_acc, avg_lev_sim, char_match_rate = evaluate(enc, dec, test_loader, hp)
    print(f"Test Word Acc: {word_acc:.4f}")
    print(f"Test Avg Levenshtein Similarity: {avg_lev_sim:.4f}")
    print(f"Test Char Match Rate (position-wise): {char_match_rate:.4f}")

    # ---------------------------
    # INFERENCE FUNCTION (single word or sentence)
    # ---------------------------
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

    enc_model, dec_model, saved_src_vocab, saved_tgt_vocab, saved_inv_tgt = load_model(hp["save_path"])

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

    print("Done.")