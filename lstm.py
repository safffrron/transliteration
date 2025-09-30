from preprocess import get_loaders

train_loader, val_loader, test_loader, input_stoi, target_stoi, input_itos, target_itos = get_loaders(batch_size=128,
                                                                                                     train_file="/kaggle/input/hindi-translit/hin_train.json",
                                                                                                     val_file="/kaggle/input/hindi-translit/hin_valid.json",
                                                                                                     test_file="/kaggle/input/hindi-translit/hin_test.json")



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
import json
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        x = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            x = trg[:, t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs




# Hyperparams
embed_size = 256
hidden_size = 512
num_layers = 2
batch_size = 128
learning_rate = 0.001
epochs = 20


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(len(input_stoi), embed_size, hidden_size, num_layers).to(device)
decoder = Decoder(len(target_stoi), embed_size, hidden_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Optimizer + Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

from tqdm.notebook import tqdm
import sys

best_val_loss = float("inf")

for epoch in range(epochs):
    # -------------------
    # Training
    # -------------------
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                     leave=False, file=sys.stdout)

    for src, trg in train_bar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
        sys.stdout.flush()   # ✅ force refresh

    train_loss /= len(train_loader)

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                   leave=False, file=sys.stdout)

    with torch.no_grad():
        for src, trg in val_bar:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            val_loss += loss.item()
            val_bar.set_postfix(loss=f"{loss.item():.4f}")
            sys.stdout.flush()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    sys.stdout.flush()

    # -------------------
    # Save Best Model
    # -------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "input_stoi": input_stoi,
            "input_itos": input_itos,
            "target_stoi": target_stoi,
            "target_itos": target_itos
        }, "best_transliteration_model.pth")
        print("✅ Saved new best model")
        sys.stdout.flush()

from nltk.translate.bleu_score import sentence_bleu
import torch

def evaluate_model(model, dataloader, input_itos, target_itos, criterion, device):
    model.eval()
    total_loss = 0
    total_chars, correct_chars = 0, 0
    total_words, correct_words = 0, 0
    bleu_scores = []

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0.0)  # no teacher forcing at eval
            output_dim = output.shape[-1]

            # reshape for loss
            output = output[:,1:].reshape(-1, output_dim)  # skip SOS
            trg = trg[:,1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

            # Predictions
            pred_tokens = output.argmax(1).view(src.size(0), -1)  # batch_size x seq_len
            trg_tokens = trg.view(src.size(0), -1)

            for i in range(pred_tokens.size(0)):
                pred_seq = [target_itos[idx.item()] for idx in pred_tokens[i] if idx.item() != 0]
                true_seq = [target_itos[idx.item()] for idx in trg_tokens[i] if idx.item() != 0]

                # Character-level accuracy
                min_len = min(len(pred_seq), len(true_seq))
                correct_chars += sum(p == t for p, t in zip(pred_seq[:min_len], true_seq[:min_len]))
                total_chars += len(true_seq)

                # Word-level accuracy
                if pred_seq == true_seq:
                    correct_words += 1
                total_words += 1

                # BLEU
                bleu_scores.append(sentence_bleu([true_seq], pred_seq, weights=(0.5, 0.5)))

    avg_loss = total_loss / len(dataloader)
    char_acc = correct_chars / total_chars
    word_acc = correct_words / total_words
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_loss, char_acc, word_acc, avg_bleu





# Must re-create model objects with same architecture
encoder = Encoder(len(input_stoi), embed_size, hidden_size, num_layers).to(device)
decoder = Decoder(len(target_stoi), embed_size, hidden_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load checkpoint
checkpoint = torch.load("/kaggle/working/best_transliteration_model.pth", map_location=device)

encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

# Restore vocabs
input_stoi = checkpoint["input_stoi"]
input_itos = checkpoint["input_itos"]
target_stoi = checkpoint["target_stoi"]
target_itos = checkpoint["target_itos"]

print("Model loaded ✅")

criterion = nn.CrossEntropyLoss(ignore_index=0) 
test_loss, char_acc, word_acc, bleu = evaluate_model(
    model, test_loader, input_itos, target_itos, criterion, device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Character Accuracy: {char_acc*100:.2f}%")
print(f"Word Accuracy: {word_acc*100:.2f}%")
print(f"BLEU Score: {bleu:.4f}")
