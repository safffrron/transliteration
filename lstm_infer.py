from preprocess import get_loaders

train_loader, val_loader, test_loader, input_stoi, target_stoi, input_itos, target_itos = get_loaders(batch_size=128,
                                                                                                     train_file="../hin//hin_train.json",
                                                                                                     val_file="../hin//hin_valid.json",
                                                                                                     test_file="../hin//hin_test.json")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
import json
import random


# Hyperparams
embed_size = 256
hidden_size = 512
num_layers = 2
batch_size = 128
learning_rate = 0.001
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Must re-create model objects with same architecture
encoder = Encoder(len(input_stoi), embed_size, hidden_size, num_layers).to(device)
decoder = Decoder(len(target_stoi), embed_size, hidden_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load checkpoint
checkpoint = torch.load("lstm_checkpoint/model_1.pth", map_location=device)

encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

# Restore vocabs
input_stoi = checkpoint["input_stoi"]
input_itos = checkpoint["input_itos"]
target_stoi = checkpoint["target_stoi"]
target_itos = checkpoint["target_itos"]

print("Model loaded ✅")


def translate_word(model, word, max_len=30):
    model.eval()
    with torch.no_grad():
        x = [input_stoi["<sos>"]] + [input_stoi[ch] for ch in word] + [input_stoi["<eos>"]]
        x = x + [0]*(max_len-len(x))
        x = torch.tensor(x).unsqueeze(0).to(device)

        hidden, cell = model.encoder(x)
        outputs = []
        next_token = torch.tensor([target_stoi["<sos>"]]).to(device)

        for _ in range(max_len):
            prediction, hidden, cell = model.decoder(next_token, hidden, cell)
            next_token = prediction.argmax(1)
            if next_token.item() == target_stoi["<eos>"]:
                break
            outputs.append(target_itos[next_token.item()])

    return "".join(outputs)

sentence = input()
words = sentence.split()
transliterated = [translate_word(model, w) for w in words]
print(" ".join(transliterated))