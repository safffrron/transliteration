

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
import json
import random


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